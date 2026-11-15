"""Personal data that a producer adds by mistake never leaves the worker (spec §04, §13).

A marker is placed where a careless producer would put a class name or a teacher signature.
It must not appear in stdout/stderr, exception texts, DLQ messages or outgoing events.
"""

import json
import logging
from typing import Any

import pytest
from kafka_support import (
    SUBSTITUTION_DLQ_TOPIC,
    SUBSTITUTION_TOPIC,
    FakeEngine,
    Journal,
    Json,
    build_request,
    encode,
    harness,
    inbound,
    substitution_request,
    with_data,
)
from pydantic import ValidationError

from lad_solver.adapters.kafka import jsonlog
from lad_solver.adapters.kafka.payload import BuildRequested

MARKER = "Іваненко-MARKER-7f3a"  # Cyrillic on purpose: non-ASCII must not slip through
# The DLQ copy is ASCII-escaped JSON: the marker must not survive in that form either.
ESCAPED = json.dumps(MARKER).strip('"').encode()


def _with_personal_data(document: Json) -> Json:
    def add(data: Json) -> None:
        data["classes"][0]["name"] = MARKER
        data["teachers"][0]["signature"] = MARKER
        data["teacher_signatures"] = {"t-1": MARKER}

    return with_data(document, add)


def _invalid(data: Json) -> None:
    data["classes"][1]["students"] = -5


def _no_job_id(data: Json) -> None:
    data.pop("job_id")


def _in_teacher_id(data: Json) -> None:
    data["teachers"][0]["id"] = MARKER  # a signature typed into an id field


def _in_kind(data: Json) -> None:
    data["kind"] = MARKER


def _in_school_days(data: Json) -> None:
    data["school_days"] = [1, MARKER]


def _in_contract_fields(document: Json) -> Json:
    """Personal data in fields the contract knows: the field name alone must not let it pass."""
    changed = with_data(document, _in_teacher_id)
    changed["source"] = MARKER
    return changed


def _without_school_id(document: Json) -> Json:
    changed = dict(document)
    changed.pop("school_id")
    return changed


CASES: dict[str, Json] = {
    "accepted": _with_personal_data(build_request()),
    "rejected": with_data(_with_personal_data(build_request()), _invalid),
    "poison": with_data(_with_personal_data(build_request()), _no_job_id),
    "expired": _with_personal_data(build_request(queue_deadline_at="2026-11-01T00:00:00Z")),
    "rejected-teacher-id": _in_contract_fields(build_request()),
    "rejected-kind": with_data(build_request(), _in_kind),
    "rejected-school-days": with_data(build_request(), _in_school_days),
    "poison-contract-fields": _without_school_id(
        with_data(_in_contract_fields(build_request()), _in_kind)
    ),
}


def _substitution_with_personal_data(document: Json) -> Json:
    """Names where a careless producer would add them: fields the contract does not know."""

    def add(data: Json) -> None:
        data["absent_teacher_name"] = MARKER
        data["lessons"][0]["teacher_signature"] = MARKER
        data["teachers"][0]["signature"] = MARKER
        data["absences"] = [
            {"teacher_id": "t-1", "date": "2026-11-11", "lesson_nos": None, "reason": MARKER}
        ]

    changed = with_data(document, add)
    changed["source"] = MARKER
    return changed


def _in_absent_teacher_id(document: Json) -> Json:
    """A signature typed into the absent teacher's id: an invalid id, so the job is rejected."""
    return with_data(document, lambda d: d.update(absent_teacher_id=MARKER))


def _other_major(document: Json) -> Json:
    changed = dict(document)
    changed["schema_version"] = "2.0"
    changed["type"] = MARKER  # free text where the contract has a constant
    return changed


SUBSTITUTION_CASES: dict[str, Json] = {
    "substitution-accepted": _substitution_with_personal_data(substitution_request()),
    "substitution-rejected": _other_major(_substitution_with_personal_data(substitution_request())),
    "substitution-rejected-id": _in_absent_teacher_id(
        _substitution_with_personal_data(substitution_request())
    ),
    "substitution-poison": with_data(
        _substitution_with_personal_data(substitution_request()), _no_job_id
    ),
}


@pytest.mark.parametrize("case", sorted(CASES | SUBSTITUTION_CASES))
def test_marker_never_leaves_the_worker(case: str, capsys: pytest.CaptureFixture[str]) -> None:
    h = harness()
    if case in SUBSTITUTION_CASES:
        delivered = inbound(encode(SUBSTITUTION_CASES[case]), topic=SUBSTITUTION_TOPIC)
        assert h.run(delivered) == 0
    else:
        assert h.run(encode(CASES[case])) == 0

    out, err = capsys.readouterr()
    assert MARKER not in out
    assert MARKER not in err
    assert h.producer.sent, "every case produces at least one message"
    if case.startswith(("rejected", "poison")):
        assert h.producer.dlq(), "the redacted DLQ copy is still produced"
    if case.startswith(("substitution-rejected", "substitution-poison")):
        assert h.producer.dlq(SUBSTITUTION_DLQ_TOPIC), "the redacted DLQ copy is still produced"
    for message in h.producer.sent:
        assert MARKER.encode() not in (message.value or b"")
        assert ESCAPED not in (message.value or b"")
        for name, value in message.headers:
            assert MARKER not in name
            assert MARKER.encode() not in value
            assert ESCAPED not in value


def test_marker_never_reaches_the_engine() -> None:
    h = harness()

    h.run(encode(CASES["accepted"]))

    problem, _ = h.engine.calls[0]
    assert MARKER not in repr(problem)


def test_marker_never_reaches_the_substitution_engine() -> None:
    h = harness()

    h.run(inbound(encode(SUBSTITUTION_CASES["substitution-accepted"]), topic=SUBSTITUTION_TOPIC))

    [(request, _)] = h.substitution.calls
    assert MARKER not in repr(request)
    assert h.producer.of_type("solve.completed")[0]["data"]["status"] == "OPTIMAL"


def test_validation_errors_do_not_quote_input() -> None:
    document: dict[str, Any] = _with_personal_data(build_request())
    document["data"]["classes"][0]["id"] = MARKER + " with spaces"  # fails the id pattern

    with pytest.raises(ValidationError) as caught:
        BuildRequested.model_validate(document)

    assert MARKER not in str(caught.value)
    assert MARKER not in repr(caught.value.errors(include_input=False))


def test_engine_exception_text_is_not_logged(capsys: pytest.CaptureFixture[str]) -> None:
    h = harness(engine=FakeEngine(Journal(), error=RuntimeError(MARKER)))

    h.run(encode(build_request()))

    out, err = capsys.readouterr()
    assert MARKER not in out + err
    assert '"error_class": "RuntimeError"' in out


def test_where_names_code_locations_only(capsys: pytest.CaptureFixture[str]) -> None:
    h = harness(engine=FakeEngine(Journal(), error=RuntimeError(MARKER)))

    h.run(encode(build_request()))

    entries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    located = [e["where"] for e in entries if "where" in e]
    assert located, "engine.failed carries a code location"
    for where in located:
        assert MARKER not in where
        for frame in where.split(" > "):
            module, function, line = frame.split(":")
            assert module.startswith("lad_solver.")
            assert function.isidentifier()
            assert line.isdigit()


def test_secrets_and_bodies_passed_to_a_logger_are_dropped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A careless call that hands a logger request headers, a body or a temporary password
    # (personal-data rules): neither the allow-list nor %-formatting lets the value through.
    secret = f"Bearer {MARKER}"
    jsonlog.log(
        logging.ERROR,
        "job.completed",
        job_id="j-1",
        Authorization=secret,
        authorization=secret,
        x_access_token=secret,
        headers={"Authorization": secret, "X-Access-Token": secret},
        body=secret,
        temporary_password=MARKER,
    )
    engine = logging.getLogger("lad_solver.app.redaction")
    engine.warning("engine.event", extra={"Authorization": secret, "X-Access-Token": secret})
    engine.info("engine.event %s", secret)
    engine.error("engine.event %(body)s", {"body": secret})

    out, err = capsys.readouterr()
    assert MARKER not in out + err
    assert len(out.splitlines()) == 4  # every call was written, only without the values
