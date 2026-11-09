"""Worker loop on fake ports: ordering, retries, rejections and shutdown (ADR 2026-11-04 §4)."""

import hashlib
import json
import threading
from uuid import UUID

import pytest
from kafka_support import (
    JOB_ID,
    SCHOOL_ID,
    FakeEngine,
    Journal,
    build_request,
    decode,
    encode,
    harness,
    with_data,
)

from lad_solver.adapters.kafka.mapping import event_id
from lad_solver.domain.result import Lesson, Result, SolveStatus


def test_accepted_job_publishes_started_commits_then_solves_then_completes() -> None:
    h = harness()

    assert h.run(encode(build_request())) == 0

    assert h.journal.entries == ["send:solve.started", "commit", "solve", "send:solve.completed"]
    assert h.consumer.committed == [7]
    for message in h.producer.sent:
        assert message.key == SCHOOL_ID.encode()
    started, completed = h.producer.of_type("solve.started"), h.producer.of_type("solve.completed")
    assert started[0]["data"] == {"job_id": JOB_ID, "started_at": "2026-11-04T10:00:00Z"}
    assert completed[0]["data"]["status"] == "FEASIBLE"
    assert completed[0]["data"]["lessons"][0] == {
        "class_id": "c-5a",
        "day": 1,
        "lesson_no": 1,
        "subject_id": "s-math",
        "teacher_id": "t-1",
    }
    assert completed[0]["correlation_id"] == "req-5f2c9a"
    assert completed[0]["event_id"] == str(event_id(UUID(JOB_ID), "solve.completed"))
    assert h.consumer.closed
    assert h.producer.closed


def test_engine_receives_problem_and_clamped_params() -> None:
    h = harness(num_workers=2, seed=7)

    h.run(encode(build_request(time_limit_s=999)))

    problem, params = h.engine.calls[0]
    assert problem.lessons_per_day == 6
    assert params.time_limit_s == 180
    assert (params.num_workers, params.random_seed) == (2, 7)


def test_expired_job_completes_without_started_and_without_solving() -> None:
    h = harness()

    h.run(encode(build_request(queue_deadline_at="2026-11-04T09:59:59Z")))

    assert h.journal.entries == ["send:solve.completed", "commit"]
    completed = h.producer.of_type("solve.completed")[0]["data"]
    assert completed == {
        "job_id": JOB_ID,
        "status": "EXPIRED",
        "lessons": [],
        "day_formats": [],
        "conflicts": [],
    }


def test_deadline_exactly_now_counts_as_expired() -> None:
    h = harness()

    h.run(encode(build_request(queue_deadline_at="2026-11-04T12:00:00+02:00")))

    assert h.producer.of_type("solve.completed")[0]["data"]["status"] == "EXPIRED"


def test_poison_message_goes_to_dlq_and_is_committed() -> None:
    h = harness()
    raw = b"\x00not json at all"

    assert h.run(raw) == 0

    assert h.journal.entries == ["send:dlq", "commit"]
    dlq = h.producer.dlq()[0]
    # Unparseable bytes never land in the DLQ as-is (they may hold personal data, kept 14 days):
    # only size and hash, enough to match the message with the producer's own logs.
    assert dlq.value is not None
    assert json.loads(dlq.value) == {
        "unparseable": True,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert dict(dlq.headers) == {
        "x-error-class": b"not_json",
        "x-original-topic": b"solve.build.requested",
        "x-original-partition": b"0",
        "x-original-offset": b"7",
    }
    assert h.engine.calls == []


def test_envelope_without_job_id_is_poison() -> None:
    h = harness()
    document = with_data(build_request(), lambda d: d.pop("job_id"))

    h.run(encode(document))

    assert h.journal.entries == ["send:dlq", "commit"]
    assert dict(h.producer.dlq()[0].headers)["x-error-class"] == b"invalid_envelope"


def test_invalid_data_completes_model_invalid_with_dlq_copy() -> None:
    h = harness()
    document = with_data(build_request(), lambda d: d["classes"][0].update(students=-1))

    h.run(encode(document))

    assert h.journal.entries == ["send:solve.completed", "send:dlq", "commit"]
    data = h.producer.of_type("solve.completed")[0]["data"]
    assert data["status"] == "MODEL_INVALID"
    assert data["error"] == {"code": "invalid_payload"}
    assert data["conflicts"] == [{"rule": "invalid_input"}]
    assert dict(h.producer.dlq()[0].headers)["x-error-class"] == b"validation"
    assert h.engine.calls == []


def test_unknown_major_schema_version_is_rejected() -> None:
    h = harness()
    document = build_request()
    document["schema_version"] = "2.0"

    h.run(encode(document))

    data = h.producer.of_type("solve.completed")[0]["data"]
    assert data["error"] == {"code": "unsupported_schema_version"}


def test_dangling_curriculum_reference_is_rejected() -> None:
    h = harness()
    document = with_data(build_request(), lambda d: d["rows"][0].update(teacher_id="t-404"))

    h.run(encode(document))

    data = h.producer.of_type("solve.completed")[0]["data"]
    assert (data["status"], data["error"]["code"]) == ("MODEL_INVALID", "invalid_payload")


def test_failed_started_is_retried_then_succeeds() -> None:
    h = harness()
    h.producer.fail_types["solve.started"] = 2

    assert h.run(encode(build_request())) == 0

    assert h.sleeps == [1.0, 2.0]
    assert h.journal.entries[-3:] == ["commit", "solve", "send:solve.completed"]


def test_started_that_never_flushes_exits_without_commit_or_solve() -> None:
    h = harness()
    h.producer.fail_types["solve.started"] = 99

    assert h.run(encode(build_request())) == 1

    assert h.sleeps == [1.0, 2.0, 4.0]
    assert h.consumer.committed == []
    assert h.engine.calls == []
    assert h.consumer.closed


def test_failed_commit_never_starts_the_solve() -> None:
    h = harness()
    h.consumer.commit_failures = 99

    assert h.run(encode(build_request())) == 1

    assert "solve" not in h.journal.entries
    assert h.journal.entries.count("commit-failed") == 4


def test_completed_that_never_flushes_logs_job_and_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    h = harness()
    h.producer.fail_types["solve.completed"] = 99

    assert h.run(encode(build_request())) == 1

    assert h.journal.entries.count("solve") == 1  # the solve itself is never retried
    out = capsys.readouterr().out
    gave_up = [line for line in out.splitlines() if "solve.completed.gave_up" in line]
    assert len(gave_up) == 1
    assert JOB_ID in gave_up[0]
    assert '"level": "error"' in gave_up[0]


def test_engine_exception_becomes_model_invalid_internal() -> None:
    engine = FakeEngine(Journal(), error=NotImplementedError("engine pending"))
    h = harness(engine=engine)

    assert h.run(encode(build_request())) == 0

    data = h.producer.of_type("solve.completed")[0]["data"]
    assert (data["status"], data["error"]) == ("MODEL_INVALID", {"code": "internal"})
    assert data["conflicts"] == []


def test_oversized_result_becomes_result_too_large() -> None:
    many = tuple(Lesson("c-5a", 1 + i % 5, 1 + i % 6, "s-math", "t-1") for i in range(300))
    engine = FakeEngine(Journal(), result=Result(status=SolveStatus.FEASIBLE, lessons=many))
    h = harness(engine=engine, result_max_bytes=4096)

    h.run(encode(build_request()))

    completed = h.producer.of_type("solve.completed")[0]["data"]
    assert completed["status"] == "MODEL_INVALID"
    assert completed["error"] == {"code": "result_too_large"}
    assert completed["lessons"] == []
    assert all(len(m.value or b"") <= 4096 for m in h.producer.sent)


def test_sigterm_during_solve_stops_engine_and_publishes_best_solution() -> None:
    engine = FakeEngine(Journal(), wait_for_stop=True)
    h = harness(engine=engine)
    h.consumer.on_idle = None  # only the signal ends this run
    h.producer.on_poll = h.worker.request_stop  # "SIGTERM" arrives on the main thread mid-solve

    assert h.run(encode(build_request())) == 0

    assert engine.stop_seen
    assert h.producer.of_type("solve.completed")[0]["data"]["status"] == "FEASIBLE"
    assert h.consumer.closed
    assert h.producer.closed
    assert h.beats  # heartbeat is touched while the solve runs


def test_engine_ignoring_stop_past_grace_exits_with_failure() -> None:
    release = threading.Event()
    engine = FakeEngine(Journal(), ignore_stop=release)
    h = harness(engine=engine, stop_grace_s=0.05)
    h.consumer.on_idle = None
    h.producer.on_poll = h.worker.request_stop
    try:
        assert h.run(encode(build_request())) == 1
    finally:
        release.set()
    assert h.producer.of_type("solve.completed") == []


def test_watchdog_hard_exits_a_solve_that_overruns() -> None:
    release = threading.Event()
    engine = FakeEngine(Journal(), ignore_stop=release)
    h = harness(engine=engine, min_time_limit_s=1, watchdog_margin_s=0.01)
    h.worker.hard_exit = lambda code: (h.exits.append(code), release.set())[0]

    h.run(encode(build_request(time_limit_s=1)))

    assert h.exits == [1]


def test_idle_sigterm_closes_and_exits_cleanly() -> None:
    h = harness()

    assert h.run() == 0

    assert h.journal.entries == []
    assert h.consumer.closed
    assert h.producer.closed


def test_messages_are_processed_one_at_a_time_in_order() -> None:
    h = harness()
    second = build_request()
    second["data"]["job_id"] = "1a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"

    h.run(encode(build_request()), encode(second))

    assert h.consumer.committed == [7, 8]
    assert [decode(m)["data"]["job_id"] for m in h.producer.sent] == [
        JOB_ID,
        JOB_ID,
        *[second["data"]["job_id"]] * 2,
    ]


def test_unparseable_poison_with_personal_data_is_not_copied_to_the_dlq() -> None:
    h = harness()
    raw = '{"teachers": [{"signature": "Іваненко"'.encode()

    h.run(raw)  # truncated JSON with a signature

    value = h.producer.dlq()[0].value or b""
    assert b"signature" not in value
    assert "Іваненко".encode() not in value
