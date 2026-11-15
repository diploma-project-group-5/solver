"""Worker loop on fake ports: ordering, retries, rejections and shutdown (ADR 2026-11-04 §4)."""

import datetime as dt
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest
from kafka_support import (
    DLQ_TOPIC,
    JOB_ID,
    SCHOOL_ID,
    STATUS_TOPIC,
    SUBSTITUTION_DLQ_TOPIC,
    SUBSTITUTION_TOPIC,
    FakeEngine,
    FakeSubstitution,
    Harness,
    Journal,
    Json,
    build_request,
    decode,
    encode,
    feasible_result,
    harness,
    inbound,
    substitution_request,
    with_data,
)

from lad_solver.adapters.kafka import jsonlog, mapping
from lad_solver.adapters.kafka.mapping import event_id
from lad_solver.adapters.kafka.ports import InboundMessage
from lad_solver.domain.problem import Params, Problem
from lad_solver.domain.result import CoverageVerdict, Lesson, Result, SolveStatus
from lad_solver.domain.substitution import (
    Proposal,
    ProposalKind,
    Slot,
    SubstitutionResult,
    SubstitutionStats,
)
from lad_solver.model.progress import Progress
from lad_solver.model.stop import StopHandle


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


def _first_solution_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if "engine.first_solution" in line]


def test_first_solution_is_logged_once_with_its_time(capsys: pytest.CaptureFixture[str]) -> None:
    # Operators (and the image SIGTERM test) see when a job has a timetable to return.
    h = harness(engine=FakeEngine(Journal(), report_solution=True))

    assert h.run(encode(build_request())) == 0

    [line] = _first_solution_lines(capsys.readouterr().out)
    record = json.loads(line)
    assert record["job_id"] == JOB_ID
    assert isinstance(record["first_solution_ms"], int)
    assert record["first_solution_ms"] >= 0


def test_no_first_solution_line_without_a_solution(capsys: pytest.CaptureFixture[str]) -> None:
    h = harness()

    assert h.run(encode(build_request())) == 0

    assert _first_solution_lines(capsys.readouterr().out) == []


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


BASE_LOG_KEYS = frozenset({"time", "level", "logger", "msg"})


def _entries(out: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in out.splitlines()]


@dataclass
class _LoggingEngine(FakeEngine):
    """Logs from inside the engine, the way any lad_solver.app / model logger would."""

    thread_names: list[str] = field(default_factory=list[str])

    def __call__(
        self, problem: Problem, params: Params, stop: StopHandle, progress: Progress
    ) -> Result:
        self.thread_names.append(threading.current_thread().name)
        logging.getLogger("lad_solver.app.fake").info("engine.event")
        return super().__call__(problem, params, stop, progress)


def test_engine_logs_from_the_solve_thread_carry_the_job_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _LoggingEngine(Journal())
    h = harness(engine=engine)

    assert h.run(encode(build_request()), b"\x00not json at all") == 0

    assert engine.thread_names == ["lad-solve"]
    entries = _entries(capsys.readouterr().out)
    [event] = [e for e in entries if e["msg"] == "engine.event"]
    assert (event["job_id"], event["school_id"], event["correlation_id"]) == (
        JOB_ID,
        SCHOOL_ID,
        "req-5f2c9a",
    )
    poison = [e for e in entries if e["msg"] in {"message.poison", "message.done"}]
    assert len(poison) == 2
    for entry in poison:  # the next message does not inherit the previous job's context
        assert {"job_id", "school_id", "correlation_id"}.isdisjoint(entry)


def _with_coverage_stats(**changes: object) -> FakeEngine:
    result = feasible_result()
    stats = replace(
        result.stats,
        coverage_verdict=CoverageVerdict.ACCEPTED,
        coverage_status=SolveStatus.FEASIBLE,
        zero_day_before=3,
        zero_day_after=1,
        phase_b_retry_days=2,
    )
    return FakeEngine(Journal(), result=replace(result, stats=replace(stats, **changes)))


def test_job_completed_reports_the_coverage_step_and_phase_b_retries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    h = harness(engine=_with_coverage_stats())

    assert h.run(encode(build_request())) == 0

    entries = _entries(capsys.readouterr().out)
    [completed] = [e for e in entries if e["msg"] == "job.completed"]
    assert completed["coverage_verdict"] == "accepted"
    assert completed["coverage_status"] == "FEASIBLE"
    assert (completed["zero_day_before"], completed["zero_day_after"]) == (3, 1)
    assert completed["phase_b_retry_days"] == 2
    for entry in entries:
        assert set(entry) <= BASE_LOG_KEYS | jsonlog.ALLOWED_FIELDS
    assert not [e for e in entries if e["msg"] == "engine.coverage_anomaly"]


@pytest.mark.parametrize("status", [SolveStatus.INFEASIBLE, SolveStatus.MODEL_INVALID])
def test_impossible_coverage_status_is_a_warning(
    status: SolveStatus, capsys: pytest.CaptureFixture[str]
) -> None:
    # A hinted feasible model cannot be INFEASIBLE / MODEL_INVALID: an engine bug worth seeing.
    engine = _with_coverage_stats(coverage_verdict=CoverageVerdict.KEPT, coverage_status=status)
    h = harness(engine=engine)

    assert h.run(encode(build_request())) == 0

    [anomaly] = [
        e for e in _entries(capsys.readouterr().out) if e["msg"] == "engine.coverage_anomaly"
    ]
    assert anomaly["level"] == "warning"
    assert anomaly["coverage_status"] == status.value
    assert anomaly["job_id"] == JOB_ID


# ---------------------------------------------------------------- substitutions (stage 2)
# ADR 2026-11-15-solver-substitution-engine: a substitution job is solved like a build job.


def _substitution(document: Json | None = None, offset: int = 7) -> InboundMessage:
    raw = encode(document if document is not None else substitution_request())
    return inbound(raw, offset, topic=SUBSTITUTION_TOPIC)


def _completed_data(h: Harness) -> Json:
    [completed] = h.producer.of_type("solve.completed")
    return cast("Json", completed["data"])


def test_substitution_publishes_started_commits_then_solves_then_completes() -> None:
    h = harness()

    assert h.run(_substitution()) == 0

    assert h.journal.entries == [
        "send:solve.started",
        "commit",
        "substitute",
        "send:solve.completed",
    ]
    assert h.engine.calls == []
    assert h.consumer.committed == [7]
    assert (h.producer.dlq(DLQ_TOPIC), h.producer.dlq(SUBSTITUTION_DLQ_TOPIC)) == ([], [])
    for message in h.producer.sent:
        assert (message.topic, message.key) == (STATUS_TOPIC, SCHOOL_ID.encode())
    started, completed = h.producer.of_type("solve.started"), h.producer.of_type("solve.completed")
    assert started[0]["data"] == {"job_id": JOB_ID, "started_at": "2026-11-04T10:00:00Z"}
    assert completed[0]["event_id"] == str(event_id(UUID(JOB_ID), "solve.completed"))
    assert completed[0]["correlation_id"] == "req-5f2c9a"
    data = completed[0]["data"]
    assert (data["status"], data["lessons"], data["day_formats"], data["conflicts"]) == (
        "OPTIMAL",
        [],
        [],
        [],
    )
    assert data["substitution"] == {
        "step": 1,
        "proposals": [
            {
                "date": "2026-11-11",
                "class_id": "c-5a",
                "lesson_no": 2,
                "kind": "teacher",
                "teacher_id": "t-2",
                "off_profile": True,
                "step": 1,
            }
        ],
    }


def test_substitution_stats_carry_provenance_only() -> None:
    h = harness()

    h.run(_substitution())

    stats = _completed_data(h)["stats"]
    assert set(stats) == {
        "wall_time_ms",
        "num_workers",
        "random_seed",
        "solver_version",
        "model_version",
        "input_sha256",
        "stop_reason",
    }
    assert (stats["num_workers"], stats["random_seed"]) == (1, 0)
    assert (stats["model_version"], stats["stop_reason"]) == ("a+a2+b+s/2026-11-15", "completed")


def test_substitution_engine_gets_the_clamped_limit_and_the_configured_seed() -> None:
    h = harness(seed=7, max_memory_mb=256)

    h.run(_substitution(substitution_request(time_limit_s=3600)))

    [(request, params)] = h.substitution.calls
    assert (params.time_limit_s, params.random_seed, params.max_memory_mb) == (180.0, 7, 256)
    assert request.absent_teacher_id == "t-1"


def test_redelivered_substitution_gives_the_same_events() -> None:
    h = harness()

    h.run(_substitution(offset=7), _substitution(offset=7))

    first, second = h.producer.of_type("solve.completed")
    assert first["event_id"] == second["event_id"]  # timetable's inbox drops the copy
    for data in (first["data"], second["data"]):
        data["stats"].pop("wall_time_ms")
    assert first == second


def test_expired_substitution_is_closed_without_started() -> None:
    h = harness()

    h.run(_substitution(substitution_request(queue_deadline_at="2026-11-01T00:00:00Z")))

    assert h.journal.entries == ["send:solve.completed", "commit"]
    assert h.substitution.calls == []
    assert _completed_data(h) == {
        "job_id": JOB_ID,
        "status": "EXPIRED",
        "lessons": [],
        "day_formats": [],
        "conflicts": [],
    }


@pytest.mark.parametrize(
    ("deadline", "clock", "expired"),
    [
        # 23:30 Kyiv on Monday is 21:30Z: still Monday in UTC as in Kyiv.
        ("2026-11-16T23:30:00+02:00", datetime(2026, 11, 16, 21, 29, 59, tzinfo=UTC), False),
        ("2026-11-16T23:30:00+02:00", datetime(2026, 11, 16, 21, 30, tzinfo=UTC), True),
        # 00:30 Kyiv on Tuesday is 22:30Z on Monday: the UTC date is the day before.
        ("2026-11-17T00:30:00+02:00", datetime(2026, 11, 16, 22, 29, 59, tzinfo=UTC), False),
        ("2026-11-17T00:30:00+02:00", datetime(2026, 11, 16, 22, 30, tzinfo=UTC), True),
        # DST ends on 2026-10-25: 03:30 happens twice in Kyiv, +03:00 first, +02:00 an hour later.
        ("2026-10-25T03:30:00+03:00", datetime(2026, 10, 25, 1, 0, tzinfo=UTC), True),
        ("2026-10-25T03:30:00+02:00", datetime(2026, 10, 25, 1, 0, tzinfo=UTC), False),
        # DST starts on 2026-03-29: 02:59+02:00 is followed by 04:00+03:00, one minute later.
        ("2026-03-29T04:00:00+03:00", datetime(2026, 3, 29, 0, 59, 59, tzinfo=UTC), False),
        ("2026-03-29T04:00:00+03:00", datetime(2026, 3, 29, 1, 0, tzinfo=UTC), True),
    ],
)
def test_substitution_queue_deadline_is_an_instant_around_kyiv_midnight_and_dst(
    deadline: str, clock: datetime, *, expired: bool
) -> None:
    h = harness(clock=clock)

    h.run(_substitution(substitution_request(queue_deadline_at=deadline)))

    status = _completed_data(h)["status"]
    assert (status == "EXPIRED") == expired
    assert bool(h.substitution.calls) is not expired


def test_invalid_substitution_is_answered_and_copied_to_its_dlq() -> None:
    h = harness()

    h.run(_substitution(substitution_request(lessons=[])))  # minItems 1

    assert h.journal.entries == ["send:solve.completed", "send:dlq", "commit"]
    assert h.substitution.calls == []
    data = _completed_data(h)
    assert (data["status"], data["error"]) == ("MODEL_INVALID", {"code": "invalid_payload"})
    assert data["conflicts"] == [{"rule": "invalid_input"}]
    assert h.producer.dlq(DLQ_TOPIC) == []
    [dead] = h.producer.dlq(SUBSTITUTION_DLQ_TOPIC)
    assert dict(dead.headers)["x-error-class"] == b"validation"
    projected = json.loads(dead.value or b"")["projected"]
    assert projected["data"] == {"job_id": JOB_ID}  # who is absent never reaches the DLQ


def test_inconsistent_substitution_week_is_refused_as_referential_integrity() -> None:
    h = harness()
    # c-9b L2 is not a lesson of the template: there is nothing to cover.
    document = substitution_request(lessons=[{"class_id": "c-9b", "lesson_no": 2}])

    h.run(_substitution(document))

    assert h.substitution.calls == []
    data = _completed_data(h)
    assert (data["status"], data["error"]) == ("MODEL_INVALID", {"code": "invalid_payload"})
    [dead] = h.producer.dlq(SUBSTITUTION_DLQ_TOPIC)
    assert dict(dead.headers)["x-error-class"] == b"referential_integrity"


def test_poison_substitution_goes_to_the_substitution_dlq() -> None:
    h = harness()

    assert h.run(inbound(b"not json", topic=SUBSTITUTION_TOPIC)) == 0

    assert h.journal.entries == ["send:dlq", "commit"]
    assert h.producer.dlq(DLQ_TOPIC) == []
    [dead] = h.producer.dlq(SUBSTITUTION_DLQ_TOPIC)
    assert dict(dead.headers)["x-original-topic"] == SUBSTITUTION_TOPIC.encode()
    assert dict(dead.headers)["x-error-class"] == b"not_json"


def test_substitution_of_another_major_is_answered_and_copied_to_its_dlq() -> None:
    h = harness()
    document = substitution_request()
    document["schema_version"] = "2.0"

    h.run(_substitution(document))

    assert h.journal.entries == ["send:solve.completed", "send:dlq", "commit"]
    assert _completed_data(h)["error"] == {"code": "unsupported_schema_version"}
    [dead] = h.producer.dlq(SUBSTITUTION_DLQ_TOPIC)
    projected = json.loads(dead.value or b"")["projected"]
    assert projected["data"] == {"job_id": JOB_ID}  # the substitution projection, not the build one


def test_substitution_parser_bug_is_internal_and_goes_to_its_dlq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_args: object) -> None:
        msg = "bug"
        raise RuntimeError(msg)

    monkeypatch.setattr(mapping, "_envelope_rejection", broken)
    h = harness()

    h.run(_substitution())

    data = _completed_data(h)
    assert (data["status"], data["error"]) == ("MODEL_INVALID", {"code": "internal"})
    assert len(h.producer.dlq(SUBSTITUTION_DLQ_TOPIC)) == 1
    assert h.consumer.committed == [7]


def test_undelivered_substitution_started_exits_without_commit_or_solve() -> None:
    h = harness()
    h.producer.fail_types["solve.started"] = 99

    assert h.run(_substitution()) == 1

    assert h.sleeps == [1.0, 2.0, 4.0]
    assert h.consumer.committed == []  # redelivered after the restart, answered then
    assert h.substitution.calls == []


def test_substitution_engine_failure_is_internal_and_never_retried() -> None:
    h = harness(substitution=FakeSubstitution(Journal(), error=RuntimeError("bug")))

    assert h.run(_substitution()) == 0

    data = _completed_data(h)
    assert (data["status"], data["error"]) == ("MODEL_INVALID", {"code": "internal"})
    assert "substitution" not in data
    assert len(h.substitution.calls) == 1
    assert h.consumer.committed == [7]


def test_sigterm_during_a_substitution_answers_the_best_known() -> None:
    """SIGTERM mid-job: the engine stops, whatever it fixed stays, the rest is self-study."""
    substitution = FakeSubstitution(Journal(), wait_for_stop=True)
    h = harness(substitution=substitution)
    h.consumer.on_idle = None
    h.producer.on_poll = h.worker.request_stop  # the signal arrives while the engine runs

    assert h.run(_substitution()) == 0

    assert substitution.stop_seen
    data = _completed_data(h)
    assert data["status"] == "FEASIBLE"
    assert data["stats"]["stop_reason"] == "stopped"
    [proposal] = data["substitution"]["proposals"]
    assert (proposal["kind"], proposal["step"]) == ("self_study", 1)
    assert h.consumer.committed == [7]


def test_too_many_proposals_become_result_too_large() -> None:
    proposals = tuple(
        Proposal(Slot(dt.date(2026, 11, 11), f"c-{i}", 1), ProposalKind.SELF_STUDY, 2)
        for i in range(mapping.MAX_PROPOSALS + 1)
    )
    result = SubstitutionResult(SolveStatus.FEASIBLE, proposals, SubstitutionStats())
    h = harness(substitution=FakeSubstitution(Journal(), result=result))

    h.run(_substitution())

    data = _completed_data(h)
    assert (data["status"], data["error"]) == ("MODEL_INVALID", {"code": "result_too_large"})


def test_support_moves_requested_are_logged_and_answered_without(
    capsys: pytest.CaptureFixture[str],
) -> None:
    h = harness()

    h.run(_substitution(substitution_request(allow_support_moves=True)))

    assert _completed_data(h)["status"] == "OPTIMAL"
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    [ignored] = [x for x in lines if x["msg"] == "job.support_moves_ignored"]
    assert (ignored["level"], ignored["job_id"]) == ("warning", JOB_ID)


def test_substitution_logs_counts_never_who_is_absent(capsys: pytest.CaptureFixture[str]) -> None:
    h = harness()

    h.run(_substitution())

    out = capsys.readouterr().out
    lines = [json.loads(x) for x in out.splitlines()]
    [started] = [x for x in lines if x["msg"] == "job.started"]
    [done] = [x for x in lines if x["msg"] == "job.completed"]
    assert (started["job_kind"], started["num_workers"]) == ("substitution", 1)
    assert (done["job_kind"], done["status"], done["step"]) == ("substitution", "OPTIMAL", 1)
    assert (done["proposals"], done["self_study"], done["covered"]) == (1, 0, [1, 0, 0])
    assert done["cp_runs"] == 1
    for leaked in ('"t-1"', '"t-2"', '"c-5a"', "2026-11-11"):
        assert leaked not in out


def test_build_after_a_substitution_is_solved_too() -> None:
    h = harness()

    h.run(_substitution(offset=3), encode(build_request()))

    assert h.journal.entries == [
        "send:solve.started",
        "commit",
        "substitute",
        "send:solve.completed",
        "send:solve.started",
        "commit",
        "solve",
        "send:solve.completed",
    ]
    assert (len(h.substitution.calls), len(h.engine.calls)) == (1, 1)


def test_message_from_a_topic_without_a_route_stops_the_worker() -> None:
    """Only a config bug can deliver one (the consumer subscribes to the routed topics)."""
    h = harness()

    assert h.run(inbound(encode(build_request()), topic="solve.other")) == 1

    assert h.producer.sent == []
    assert h.consumer.committed == []
