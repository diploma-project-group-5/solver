"""Worker robustness: signals, crashes, best-effort DLQ, shutdown budget, health (review fixes).

Each test names the review item it pins down (ADR 2026-11-04 §4).
"""

import json
import threading
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from kafka_support import (
    JOB_ID,
    FakeEngine,
    Journal,
    Json,
    build_request,
    encode,
    feasible_result,
    harness,
    with_data,
)

from lad_solver.adapters.kafka import mapping
from lad_solver.domain.problem import Params, Problem
from lad_solver.domain.result import Conflict, ConflictRule, Result, SolveStatus
from lad_solver.model.progress import Progress, StallPolicy
from lad_solver.model.stop import StopHandle


def _logs(out: str) -> list[Json]:
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def _events(out: str, msg: str) -> list[Json]:
    return [entry for entry in _logs(out) if entry["msg"] == msg]


class _Stoppable:
    def __init__(self) -> None:
        self.calls = 0

    def stop_search(self) -> None:
        self.calls += 1


# ---------------------------------------------------------------- 1. signal handler


def test_request_stop_only_sets_a_flag(capsys: pytest.CaptureFixture[str]) -> None:
    stoppable = _Stoppable()
    seen: list[tuple[int, str]] = []

    class AttachingEngine(FakeEngine):
        def __call__(
            self, problem: Problem, params: Params, stop: StopHandle, progress: Progress
        ) -> Result:
            stop.attach(stoppable)
            return super().__call__(problem, params, stop, progress)

    engine = AttachingEngine(Journal(), wait_for_stop=True)
    h = harness(engine=engine)
    h.consumer.on_idle = None

    def sigterm() -> None:
        if h.worker.stop_requested:
            return
        capsys.readouterr()
        before = stoppable.calls
        h.worker.request_stop("SIGTERM")
        seen.append((stoppable.calls - before, capsys.readouterr().out))

    h.producer.on_poll = sigterm

    assert h.run(encode(build_request())) == 0

    assert seen == [(0, "")]  # no stop_search and no logging inside the "handler"
    assert stoppable.calls >= 1  # the main loop stopped the search afterwards
    assert engine.stop_seen
    stopping = _events(capsys.readouterr().out, "worker.stopping")
    assert [e["signal"] for e in stopping] == ["SIGTERM"]


# ---------------------------------------------------------------- 2. stop between poll and handle


def test_stop_requested_while_polling_leaves_the_message_uncommitted() -> None:
    h = harness()
    h.consumer.before_deliver = h.worker.request_stop

    assert h.run(encode(build_request())) == 0

    assert h.journal.entries == []
    assert h.consumer.committed == []
    assert h.engine.calls == []


def test_stop_requested_before_started_is_sent_leaves_the_message_uncommitted() -> None:
    h = harness()
    # the clock is read right before `started` is built; a signal can land there
    h.worker.clock = lambda: (h.worker.request_stop(), datetime(2026, 11, 4, 10, tzinfo=UTC))[1]

    assert h.run(encode(build_request())) == 0

    assert h.journal.entries == []
    assert h.consumer.committed == []


# ---------------------------------------------------------------- 3. crash during a solve


def test_heartbeat_failure_mid_solve_stops_the_engine_and_exits_1(
    capsys: pytest.CaptureFixture[str],
) -> None:
    handles: list[StopHandle] = []

    class CapturingEngine(FakeEngine):
        def __call__(
            self, problem: Problem, params: Params, stop: StopHandle, progress: Progress
        ) -> Result:
            handles.append(stop)
            return super().__call__(problem, params, stop, progress)

    engine = CapturingEngine(Journal(), wait_for_stop=True)
    h = harness(engine=engine)
    beats = 0

    def failing_beat() -> None:
        nonlocal beats
        beats += 1
        if beats > 1:  # the first beat is the poll loop, later ones are inside _wait
            msg = "disk full"
            raise OSError(msg)

    h.worker.heartbeat = failing_beat

    assert h.run(encode(build_request())) == 1

    assert handles[0].stopped  # the search is interrupted, not left running
    crashed = _events(capsys.readouterr().out, "worker.crashed")
    assert len(crashed) == 1
    assert crashed[0]["level"] == "error"
    assert crashed[0]["job_id"] == JOB_ID
    assert crashed[0]["error_class"] == "OSError"
    assert "lad_solver.adapters.kafka.worker:" in crashed[0]["where"]
    assert h.producer.closed


def test_crash_while_the_engine_keeps_running_hard_exits() -> None:
    release = threading.Event()
    engine = FakeEngine(Journal(), ignore_stop=release)
    h = harness(engine=engine)
    h.worker.hard_exit = lambda code: (h.exits.append(code), release.set())[0]

    def failing_poll() -> None:
        msg = "unexpected"
        raise RuntimeError(msg)

    h.producer.on_poll = failing_poll  # producer.poll(0) inside _wait raises

    assert h.run(encode(build_request())) == 1

    assert h.exits == [1]


def test_engine_ignoring_stop_hard_exits_after_closing() -> None:
    release = threading.Event()
    engine = FakeEngine(Journal(), ignore_stop=release)
    h = harness(engine=engine, stop_grace_s=0.05)
    h.consumer.on_idle = None
    h.producer.on_poll = h.worker.request_stop
    h.worker.hard_exit = lambda code: (h.exits.append(code), release.set())[0]

    assert h.run(encode(build_request())) == 1

    assert h.exits == [1]
    assert h.producer.closed


# ---------------------------------------------------------------- 4. best-effort DLQ


def test_failed_dlq_copy_of_a_rejected_job_is_logged_and_committed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    h = harness()
    h.producer.fail_types["dlq"] = 99
    document = with_data(build_request(), lambda d: d["classes"][0].update(students=-1))

    assert h.run(encode(document)) == 0

    assert h.producer.of_type("solve.completed")[0]["data"]["status"] == "MODEL_INVALID"
    assert h.consumer.committed == [7]
    failed = _events(capsys.readouterr().out, "dlq.failed")
    assert len(failed) == 1
    assert failed[0]["level"] == "error"
    assert (failed[0]["topic"], failed[0]["partition"], failed[0]["offset"]) == (
        "solve.build.requested",
        0,
        7,
    )
    assert failed[0]["job_id"] == JOB_ID


def test_failed_dlq_copy_of_poison_is_logged_and_committed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    h = harness()
    h.producer.fail_types["dlq"] = 99

    assert h.run(b"not json", b"also not json") == 0

    assert h.consumer.committed == [7, 8]  # no crash loop on one offset
    assert len(_events(capsys.readouterr().out, "dlq.failed")) == 2


def test_completed_is_sent_before_and_independently_of_the_dlq_copy() -> None:
    h = harness()
    document = with_data(build_request(), lambda d: d["classes"][0].update(students=-1))

    h.run(encode(document))

    assert h.journal.entries == ["send:solve.completed", "send:dlq", "commit"]
    assert h.producer.batches == [["solve.completed"], ["dlq"]]


# ---------------------------------------------------------------- 5. parser bug with an address


def test_parser_bug_with_an_address_completes_internal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(_data: object) -> Problem:
        msg = "bug"
        raise RuntimeError(msg)

    monkeypatch.setattr(mapping, "to_problem", broken)
    h = harness()

    assert h.run(encode(build_request())) == 0

    data = h.producer.of_type("solve.completed")[0]["data"]
    assert (data["status"], data["error"]) == ("MODEL_INVALID", {"code": "internal"})
    assert h.consumer.committed == [7]
    failed = _events(capsys.readouterr().out, "parser.failed")
    assert len(failed) == 1
    assert failed[0]["level"] == "error"
    assert failed[0]["job_id"] == JOB_ID
    assert failed[0]["error_class"] == "RuntimeError"
    assert "lad_solver.adapters.kafka.mapping:" in failed[0]["where"]


# ---------------------------------------------------------------- 6. every MODEL_INVALID has a code


def test_engine_model_invalid_without_code_becomes_internal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = FakeEngine(Journal(), result=Result(status=SolveStatus.MODEL_INVALID))
    h = harness(engine=engine)

    assert h.run(encode(build_request())) == 0

    data = h.producer.of_type("solve.completed")[0]["data"]
    assert (data["status"], data["error"]) == ("MODEL_INVALID", {"code": "internal"})
    logged = _events(capsys.readouterr().out, "engine.model_invalid")
    assert len(logged) == 1
    assert (logged[0]["level"], logged[0]["job_id"]) == ("error", JOB_ID)


def test_precheck_invalid_input_becomes_invalid_payload() -> None:
    conflicts = (Conflict(ConflictRule.INVALID_INPUT, class_id="c-5a"),)
    result = Result(status=SolveStatus.MODEL_INVALID, conflicts=conflicts)
    h = harness(engine=FakeEngine(Journal(), result=result))

    h.run(encode(build_request()))

    data = h.producer.of_type("solve.completed")[0]["data"]
    assert (data["status"], data["error"]) == ("MODEL_INVALID", {"code": "invalid_payload"})
    assert data["conflicts"] == [{"rule": "invalid_input", "class_id": "c-5a"}]


# ---------------------------------------------------------------- 8. undelivered on close


def test_undelivered_messages_on_close_are_logged(capsys: pytest.CaptureFixture[str]) -> None:
    h = harness()
    h.producer.undelivered_on_close = 2

    h.run(encode(build_request()))

    logged = _events(capsys.readouterr().out, "producer.undelivered")
    assert len(logged) == 1
    assert logged[0]["level"] == "error"
    assert logged[0]["count"] == 2
    assert logged[0]["job_id"] == JOB_ID


# ---------------------------------------------------------------- 9. health and shutdown budget


def test_heartbeat_stops_after_many_consecutive_consumer_errors() -> None:
    h = harness()
    h.consumer.poll_errors = 35

    assert h.run() == 0

    assert len(h.beats) == 30


def test_heartbeat_resumes_after_a_successful_poll() -> None:
    h = harness()
    h.consumer.poll_errors = 3

    assert h.run(encode(build_request())) == 0

    assert len(h.beats) > 3
    assert h.consumer.committed == [7]


def test_heartbeat_is_touched_between_retry_attempts() -> None:
    h = harness()
    h.producer.fail_types["solve.started"] = 2
    h.worker.heartbeat = lambda: h.journal.entries.append("beat")
    h.worker.sleep = lambda s: h.journal.entries.append(f"sleep:{s}")

    h.run(encode(build_request()))

    entries = h.journal.entries
    first = entries.index("send-failed:solve.started")
    assert entries[first + 1 : first + 3] == ["beat", "sleep:1.0"]


def test_shutdown_caps_the_completed_and_close_flushes() -> None:
    engine = FakeEngine(Journal(), wait_for_stop=True)
    h = harness(engine=engine)
    h.consumer.on_idle = None
    h.producer.on_poll = h.worker.request_stop

    assert h.run(encode(build_request())) == 0

    config = h.worker.config
    completed = [t for label, t in h.producer.send_timeouts if label == "solve.completed"]
    assert completed == [config.stop_flush_timeout_s]
    assert h.producer.close_timeouts == [config.stop_flush_timeout_s]
    assert config.stop_grace_s + 2 * config.stop_flush_timeout_s < 30


def test_completed_is_not_retried_once_stopping() -> None:
    engine = FakeEngine(Journal(), wait_for_stop=True)
    h = harness(engine=engine)
    h.consumer.on_idle = None
    h.producer.on_poll = h.worker.request_stop
    h.producer.fail_types["solve.completed"] = 99

    assert h.run(encode(build_request())) == 1

    assert h.sleeps == []


# ---------------------------------------------------------------- 20. deadlines across offsets


@pytest.mark.parametrize(
    ("now", "deadline", "expired"),
    [
        # summer time (+03:00): the same instant counts as expired, one second later does not
        (datetime(2026, 7, 1, 9, 0, tzinfo=UTC), "2026-07-01T12:00:00+03:00", True),
        (datetime(2026, 7, 1, 9, 0, tzinfo=UTC), "2026-07-01T12:00:01+03:00", False),
        # spring forward, Sun 29.03.2026: 03:00 EET -> 04:00 EEST (01:00Z); now = 04:30 EEST
        (datetime(2026, 3, 29, 1, 30, tzinfo=UTC), "2026-03-29T02:59:00+02:00", True),
        (datetime(2026, 3, 29, 1, 30, tzinfo=UTC), "2026-03-29T04:45:00+03:00", False),
        # fall back, Sun 25.10.2026: 04:00 EEST -> 03:00 EET (01:00Z); 03:30 local occurs twice
        (datetime(2026, 10, 25, 1, 0, tzinfo=UTC), "2026-10-25T03:30:00+03:00", True),
        (datetime(2026, 10, 25, 1, 0, tzinfo=UTC), "2026-10-25T03:30:00+02:00", False),
    ],
)
def test_queue_deadline_is_compared_as_an_instant(
    now: datetime, deadline: str, *, expired: bool
) -> None:
    h = harness(clock=now)

    h.run(encode(build_request(queue_deadline_at=deadline)))

    status = h.producer.of_type("solve.completed")[0]["data"]["status"]
    assert (status == "EXPIRED") is expired
    assert (h.engine.calls == []) is expired


# ---------------------------------------------------------------- 9. poison that breaks encoding


def _ascii(document: Json) -> bytes:
    """Lone surrogates survive only as JSON escapes; json.loads turns them into code points."""
    return json.dumps(document, ensure_ascii=True).encode()


@pytest.mark.parametrize("depth", [2_000, 50_000])
def test_deeply_nested_poison_is_dlq_copied_and_committed(depth: int) -> None:
    h = harness()

    assert h.run(b"[" * depth + b"]" * depth, b"not json") == 0

    assert h.consumer.committed == [7, 8]  # no crash loop on one offset
    assert len(h.producer.dlq()) == 2


def test_deeply_nested_unknown_field_of_an_addressed_job_is_ignored() -> None:
    h = harness()
    head, tail = encode(build_request()).rsplit(b"}", 1)
    raw = head + b',"extra":' + b"[" * 50_000 + b"]" * 50_000 + b"}" + tail

    assert h.run(raw) == 0

    assert h.producer.of_type("solve.completed")[0]["data"]["status"] == "FEASIBLE"
    assert h.consumer.committed == [7]


def test_lone_surrogate_in_correlation_id_is_rejected_under_the_job_id() -> None:
    h = harness()
    document = build_request()
    document["correlation_id"] = "req-\ud800"

    assert h.run(_ascii(document)) == 0

    completed = h.producer.of_type("solve.completed")[0]
    assert completed["data"]["status"] == "MODEL_INVALID"
    assert completed["data"]["error"] == {"code": "invalid_payload"}
    assert completed["correlation_id"] == JOB_ID  # the echo cannot break the reply
    assert h.engine.calls == []
    assert h.consumer.committed == [7]


def test_lone_surrogate_in_payload_is_rejected_not_crashed() -> None:
    h = harness()
    document = with_data(build_request(), lambda d: d["classes"][0].update(id="c-\ud800"))

    assert h.run(_ascii(document)) == 0

    data = h.producer.of_type("solve.completed")[0]["data"]
    assert data["status"] == "MODEL_INVALID"
    assert data["error"] == {"code": "invalid_payload"}
    assert h.engine.calls == []
    assert h.consumer.committed == [7]
    json.loads(h.producer.dlq()[0].value or b"")  # the DLQ copy is still valid JSON


def test_lone_surrogate_in_poison_still_gets_a_dlq_copy() -> None:
    h = harness()

    assert h.run(_ascii({"school_id": "\ud800"})) == 0

    assert h.consumer.committed == [7]
    assert len(h.producer.dlq()) == 1


def test_any_failure_building_the_dlq_copy_is_logged_and_committed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*_: object, **__: object) -> object:
        raise MemoryError

    monkeypatch.setattr("lad_solver.adapters.kafka.worker.dlq_message", broken)
    h = harness()

    assert h.run(b"not json") == 0

    assert h.consumer.committed == [7]
    assert len(_events(capsys.readouterr().out, "dlq.failed")) == 1


# ---------------------------------------------------------------- 10. stall stop and timings


def test_wait_loop_ticks_the_no_improvement_stop(capsys: pytest.CaptureFixture[str]) -> None:
    class PlateauEngine(FakeEngine):
        """Reports one solution, then waits for the worker's stall check to stop it."""

        def __call__(
            self, problem: Problem, params: Params, stop: StopHandle, progress: Progress
        ) -> Result:
            assert params.stall_min_s == 0.2
            stopped = threading.Event()

            class Stopper:
                def stop_search(self) -> None:
                    stopped.set()

            progress.begin(StallPolicy(min_s=0.2, ratio=1.0))
            progress.attach(Stopper())
            progress.record(10)
            self.stop_seen = stopped.wait(timeout=10)
            progress.detach()
            return super().__call__(problem, params, stop, progress)

    engine = PlateauEngine(Journal())
    h = harness(engine=engine, stall_min_s=0.2)

    assert h.run(encode(build_request())) == 0

    assert engine.stop_seen  # the worker's main loop ran progress.check()
    assert engine.calls[0][1].stall_min_s is not None
    started = _events(capsys.readouterr().out, "job.started")[0]
    assert started["stall_min_s"] == 0.2


def test_stall_is_off_by_default_in_the_worker() -> None:
    h = harness()

    h.run(encode(build_request()))

    assert h.engine.calls[0][1].stall_min_s is None


def test_completed_log_carries_processing_and_search_timings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = feasible_result()
    timed = replace(base, stats=replace(base.stats, first_solution_s=0.2, last_improvement_s=0.9))
    h = harness(engine=FakeEngine(Journal(), result=timed))

    h.run(encode(build_request()))

    completed = _events(capsys.readouterr().out, "job.completed")[0]
    for key in ("handle_ms", "wall_time_ms", "first_solution_ms", "last_improvement_ms", "gap"):
        assert key in completed, key
    assert (completed["first_solution_ms"], completed["last_improvement_ms"]) == (200, 900)
    assert completed["gap"] == round((150 - 140) / 150, 4)
    assert isinstance(completed["handle_ms"], int)
    assert completed["handle_ms"] >= completed["wall_time_ms"]


def test_poison_log_carries_the_processing_time(capsys: pytest.CaptureFixture[str]) -> None:
    h = harness()

    h.run(b"not json")

    done = _events(capsys.readouterr().out, "message.done")
    assert len(done) == 1
    assert isinstance(done[0]["handle_ms"], int)


def test_a_failing_stall_check_is_logged_and_does_not_fail_the_job(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(_self: Progress) -> bool:
        raise RuntimeError

    monkeypatch.setattr(Progress, "check", broken)
    release = threading.Event()
    timer = threading.Timer(1.2, release.set)  # the engine outlives at least two wait ticks
    h = harness(engine=FakeEngine(Journal(), ignore_stop=release), stall_min_s=8.0)
    timer.start()
    try:
        assert h.run(encode(build_request())) == 0
    finally:
        timer.cancel()

    assert h.producer.of_type("solve.completed")[0]["data"]["status"] == "FEASIBLE"
    failed = _events(capsys.readouterr().out, "progress.check_failed")
    assert len(failed) == 1  # logged once, then the rule is off for this job


# ---------------------------------------------------------------- audit: retries, fatal DLQ, logs


def test_stop_during_a_retry_pause_cuts_the_remaining_retries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    h = harness()
    h.consumer.on_idle = None
    h.producer.fail_types["solve.started"] = 99

    def sleep_then_stop(seconds: float) -> None:
        h.sleeps.append(seconds)
        h.worker.request_stop()  # SIGTERM lands inside the first pause

    h.worker.sleep = sleep_then_stop

    assert h.run(encode(build_request())) == 1

    assert h.sleeps == [1.0]  # no 2 s and 4 s pauses on top of the stop budget
    assert h.consumer.committed == []
    [gave_up] = _events(capsys.readouterr().out, "solve.started.gave_up")
    assert (gave_up["attempt"], gave_up["reason"]) == (2, "flush")


def test_retry_and_unhealthy_logs_carry_the_kafka_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    h = harness()
    h.consumer.poll_errors = 30
    h.producer.fail_types["solve.started"] = 1

    assert h.run(encode(build_request())) == 0

    out = capsys.readouterr().out
    assert _events(out, "solve.started.retry")[0]["reason"] == "flush"
    assert _events(out, "consumer.unhealthy")[0]["reason"] == "consumer"


def test_fatal_kafka_error_on_the_dlq_copy_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    h = harness()
    h.producer.fatal_types.add("dlq")

    assert h.run(b"not json") == 1

    out = capsys.readouterr().out
    assert _events(out, "kafka.fatal")
    assert not _events(out, "dlq.failed")
    assert h.consumer.committed == []  # redelivered after the restart


def test_engine_failure_log_has_no_search_telemetry(capsys: pytest.CaptureFixture[str]) -> None:
    h = harness(engine=FakeEngine(Journal(), error=RuntimeError("boom")))

    assert h.run(encode(build_request())) == 0

    [completed] = _events(capsys.readouterr().out, "job.completed")
    assert completed["error_code"] == "internal"
    assert "wall_time_ms" in completed
    for field_name in ("stop_reason", "num_workers", "quality_relaxed", "objective"):
        assert field_name not in completed
