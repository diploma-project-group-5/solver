"""Builders and in-memory fakes for the Kafka adapter tests (no broker, no real sleeps)."""

import copy
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lad_solver.adapters.kafka.config import WorkerConfig
from lad_solver.adapters.kafka.ports import InboundMessage, OutboundMessage, TransientKafkaError
from lad_solver.adapters.kafka.worker import Worker
from lad_solver.domain.problem import Params, Problem
from lad_solver.domain.result import (
    Conflict,
    ConflictRule,
    DayFormat,
    Lesson,
    Result,
    SolveStatus,
    Stats,
    StopReason,
)
from lad_solver.model.stop import StopHandle

NOW = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)
SCHOOL_ID = "7b0f8c1e-3d2a-4c51-9a6e-1f2b3c4d5e6f"
JOB_ID = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
BUILD_TOPIC = "solve.build.requested"
STATUS_TOPIC = "solve.status"
DLQ_TOPIC = "solve.build.requested.dlq"

type Json = dict[str, Any]


def build_request(**data_overrides: object) -> Json:
    """A valid solve.build.requested envelope (kind=main) with an id-only snapshot."""
    data: Json = {
        "job_id": JOB_ID,
        "kind": "main",
        "model": "1.1",
        "snapshot_version": 42,
        "time_limit_s": 60,
        "queue_deadline_at": "2026-11-04T10:10:00Z",
        "school_days": [1, 2, 3, 4, 5],
        "lessons_per_day": 6,
        "staff": 4,
        "capacity": {"passport": 60},
        "classes": [
            {"id": "c-5a", "students": 28, "priority": True},
            {"id": "c-9b", "students": 24, "priority": False},
        ],
        "subjects": [
            {"id": "s-math", "priority_area": True, "no_shelter": False},
            {"id": "s-pe", "priority_area": False, "no_shelter": True},
        ],
        "teachers": [
            {"id": "t-1", "weekly_limit": 20, "unavailable_days": []},
            {"id": "t-2", "weekly_limit": 18, "unavailable_days": [3]},
        ],
        "rows": [
            {"class_id": "c-5a", "subject_id": "s-math", "teacher_id": "t-1", "hours": 5},
            {"class_id": "c-9b", "subject_id": "s-pe", "teacher_id": "t-2", "hours": 2},
        ],
    }
    data.update(data_overrides)
    return {
        "event_id": "6f1d2c3b-4a5e-4f60-8b71-0c2d3e4f5a6b",
        "type": "solve.build.requested",
        "schema_version": "1.0",
        "source": "timetable-service",
        "time": "2026-11-04T09:59:00Z",
        "correlation_id": "req-5f2c9a",
        "school_id": SCHOOL_ID,
        "data": data,
    }


def encode(document: Json) -> bytes:
    return json.dumps(document, ensure_ascii=False).encode()


def inbound(value: bytes | None, offset: int = 7) -> InboundMessage:
    return InboundMessage(BUILD_TOPIC, 0, offset, SCHOOL_ID.encode(), value)


def with_data(document: Json, mutate: Callable[[Json], None]) -> Json:
    changed = copy.deepcopy(document)
    mutate(changed["data"])
    return changed


def feasible_result() -> Result:
    return Result(
        status=SolveStatus.FEASIBLE,
        lessons=(Lesson("c-5a", 1, 1, "s-math", "t-1"), Lesson("c-9b", 2, 1, "s-pe", "t-2")),
        day_formats=(DayFormat("c-5a", 1, in_person=True), DayFormat("c-9b", 2, in_person=False)),
        conflicts=(Conflict(ConflictRule.PACKING, teacher_id="t-1", hint=True),),
        stats=Stats(
            objective=140,
            best_bound=150.0,
            primary=28,
            secondary=4,
            wall_time_s=1.5,
            stop_reason=StopReason.TIME_LIMIT,
            num_workers=4,
            random_seed=0,
        ),
    )


def decode(message: OutboundMessage) -> Json:
    assert message.value is not None
    return json.loads(message.value)


@dataclass
class Journal:
    """Ordered record of side effects across all fakes."""

    entries: list[str] = field(default_factory=list[str])


@dataclass
class FakeConsumer:
    journal: Journal
    messages: deque[InboundMessage] = field(default_factory=deque[InboundMessage])
    commit_failures: int = 0
    poll_errors: int = 0  # polls that fail with a (non-fatal) consumer error before delivering
    on_idle: Callable[[], None] | None = None
    before_deliver: Callable[[], None] | None = None  # runs inside poll() just before a message
    committed: list[int] = field(default_factory=list[int])
    closed: bool = False

    def poll(self, timeout_s: float) -> InboundMessage | None:
        assert timeout_s > 0
        if self.poll_errors > 0:
            self.poll_errors -= 1
            msg = "consumer"
            raise TransientKafkaError(msg)
        if self.messages:
            if self.before_deliver is not None:
                self.before_deliver()
            return self.messages.popleft()
        if self.on_idle is not None:
            self.on_idle()
        return None

    def commit(self, message: InboundMessage) -> None:
        if self.commit_failures > 0:
            self.commit_failures -= 1
            self.journal.entries.append("commit-failed")
            msg = "commit"
            raise TransientKafkaError(msg)
        self.committed.append(message.offset)
        self.journal.entries.append("commit")

    def close(self) -> None:
        self.closed = True


def _label(message: OutboundMessage) -> str:
    if message.topic == DLQ_TOPIC:
        return "dlq"
    return str(decode(message)["type"])


@dataclass
class FakeProducer:
    journal: Journal
    fail_types: dict[str, int] = field(default_factory=dict[str, int])
    on_poll: Callable[[], None] | None = None
    sent: list[OutboundMessage] = field(default_factory=list[OutboundMessage])
    send_timeouts: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])
    close_timeouts: list[float] = field(default_factory=list[float])
    batches: list[list[str]] = field(default_factory=list[list[str]])  # labels per send() call
    undelivered_on_close: int = 0
    polls: int = 0
    closed: bool = False

    def send(self, messages: list[OutboundMessage], timeout_s: float) -> None:
        assert timeout_s > 0
        self.send_timeouts.extend((_label(m), timeout_s) for m in messages)
        self.batches.append([_label(m) for m in messages])
        for message in messages:
            label = _label(message)
            if self.fail_types.get(label, 0) > 0:
                self.fail_types[label] -= 1
                self.journal.entries.append(f"send-failed:{label}")
                msg = "flush"
                raise TransientKafkaError(msg)
        for message in messages:
            self.sent.append(message)
            self.journal.entries.append(f"send:{_label(message)}")

    def poll(self, timeout_s: float) -> None:
        assert timeout_s == 0
        self.polls += 1
        if self.on_poll is not None:
            self.on_poll()

    def close(self, timeout_s: float) -> int:
        assert timeout_s > 0
        self.close_timeouts.append(timeout_s)
        self.closed = True
        return self.undelivered_on_close

    def of_type(self, event_type: str) -> list[Json]:
        return [decode(m) for m in self.sent if m.topic == STATUS_TOPIC and _label(m) == event_type]

    def dlq(self) -> list[OutboundMessage]:
        return [m for m in self.sent if m.topic == DLQ_TOPIC]


@dataclass
class FakeEngine:
    journal: Journal
    result: Result = field(default_factory=feasible_result)
    error: Exception | None = None
    wait_for_stop: bool = False
    ignore_stop: threading.Event | None = None
    calls: list[tuple[Problem, Params]] = field(default_factory=list[tuple[Problem, Params]])
    stop_seen: bool = False

    def __call__(self, problem: Problem, params: Params, stop: StopHandle) -> Result:
        self.calls.append((problem, params))
        self.journal.entries.append("solve")
        if self.ignore_stop is not None:
            self.ignore_stop.wait(timeout=30)
        if self.wait_for_stop:
            deadline = time.monotonic() + 10
            while not stop.stopped and time.monotonic() < deadline:
                time.sleep(0.005)
            self.stop_seen = stop.stopped
        if self.error is not None:
            raise self.error
        return self.result


def config(**overrides: Any) -> WorkerConfig:  # noqa: ANN401 -- forwarded to the dataclass
    values: dict[str, Any] = {
        "bootstrap_servers": "fake:9092",
        "build_topic": BUILD_TOPIC,
        "status_topic": STATUS_TOPIC,
        "build_dlq_topic": DLQ_TOPIC,
        "heartbeat_file": Path("/nonexistent/heartbeat"),
    } | overrides
    built = WorkerConfig(**values)
    built.validate()
    return built


@dataclass
class Harness:
    worker: Worker
    consumer: FakeConsumer
    producer: FakeProducer
    engine: FakeEngine
    journal: Journal
    sleeps: list[float]
    exits: list[int]
    beats: list[int]

    def run(self, *values: bytes | None) -> int:
        for offset, value in enumerate(values, start=7):
            self.consumer.messages.append(inbound(value, offset))
        return self.worker.run()


def harness(
    *,
    engine: FakeEngine | None = None,
    clock: datetime = NOW,
    **config_overrides: Any,  # noqa: ANN401 -- forwarded to config()
) -> Harness:
    journal = Journal()
    consumer = FakeConsumer(journal)
    producer = FakeProducer(journal)
    engine = engine or FakeEngine(journal)
    engine.journal = journal
    sleeps: list[float] = []
    exits: list[int] = []
    beats: list[int] = []
    worker = Worker(
        consumer=consumer,
        producer=producer,
        engine=engine,
        config=config(**config_overrides),
        clock=lambda: clock,
        sleep=sleeps.append,
        hard_exit=exits.append,
        heartbeat=lambda: beats.append(1),
    )
    consumer.on_idle = worker.request_stop  # stop when the queue is drained
    return Harness(worker, consumer, producer, engine, journal, sleeps, exits, beats)
