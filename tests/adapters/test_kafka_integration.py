"""Worker against a real broker (apache/kafka, KRaft, single node) in Docker.

testcontainers' KafkaContainer targets confluentinc/cp-kafka, so the apache/kafka image (the one
platform runs) is started as a generic container: the advertised listener must equal the host
port, hence a fixed free port chosen up front. Waits are deadline-bounded polls, never sleeps.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from confluent_kafka import Consumer, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic  # pyright: ignore[reportPrivateImportUsage]
from kafka_support import (
    SCHOOL_ID,
    FakeEngine,
    Journal,
    Json,
    build_request,
    encode,
)
from testcontainers.core.container import DockerContainer

from lad_solver.adapters.kafka.config import WorkerConfig
from lad_solver.adapters.kafka.kafka_io import KafkaConsumer, KafkaProducer
from lad_solver.adapters.kafka.ports import Engine
from lad_solver.adapters.kafka.worker import Worker
from lad_solver.app.solve import solve

pytestmark = pytest.mark.integration

KAFKA_IMAGE = "apache/kafka:4.3.2"  # keep equal to the broker tag in platform compose
DEADLINE_S = 60.0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_until[T](probe: Callable[[], T | None], what: str, timeout_s: float = DEADLINE_S) -> T:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = probe()
        if value is not None:
            return value
    pytest.fail(f"timed out waiting for {what}")


@pytest.fixture(scope="session")
def bootstrap() -> Iterator[str]:
    port = _free_port()
    container = (
        DockerContainer(KAFKA_IMAGE)
        .with_bind_ports(9092, port)
        .with_env("KAFKA_NODE_ID", "1")
        .with_env("KAFKA_PROCESS_ROLES", "broker,controller")
        .with_env("KAFKA_LISTENERS", "PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093")
        .with_env("KAFKA_ADVERTISED_LISTENERS", f"PLAINTEXT://127.0.0.1:{port}")
        .with_env("KAFKA_CONTROLLER_LISTENER_NAMES", "CONTROLLER")
        .with_env(
            "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP", "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT"
        )
        .with_env("KAFKA_CONTROLLER_QUORUM_VOTERS", "1@localhost:9093")
        .with_env("KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR", "1")
        .with_env("KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR", "1")
        .with_env("KAFKA_TRANSACTION_STATE_LOG_MIN_ISR", "1")
        .with_env("KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS", "0")
        .with_env("KAFKA_AUTO_CREATE_TOPICS_ENABLE", "false")
    )
    with container:
        servers = f"127.0.0.1:{port}"
        admin = AdminClient({"bootstrap.servers": servers})

        def ready() -> bool | None:
            try:
                admin.list_topics(timeout=1)
            except Exception:  # noqa: BLE001 -- broker still starting
                return None
            return True

        _wait_until(ready, "the broker")
        yield servers


@dataclass(frozen=True)
class Topics:
    build: str
    status: str
    dlq: str
    group: str


@pytest.fixture
def topics(bootstrap: str) -> Topics:
    suffix = uuid.uuid4().hex[:8]
    names = Topics(
        f"solve.build.requested.{suffix}",
        f"solve.status.{suffix}",
        f"solve.build.requested.{suffix}.dlq",
        f"lad-solver-{suffix}",
    )
    admin = AdminClient({"bootstrap.servers": bootstrap})
    futures = admin.create_topics(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        [
            NewTopic(name, num_partitions=1, replication_factor=1)
            for name in (names.build, names.status, names.dlq)
        ]
    )
    for future in futures.values():  # pyright: ignore[reportUnknownVariableType]
        future.result(timeout=30)  # pyright: ignore[reportUnknownMemberType]
    return names


def _config(bootstrap: str, topics: Topics) -> WorkerConfig:
    config = WorkerConfig(
        bootstrap_servers=bootstrap,
        group_id=topics.group,
        build_topic=topics.build,
        status_topic=topics.status,
        build_dlq_topic=topics.dlq,
    )
    config.validate()
    return config


def _send(bootstrap: str, topic: str, value: bytes) -> None:
    producer = Producer({"bootstrap.servers": bootstrap})
    producer.produce(topic, value=value, key=SCHOOL_ID.encode())
    assert producer.flush(30) == 0


@dataclass(frozen=True)
class Received:
    key: bytes | None
    partition: int
    value: bytes | None
    headers: dict[str, bytes]

    def json(self) -> Json:
        assert self.value is not None
        return json.loads(self.value)


def _read(bootstrap: str, topic: str, count: int) -> list[Received]:
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"reader-{uuid.uuid4().hex}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    received: list[Received] = []
    try:

        def enough() -> bool | None:
            message = consumer.poll(0.5)
            if message is not None and message.error() is None:
                headers = {k: v for k, v in (message.headers() or []) if isinstance(v, bytes)}
                partition = message.partition()
                assert partition is not None
                received.append(Received(message.key(), partition, message.value(), headers))
            return True if len(received) >= count else None

        _wait_until(enough, f"{count} message(s) on {topic}")
    finally:
        consumer.close()
    return received


def _committed_offset(bootstrap: str, topics: Topics) -> int:
    consumer = Consumer({"bootstrap.servers": bootstrap, "group.id": topics.group})
    try:
        [partition] = consumer.committed([TopicPartition(topics.build, 0)], timeout=30)
        return partition.offset
    finally:
        consumer.close()


@dataclass
class RunningWorker:
    worker: Worker
    thread: threading.Thread
    exit_codes: list[int]

    def stop(self) -> int:
        self.worker.request_stop()
        self.thread.join(timeout=DEADLINE_S)
        assert not self.thread.is_alive()
        return self.exit_codes[0]


def _start_worker(bootstrap: str, topics: Topics, engine: Engine) -> RunningWorker:
    config = _config(bootstrap, topics)
    worker = Worker(
        consumer=KafkaConsumer(config, [config.build_topic]),
        producer=KafkaProducer(config),
        engine=engine,
        config=config,
    )
    exit_codes: list[int] = []
    thread = threading.Thread(target=lambda: exit_codes.append(worker.run()), daemon=True)
    thread.start()
    return RunningWorker(worker, thread, exit_codes)


def _future_deadline() -> str:
    return "2099-01-01T00:00:00Z"


def test_started_then_completed_keyed_by_school_and_offset_committed(
    bootstrap: str, topics: Topics
) -> None:
    _send(bootstrap, topics.build, encode(build_request(queue_deadline_at=_future_deadline())))
    running = _start_worker(bootstrap, topics, FakeEngine(Journal()))
    try:
        started, completed = _read(bootstrap, topics.status, 2)
    finally:
        assert running.stop() == 0

    assert [started.json()["type"], completed.json()["type"]] == [
        "solve.started",
        "solve.completed",
    ]
    assert started.key == completed.key == SCHOOL_ID.encode()
    assert started.partition == completed.partition
    assert completed.json()["data"]["status"] == "FEASIBLE"
    assert _committed_offset(bootstrap, topics) == 1


def test_real_engine_end_to_end_through_the_broker(bootstrap: str, topics: Topics) -> None:
    """The CP-SAT engine itself behind the worker: request in, full timetable out."""
    _send(bootstrap, topics.build, encode(build_request(queue_deadline_at=_future_deadline())))
    running = _start_worker(bootstrap, topics, solve)
    try:
        _, completed = _read(bootstrap, topics.status, 2)
    finally:
        assert running.stop() == 0

    data = completed.json()["data"]
    assert data["status"] in {"OPTIMAL", "FEASIBLE"}
    assert len(data["lessons"]) == 5 + 2  # every curriculum hour is scheduled
    assert len(data["day_formats"]) == 2 * 5  # every class x school day has a format
    assert _committed_offset(bootstrap, topics) == 1


def test_expired_job_is_closed_without_started(bootstrap: str, topics: Topics) -> None:
    _send(bootstrap, topics.build, encode(build_request(queue_deadline_at="2026-01-01T00:00:00Z")))
    engine = FakeEngine(Journal())
    running = _start_worker(bootstrap, topics, engine)
    try:
        [completed] = _read(bootstrap, topics.status, 1)
        _wait_until(
            lambda: True if _committed_offset(bootstrap, topics) == 1 else None, "the commit"
        )
    finally:
        assert running.stop() == 0

    assert completed.json()["data"]["status"] == "EXPIRED"
    assert engine.calls == []


def test_poison_message_lands_in_dlq_and_is_committed(bootstrap: str, topics: Topics) -> None:
    _send(bootstrap, topics.build, b"definitely not json")
    running = _start_worker(bootstrap, topics, FakeEngine(Journal()))
    try:
        [dead] = _read(bootstrap, topics.dlq, 1)
        _wait_until(
            lambda: True if _committed_offset(bootstrap, topics) == 1 else None, "the commit"
        )
    finally:
        assert running.stop() == 0

    assert dead.value == b"definitely not json"
    assert dead.headers["x-error-class"] == b"not_json"
    assert dead.headers["x-original-offset"] == b"0"


_SLOW_WORKER = """
import sys, threading
from lad_solver.adapters.kafka import jsonlog
from lad_solver.adapters.kafka.config import WorkerConfig
from lad_solver.adapters.kafka.kafka_io import KafkaConsumer, KafkaProducer
from lad_solver.adapters.kafka.worker import Worker

def never_returns(problem, params, stop):
    threading.Event().wait()

jsonlog.configure()
bootstrap, build, status, dlq, group = sys.argv[1:6]
config = WorkerConfig(bootstrap_servers=bootstrap, group_id=group, build_topic=build,
                      status_topic=status, build_dlq_topic=dlq)
config.validate()
worker = Worker(KafkaConsumer(config, [build]), KafkaProducer(config), never_returns, config)
sys.exit(worker.run())
"""


@pytest.mark.slow
def test_kill_9_during_solve_does_not_redeliver_the_job(bootstrap: str, topics: Topics) -> None:
    """testing.md: kill -9 during a solve -> the job is not redelivered, the next one runs.

    The `failed` transition of job A belongs to timetable's watchdog (platform smoke-e2e).
    """
    job_a = build_request(queue_deadline_at=_future_deadline())
    _send(bootstrap, topics.build, encode(job_a))
    args = [bootstrap, topics.build, topics.status, topics.dlq, topics.group]
    process = subprocess.Popen(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", _SLOW_WORKER, *args],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ.copy(),
    )
    try:
        [started_a] = _read(bootstrap, topics.status, 1)
        assert started_a.json()["type"] == "solve.started"
        _wait_until(lambda: True if _committed_offset(bootstrap, topics) == 1 else None, "A+1")
    finally:
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=30)

    job_b = build_request(queue_deadline_at=_future_deadline())
    job_b["data"]["job_id"] = str(uuid.uuid4())
    _send(bootstrap, topics.build, encode(job_b))
    running = _start_worker(bootstrap, topics, FakeEngine(Journal()))
    try:
        events = _read(bootstrap, topics.status, 3)
    finally:
        assert running.stop() == 0

    after_kill = [(e.json()["type"], e.json()["data"]["job_id"]) for e in events[1:]]
    assert after_kill == [
        ("solve.started", job_b["data"]["job_id"]),
        ("solve.completed", job_b["data"]["job_id"]),
    ]
    assert _committed_offset(bootstrap, topics) == 2
