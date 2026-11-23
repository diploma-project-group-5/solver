"""SIGTERM on the built image (Trello #73): ``docker stop`` mid-solve -> best solution, exit 0.

The in-process tests cover the stop logic; this one covers what only the container shows: tini
(compose ``init: true``, here ``--init``) forwards the signal to Python, the worker returns its best
timetable and exits within compose's ``stop_grace_period`` (30 s) instead of being SIGKILLed.

The image is ``SOLVER_TEST_IMAGE`` (CI passes the one it built) or is built from this repo.
The broker has two listeners: one advertised to the host (this test) and one on the container
network (the solver). Waits are deadline-bounded on events, never fixed sleeps.
"""

import json
import os
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from confluent_kafka import Consumer, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic  # pyright: ignore[reportPrivateImportUsage]
from kafka_support import SCHOOL_ID, Json, build_request, encode
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network

from lad_solver.cli.generate import generate_problem
from lad_solver.cli.io import problem_to_json

pytestmark = [pytest.mark.integration, pytest.mark.image]

KAFKA_IMAGE = "apache/kafka:4.3.2"  # keep equal to the broker tag in platform compose
REPO = Path(__file__).resolve().parents[2]
STOP_GRACE_PERIOD_S = 30  # platform compose: stop_grace_period
DEADLINE_S = 120.0
POLL_S = 0.5


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


@pytest.fixture(scope="module")
def image() -> str:
    tag = os.environ.get("SOLVER_TEST_IMAGE")
    if tag:
        return tag
    tag = "lad-solver:image-test"
    subprocess.run(  # noqa: S603 -- fixed argv
        ["docker", "build", "--quiet", "--tag", tag, str(REPO)],  # noqa: S607 -- docker on PATH
        check=True,
        capture_output=True,
        timeout=900,
    )
    return tag


@dataclass(frozen=True)
class Broker:
    network: Network
    host_servers: str  # for this test
    network_servers: str  # for containers on the same network


@pytest.fixture(scope="module")
def broker() -> Iterator[Broker]:
    port = _free_port()
    with Network() as network:
        container = (
            DockerContainer(KAFKA_IMAGE)
            .with_network(network)
            .with_network_aliases("kafka")
            .with_bind_ports(9092, port)
            .with_env("KAFKA_NODE_ID", "1")
            .with_env("KAFKA_PROCESS_ROLES", "broker,controller")
            .with_env(
                "KAFKA_LISTENERS",
                "PLAINTEXT://0.0.0.0:9092,INTERNAL://0.0.0.0:19092,CONTROLLER://0.0.0.0:9093",
            )
            .with_env(
                "KAFKA_ADVERTISED_LISTENERS",
                f"PLAINTEXT://127.0.0.1:{port},INTERNAL://kafka:19092",
            )
            .with_env("KAFKA_INTER_BROKER_LISTENER_NAME", "INTERNAL")
            .with_env("KAFKA_CONTROLLER_LISTENER_NAMES", "CONTROLLER")
            .with_env(
                "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP",
                "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT,INTERNAL:PLAINTEXT",
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
            yield Broker(network, servers, "kafka:19092")


@dataclass(frozen=True)
class Topics:
    build: str
    status: str
    dlq: str
    group: str


def _topics(servers: str) -> Topics:
    suffix = uuid.uuid4().hex[:8]
    names = Topics(
        f"solve.build.requested.{suffix}",
        f"solve.status.{suffix}",
        f"solve.build.requested.{suffix}.dlq",
        f"lad-solver-{suffix}",
    )
    admin = AdminClient({"bootstrap.servers": servers})
    futures = admin.create_topics(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        [
            NewTopic(name, num_partitions=1, replication_factor=1)
            for name in (names.build, names.status, names.dlq)
        ]
    )
    for future in futures.values():  # pyright: ignore[reportUnknownVariableType]
        future.result(timeout=30)
    return names


def _long_job(job_id: str) -> Json:
    """The demo-sized school (30 classes): its search is still running when the stop arrives."""
    problem = problem_to_json(generate_problem(30, 0.55, seed=0))
    del problem["kind"]
    request = build_request(
        **problem,
        job_id=job_id,
        time_limit_s=120,
        queue_deadline_at="2099-01-01T00:00:00Z",
    )
    request["event_id"] = str(uuid.uuid4())
    return request


def _send(servers: str, topic: str, value: bytes) -> None:
    producer = Producer({"bootstrap.servers": servers})
    producer.produce(topic, value=value, key=SCHOOL_ID.encode())
    assert producer.flush(30) == 0


def _read_status(servers: str, topic: str, count: int) -> list[Json]:
    consumer = Consumer(
        {
            "bootstrap.servers": servers,
            "group.id": f"reader-{uuid.uuid4().hex}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    received: list[Json] = []
    try:

        def enough() -> bool | None:
            message = consumer.poll(0.5)
            if message is not None and message.error() is None:
                value = message.value()
                assert value is not None
                received.append(json.loads(value))
            return True if len(received) >= count else None

        _wait_until(enough, f"{count} message(s) on {topic}")
    finally:
        consumer.close()
    return received


def _committed_offset(servers: str, topics: Topics) -> int:
    consumer = Consumer({"bootstrap.servers": servers, "group.id": topics.group})
    try:
        [partition] = consumer.committed([TopicPartition(topics.build, 0)], timeout=30)
        return partition.offset
    finally:
        consumer.close()


def _wait_for_log(container: Any, needle: str, timeout_s: float = DEADLINE_S) -> None:  # noqa: ANN401 -- docker SDK container
    """Poll the container's stdout until a line contains ``needle`` (or the deadline).

    A followed log stream would need closing from another thread mid-read; polling is simpler.
    """
    tick = threading.Event()  # never set: wait() is the poll interval

    def seen() -> bool | None:
        if needle.encode() in container.logs():
            return True
        tick.wait(POLL_S)
        return None

    _wait_until(seen, f"{needle!r} in the solver log", timeout_s)


def test_docker_stop_mid_solve_publishes_the_best_timetable_and_exits_0(
    image: str, broker: Broker
) -> None:
    topics = _topics(broker.host_servers)
    job_id = str(uuid.uuid4())
    solver = (
        DockerContainer(image)
        .with_network(broker.network)
        .with_kwargs(init=True)  # compose: init: true (tini as PID 1)
        .with_env("KAFKA_BOOTSTRAP_SERVERS", broker.network_servers)
        .with_env("SOLVER_GROUP_ID", topics.group)
        .with_env("SOLVER_BUILD_TOPIC", topics.build)
        .with_env("SOLVER_STATUS_TOPIC", topics.status)
        .with_env("SOLVER_BUILD_DLQ_TOPIC", topics.dlq)
        .with_env("SOLVER_NUM_WORKERS", "2")
        .with_env("SOLVER_MAX_TIME_LIMIT_S", "180")
    )
    with solver:
        container = solver.get_wrapped_container()
        _send(broker.host_servers, topics.build, encode(_long_job(job_id)))
        _wait_for_log(container, '"engine.first_solution"')

        pid1 = container.exec_run(["python", "-c", "print(open('/proc/1/comm').read().strip())"])
        output = pid1.output
        assert isinstance(output, bytes)
        assert output.decode().strip() == "docker-init"  # the signal goes through tini

        stopped_at = time.monotonic()
        container.stop(timeout=STOP_GRACE_PERIOD_S)
        took = time.monotonic() - stopped_at
        container.reload()
        state = container.attrs["State"]
        logs = container.logs().decode(errors="replace")

    assert state["ExitCode"] == 0, f"exit {state['ExitCode']} (137 = SIGKILL):\n{logs[-4000:]}"
    assert took < STOP_GRACE_PERIOD_S
    assert '"worker.stopping"' in logs

    started, completed = _read_status(broker.host_servers, topics.status, 2)
    assert (started["type"], started["data"]["job_id"]) == ("solve.started", job_id)
    assert (completed["type"], completed["data"]["job_id"]) == ("solve.completed", job_id)
    data = completed["data"]
    assert data["status"] in {"FEASIBLE", "OPTIMAL"}
    assert data["stats"]["stop_reason"] == "stopped"
    assert data["lessons"]
    assert data["day_formats"]
    # Committed before the solve (spec 12): a restarted worker does not run the job again.
    assert _committed_offset(broker.host_servers, topics) == 1
