"""confluent-kafka adapters on a fake client: no duplicate re-produce, honest close."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from kafka_support import config

from lad_solver.adapters.kafka import kafka_io
from lad_solver.adapters.kafka.ports import FatalKafkaError, OutboundMessage, TransientKafkaError

type Report = Callable[[Any, Any], None]


@dataclass
class _Error:
    code: str
    fatal_: bool = False

    def name(self) -> str:
        return self.code

    def fatal(self) -> bool:
        return self.fatal_


@dataclass
class FakeClient:
    """Queue of produced messages; ``deliver`` decides what the next flush reports."""

    produced: list[bytes] = field(default_factory=list[bytes])
    pending: list[tuple[bytes, Report]] = field(default_factory=list[tuple[bytes, Report]])
    outcomes: list[str | None] = field(default_factory=list[str | None])  # per flush: None=ok
    flush_timeouts: list[float] = field(default_factory=list[float])

    def produce(self, _topic: str, *, value: bytes, on_delivery: Report, **_: object) -> None:
        self.produced.append(value)
        self.pending.append((value, on_delivery))

    def flush(self, timeout: float) -> int:
        self.flush_timeouts.append(timeout)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome == "stuck":
            return len(self.pending)
        reports, self.pending = self.pending, []
        for _value, report in reports:
            report(None if outcome is None else _Error(outcome), None)
        return 0


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient()

    def build(_conf: dict[str, Any]) -> FakeClient:
        return fake

    monkeypatch.setattr(kafka_io, "Producer", build)
    return fake


MESSAGE = OutboundMessage("solve.status", b"k", b"v1")


def test_flush_timeout_does_not_re_produce_queued_messages(client: FakeClient) -> None:
    producer = kafka_io.KafkaProducer(config())
    client.outcomes = ["stuck", None]

    with pytest.raises(TransientKafkaError):
        producer.send([MESSAGE], 1.0)
    producer.send([MESSAGE], 1.0)  # the retry only flushes again

    assert client.produced == [b"v1"]


def test_delivery_error_is_re_produced_on_retry(client: FakeClient) -> None:
    producer = kafka_io.KafkaProducer(config())
    client.outcomes = ["_MSG_TIMED_OUT", None]

    with pytest.raises(TransientKafkaError):
        producer.send([MESSAGE], 1.0)
    producer.send([MESSAGE], 1.0)

    assert client.produced == [b"v1", b"v1"]


def test_close_reports_undelivered_count(client: FakeClient) -> None:
    producer = kafka_io.KafkaProducer(config())
    client.outcomes = ["stuck", "stuck"]
    with pytest.raises(TransientKafkaError):
        producer.send([MESSAGE], 1.0)

    assert producer.close(2.0) == 1
    assert client.flush_timeouts[-1] == 2.0


# ---------------------------------------------------------------- consumer errors


@dataclass
class _Message:
    error_: _Error | None

    def error(self) -> _Error | None:
        return self.error_


@dataclass
class FakeConsumerClient:
    script: list[_Message | str | None]
    conf: dict[str, Any] = field(default_factory=dict[str, Any])

    def subscribe(self, _topics: list[str]) -> None:
        return None

    def poll(self, _timeout: float) -> _Message | None:
        item = self.script.pop(0)
        if item == "error_cb":
            self.conf["error_cb"](_Error("_TRANSPORT"))
            return None
        assert not isinstance(item, str)
        return item


def _consumer(monkeypatch: pytest.MonkeyPatch, script: list[_Message | str | None]) -> Any:  # noqa: ANN401
    def build(conf: dict[str, Any]) -> FakeConsumerClient:
        return FakeConsumerClient(script, conf)

    monkeypatch.setattr(kafka_io, "Consumer", build)
    return kafka_io.KafkaConsumer(config(), ["solve.build.requested"])


def test_consumer_error_without_a_message_raises_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer = _consumer(monkeypatch, [_Message(_Error("_UNKNOWN_TOPIC")), "error_cb", None])

    with pytest.raises(TransientKafkaError):
        consumer.poll(1.0)
    with pytest.raises(TransientKafkaError):
        consumer.poll(1.0)
    assert consumer.poll(1.0) is None  # a clean idle poll


def test_fatal_consumer_error_still_raises_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer = _consumer(monkeypatch, [_Message(_Error("_FENCED", fatal_=True))])

    with pytest.raises(FatalKafkaError):
        consumer.poll(1.0)
