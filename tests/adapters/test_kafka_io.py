"""confluent-kafka adapters on a fake client: no duplicate re-produce, honest close."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from confluent_kafka import KafkaError, KafkaException
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
    # per flush: None=ok, an error name for every message, or a per-value mapping
    outcomes: list[str | dict[bytes, str | None] | None] = field(
        default_factory=list[str | dict[bytes, str | None] | None]
    )
    flush_timeouts: list[float] = field(default_factory=list[float])
    produce_error: BaseException | None = None

    def produce(self, _topic: str, *, value: bytes, on_delivery: Report, **_: object) -> None:
        if self.produce_error is not None:
            raise self.produce_error
        self.produced.append(value)
        self.pending.append((value, on_delivery))

    def flush(self, timeout: float) -> int:
        self.flush_timeouts.append(timeout)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome == "stuck":
            return len(self.pending)
        reports, self.pending = self.pending, []
        for value, report in reports:
            error = outcome.get(value) if isinstance(outcome, dict) else outcome
            report(None if error is None else _Error(error), None)
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


def test_partly_failed_batch_re_produces_only_the_failed_message(client: FakeClient) -> None:
    producer = kafka_io.KafkaProducer(config())
    second = OutboundMessage("solve.status", b"k", b"v2")
    client.outcomes = [{b"v1": None, b"v2": "_MSG_TIMED_OUT"}, None]

    with pytest.raises(TransientKafkaError, match="_MSG_TIMED_OUT"):
        producer.send([MESSAGE, second], 1.0)
    producer.send([MESSAGE, second], 1.0)

    assert client.produced == [b"v1", b"v2", b"v2"]  # v1 was delivered: never produced again


def test_a_failed_one_shot_send_does_not_stay_in_the_producer_state(client: FakeClient) -> None:
    producer = kafka_io.KafkaProducer(config())
    dlq_copy = OutboundMessage("solve.build.requested.dlq", b"k", b"payload")
    client.outcomes = ["_MSG_TIMED_OUT", None]

    with pytest.raises(TransientKafkaError):
        producer.send([dlq_copy], 1.0)  # best effort: the worker never retries it
    producer.send([MESSAGE], 1.0)

    assert producer._state == {}  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_fatal_produce_error_is_fatal(client: FakeClient) -> None:
    producer = kafka_io.KafkaProducer(config())
    client.produce_error = KafkaException(KafkaError(KafkaError._FATAL))  # noqa: SLF001

    with pytest.raises(FatalKafkaError, match="_FATAL"):
        producer.send([MESSAGE], 1.0)


def test_produce_error_flagged_fatal_is_fatal(client: FakeClient) -> None:
    producer = kafka_io.KafkaProducer(config())
    client.produce_error = KafkaException(KafkaError(KafkaError._STATE, fatal=True))  # noqa: SLF001

    with pytest.raises(FatalKafkaError):
        producer.send([MESSAGE], 1.0)


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (KafkaException(KafkaError(KafkaError.MSG_SIZE_TOO_LARGE)), "MSG_SIZE_TOO_LARGE"),
        (BufferError("queue full"), "BufferError"),
    ],
)
def test_other_produce_errors_are_transient_with_their_name(
    client: FakeClient, error: BaseException, reason: str
) -> None:
    producer = kafka_io.KafkaProducer(config())
    client.produce_error = error

    with pytest.raises(TransientKafkaError) as caught:
        producer.send([MESSAGE], 1.0)

    assert str(caught.value) == reason


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


def test_consumer_config_leaves_the_error_callback_to_the_consumer() -> None:
    assert "error_cb" not in kafka_io.consumer_config(config())


# ---------------------------------------------------------------- error callback


@dataclass
class _Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


def _kafka_errors(out: str) -> list[dict[str, Any]]:
    entries = [json.loads(line) for line in out.splitlines()]
    return [e for e in entries if e["msg"] == "kafka.error"]


def test_repeated_errors_are_logged_once_per_minute_with_a_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = _Clock()
    handler = kafka_io.KafkaErrorHandler(clock=clock)

    for _ in range(19):  # one second of a broker outage
        handler(KafkaError(KafkaError._TRANSPORT))  # noqa: SLF001
        clock.now += 1 / 19
    handler(KafkaError(KafkaError._ALL_BROKERS_DOWN))  # noqa: SLF001
    clock.now = 61.0
    handler(KafkaError(KafkaError._TRANSPORT))  # noqa: SLF001

    logged = _kafka_errors(capsys.readouterr().out)
    assert [(e["error_class"], e.get("count")) for e in logged] == [
        ("_TRANSPORT", None),
        ("_ALL_BROKERS_DOWN", None),
        ("_TRANSPORT", 18),
    ]


def test_fatal_errors_are_always_logged_and_raised(capsys: pytest.CaptureFixture[str]) -> None:
    handler = kafka_io.KafkaErrorHandler(clock=_Clock())

    for _ in range(2):
        with pytest.raises(FatalKafkaError, match="_FENCED"):
            handler(KafkaError(KafkaError._FENCED, fatal=True))  # noqa: SLF001

    assert len(_kafka_errors(capsys.readouterr().out)) == 2
