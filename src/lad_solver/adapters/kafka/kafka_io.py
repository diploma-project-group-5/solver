"""confluent-kafka implementations of the worker ports (configs from ADR 2026-11-04 §4)."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from logging import WARNING
from typing import Any, Final

from confluent_kafka import Consumer, KafkaError, KafkaException, Message, Producer

from lad_solver.adapters.kafka import jsonlog
from lad_solver.adapters.kafka.config import WorkerConfig
from lad_solver.adapters.kafka.ports import (
    FatalKafkaError,
    InboundMessage,
    OutboundMessage,
    TransientKafkaError,
)

ERROR_LOG_EVERY_S: Final = 60.0


@dataclass
class KafkaErrorHandler:
    """librdkafka error callback: raises on a fatal error, logs the others rate-limited.

    Called on the main thread (inside poll/flush); the exception propagates out of that call and
    the worker exits with code 1. While the broker is down librdkafka reports an error many times
    a second, so each error name is logged at most once per ``every_s`` with the suppressed count.
    """

    clock: Callable[[], float] = time.monotonic
    every_s: float = ERROR_LOG_EVERY_S
    _seen: dict[str, tuple[float, int]] = field(default_factory=dict[str, tuple[float, int]])

    def __call__(self, error: KafkaError) -> None:
        name = error.name()
        if error.fatal():
            jsonlog.log(WARNING, "kafka.error", error_class=name)
            raise FatalKafkaError(name)
        now = self.clock()
        seen = self._seen.get(name)
        if seen is not None and now - seen[0] < self.every_s:
            self._seen[name] = (seen[0], seen[1] + 1)
            return
        self._seen[name] = (now, 0)
        suppressed = None if seen is None else seen[1]
        jsonlog.log(WARNING, "kafka.error", error_class=name, count=suppressed)


_on_kafka_error: Final = KafkaErrorHandler()


def consumer_config(config: WorkerConfig) -> dict[str, Any]:
    return {
        "bootstrap.servers": config.bootstrap_servers,
        "group.id": config.group_id,
        "enable.auto.commit": False,
        "enable.auto.offset.store": False,
        "auto.offset.reset": "earliest",
        "session.timeout.ms": config.session_timeout_ms,
        "heartbeat.interval.ms": config.heartbeat_interval_ms,
        "max.poll.interval.ms": config.max_poll_interval_ms,
        "queued.max.messages.kbytes": 4096,
        "client.id": "lad-solver",
        # No static membership (group.instance.id) and the default eager assignor: ADR §4.
    }


def producer_config(config: WorkerConfig) -> dict[str, Any]:
    return {
        "bootstrap.servers": config.bootstrap_servers,
        "enable.idempotence": True,  # implies acks=all, bounded in-flight, retries
        "acks": "all",
        "linger.ms": 5,
        "delivery.timeout.ms": config.delivery_timeout_ms,
        "compression.type": "none",  # the KafkaJS consumer in timetable lacks zstd/lz4
        "client.id": "lad-solver",
        "error_cb": _on_kafka_error,
    }


class KafkaConsumer:
    def __init__(self, config: WorkerConfig, topics: list[str]) -> None:
        self._errors = 0
        # error_cb is set here, not in consumer_config(): it also counts errors per poll.
        self._consumer = Consumer(consumer_config(config) | {"error_cb": self._on_error})
        self._consumer.subscribe(topics)

    def _on_error(self, error: KafkaError) -> None:
        self._errors += 1
        _on_kafka_error(error)

    def poll(self, timeout_s: float) -> InboundMessage | None:
        """Raises TransientKafkaError when the poll saw only errors (the worker's health count)."""
        self._errors = 0
        message = self._consumer.poll(timeout_s)
        if message is None:
            if self._errors:
                reason = "consumer_error"
                raise TransientKafkaError(reason)
            return None
        error = message.error()
        if error is not None:
            _on_kafka_error(error)
            raise TransientKafkaError(error.name())
        topic, partition, offset = message.topic(), message.partition(), message.offset()
        if topic is None or partition is None or offset is None:
            return None
        return InboundMessage(topic, partition, offset, message.key(), message.value(), message)

    def commit(self, message: InboundMessage) -> None:
        handle: object = message.handle
        if not isinstance(handle, Message):
            msg = "message was not produced by this consumer"
            raise TypeError(msg)
        try:
            raw: Message[bytes, bytes] = handle  # pyright: ignore[reportUnknownVariableType]
            partitions = self._consumer.commit(message=raw, asynchronous=False)
        except KafkaException as exc:
            raise TransientKafkaError(_kafka_error_name(exc)) from None
        for partition in partitions:
            if partition.error is not None:
                raise TransientKafkaError(partition.error.name())

    def close(self) -> None:
        self._consumer.close()  # leaves the group at once, no session-timeout wait


_PENDING: Final = "pending"  # produced, no delivery report yet
_DELIVERED: Final = "delivered"


class KafkaProducer:
    """``send`` re-produces a message only after a delivery *error*.

    A flush timeout leaves the message queued in librdkafka, which reports it within
    ``delivery.timeout.ms`` either way; producing it again would publish a duplicate.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self._producer = Producer(producer_config(config))
        self._state: dict[OutboundMessage, str] = {}  # pending / delivered / error name

    def _report(self, message: OutboundMessage, error: KafkaError | None, _raw: Message) -> None:
        self._state[message] = _DELIVERED if error is None else error.name()

    def send(self, messages: list[OutboundMessage], timeout_s: float) -> None:
        # The worker sends one batch at a time and retries only that batch, so a final state
        # outside it (e.g. a failed one-shot DLQ copy) is garbage; a pending one waits its report.
        batch = set(messages)
        for stale in [m for m, s in self._state.items() if m not in batch and s != _PENDING]:
            del self._state[stale]
        for message in messages:
            if self._state.get(message) in {_PENDING, _DELIVERED}:
                continue  # queued from an earlier attempt, or already delivered: no duplicate
            try:
                self._producer.produce(
                    message.topic,
                    value=message.value,
                    key=message.key,
                    headers=list(message.headers),
                    on_delivery=partial(self._report, message),
                )
            except KafkaException as exc:
                error: object = exc.args[0] if exc.args else None
                if isinstance(error, KafkaError) and (
                    error.fatal() or error.code() == KafkaError._FATAL  # noqa: SLF001 -- librdkafka code
                ):
                    raise FatalKafkaError(error.name()) from None
                raise TransientKafkaError(_kafka_error_name(exc)) from None
            except BufferError:  # the local queue is full: it drains on the next flush
                raise TransientKafkaError(BufferError.__name__) from None
            self._state[message] = _PENDING  # reports arrive only inside flush/poll
        self._producer.flush(timeout_s)
        states = [self._state[m] for m in messages]
        if all(state == _DELIVERED for state in states):
            for message in messages:
                self._state.pop(message, None)
            return
        failures = [state for state in states if state not in {_PENDING, _DELIVERED}]
        reason = failures[0] if failures else "flush_timeout"
        raise TransientKafkaError(reason)

    def poll(self, timeout_s: float) -> None:
        self._producer.poll(timeout_s)

    def close(self, timeout_s: float) -> int:
        return self._producer.flush(timeout_s)


def _kafka_error_name(exc: KafkaException) -> str:
    args: tuple[object, ...] = exc.args
    if args and isinstance(args[0], KafkaError):
        return args[0].name()
    return type(exc).__name__
