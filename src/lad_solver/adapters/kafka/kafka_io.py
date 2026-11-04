"""confluent-kafka implementations of the worker ports (configs from ADR 2026-11-04 §4)."""

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


def _raise_if_fatal(error: KafkaError) -> None:
    # Called from librdkafka callbacks on the main thread (inside poll/flush); the exception
    # propagates out of that poll/flush call and the worker exits with code 1.
    jsonlog.log(WARNING, "kafka.error", error_class=error.name())
    if error.fatal():
        raise FatalKafkaError(error.name())


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
        "error_cb": _raise_if_fatal,
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
        "error_cb": _raise_if_fatal,
    }


class KafkaConsumer:
    def __init__(self, config: WorkerConfig, topics: list[str]) -> None:
        self._errors = 0
        self._consumer = Consumer(consumer_config(config) | {"error_cb": self._on_error})
        self._consumer.subscribe(topics)

    def _on_error(self, error: KafkaError) -> None:
        self._errors += 1
        _raise_if_fatal(error)

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
            _raise_if_fatal(error)
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
        for message in messages:
            if self._state.get(message) == _PENDING:
                continue  # still queued from an earlier attempt: flush, do not duplicate
            try:
                self._producer.produce(
                    message.topic,
                    value=message.value,
                    key=message.key,
                    headers=list(message.headers),
                    on_delivery=partial(self._report, message),
                )
            except (KafkaException, BufferError) as exc:
                raise TransientKafkaError(type(exc).__name__) from None
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
