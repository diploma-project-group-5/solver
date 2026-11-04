"""Composition root: a Kafka client that cannot be built fails with a log line, not a traceback."""

import json

import pytest
from confluent_kafka import KafkaException

from lad_solver.adapters.kafka import main as worker_main

ENV = {"KAFKA_BOOTSTRAP_SERVERS": "kafka:9092"}


class _Closable:
    def __init__(self, *_args: object) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _fail(*_args: object) -> None:
    raise KafkaException


def test_consumer_init_failure_logs_and_returns_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(worker_main, "KafkaConsumer", _fail)

    assert worker_main.run(ENV) == 1

    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    failed = [x for x in lines if x["msg"] == "kafka.init_failed"]
    assert failed == [failed[0]]
    assert (failed[0]["level"], failed[0]["error_class"]) == ("error", "KafkaException")


def test_producer_init_failure_closes_the_consumer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    consumers: list[_Closable] = []

    def consumer(*args: object) -> _Closable:
        consumers.append(_Closable(*args))
        return consumers[-1]

    monkeypatch.setattr(worker_main, "KafkaConsumer", consumer)
    monkeypatch.setattr(worker_main, "KafkaProducer", _fail)

    assert worker_main.run(ENV) == 1

    assert consumers[0].closed
    assert "kafka.init_failed" in capsys.readouterr().out
