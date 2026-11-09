"""Composition root: a Kafka client that cannot be built fails with a log line, not a traceback."""

import json
import signal

import pytest
from confluent_kafka import KafkaException
from kafka_support import build_request, encode, harness

from lad_solver.adapters.kafka import main as worker_main
from lad_solver.adapters.kafka.mapping import Accepted, parse_build_request
from lad_solver.domain.problem import Params, Problem
from lad_solver.domain.result import Result, SolveStatus
from lad_solver.model.progress import Progress
from lad_solver.model.stop import StopHandle

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


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_handlers_request_a_stop(
    signum: signal.Signals, capsys: pytest.CaptureFixture[str]
) -> None:
    h = harness()
    h.consumer.on_idle = None  # only the signal may stop the worker
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        worker_main.install_signal_handlers(h.worker)
        signal.raise_signal(signum)
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)

    assert h.worker.stop_requested
    assert h.worker.run() == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    stopping = [x for x in lines if x["msg"] == "worker.stopping"]
    assert [x["signal"] for x in stopping] == [signum.name]


def test_production_engine_passes_stop_and_progress_to_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Problem, Params, StopHandle | None, Progress | None]] = []

    def fake_solve(
        problem: Problem,
        params: Params,
        stop: StopHandle | None = None,
        *,
        progress: Progress | None = None,
    ) -> Result:
        calls.append((problem, params, stop, progress))
        return Result(status=SolveStatus.UNKNOWN)

    monkeypatch.setattr(worker_main, "solve", fake_solve)
    outcome = parse_build_request(encode(build_request()))
    assert isinstance(outcome, Accepted)
    params, stop, progress = Params(time_limit_s=10.0), StopHandle(), Progress()

    result = worker_main._engine(outcome.problem, params, stop, progress)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    assert result.status is SolveStatus.UNKNOWN
    assert calls == [(outcome.problem, params, stop, progress)]
