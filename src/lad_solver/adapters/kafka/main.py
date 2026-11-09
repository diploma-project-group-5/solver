"""Composition root of the Kafka worker: env config + confluent-kafka + the CP-SAT engine."""

import os
import signal
from collections.abc import Mapping
from logging import ERROR, INFO
from pathlib import Path
from types import FrameType

from lad_solver import __version__
from lad_solver.adapters.kafka import jsonlog
from lad_solver.adapters.kafka.config import ConfigError, WorkerConfig
from lad_solver.adapters.kafka.kafka_io import KafkaConsumer, KafkaProducer
from lad_solver.adapters.kafka.mapping import MODEL_VERSION
from lad_solver.adapters.kafka.worker import EXIT_FAILURE, Worker
from lad_solver.app.solve import solve
from lad_solver.domain.problem import Params, Problem
from lad_solver.domain.result import Result
from lad_solver.model.progress import Progress
from lad_solver.model.stop import StopHandle


def _engine(problem: Problem, params: Params, stop: StopHandle, progress: Progress) -> Result:
    return solve(problem, params, stop, progress=progress)


def touch(path: Path) -> None:
    """Heartbeat for the image HEALTHCHECK (mtime must be younger than 30 s)."""
    path.touch(exist_ok=True)


def install_signal_handlers(worker: Worker) -> None:
    def handler(signum: int, _frame: FrameType | None) -> None:
        worker.request_stop(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def run(env: Mapping[str, str] | None = None) -> int:
    jsonlog.configure()
    try:
        config = WorkerConfig.from_env(os.environ if env is None else env)
    except ConfigError as exc:
        # The message names variables only, never values.
        jsonlog.log(ERROR, f"config.invalid: {exc}")
        return EXIT_FAILURE

    consumer: KafkaConsumer | None = None
    try:
        consumer = KafkaConsumer(config, [config.build_topic])  # substitutions: stage 2
        producer = KafkaProducer(config)
    except Exception as exc:  # noqa: BLE001 -- bad client config / librdkafka init; class only
        jsonlog.log(ERROR, "kafka.init_failed", error_class=type(exc).__name__)
        if consumer is not None:
            consumer.close()
        return EXIT_FAILURE
    worker = Worker(
        consumer=consumer,
        producer=producer,
        engine=_engine,
        config=config,
        heartbeat=lambda: touch(config.heartbeat_file),
    )
    install_signal_handlers(worker)
    jsonlog.log(
        INFO,
        "worker.ready",
        topic=config.build_topic,
        group_id=config.group_id,
        solver_version=__version__,
        model_version=MODEL_VERSION,
        num_workers=config.num_workers,
        time_limit_s=[config.min_time_limit_s, config.max_time_limit_s],
        max_memory_mb=config.max_memory_mb,
        stall_min_s=config.stall_min_s or None,  # None: the no-improvement stop is off
        stall_ratio=config.stall_ratio,
    )
    code = worker.run()
    jsonlog.log(INFO, "worker.exited", status=code)
    return code


if __name__ == "__main__":
    raise SystemExit(run())
