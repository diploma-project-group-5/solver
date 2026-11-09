"""Worker settings: environment only (12-factor), validated once at startup, fail fast."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# Topics are created by platform/kafka-init (auto.create.topics.enable=false).
DEFAULT_BUILD_TOPIC = "solve.build.requested"
DEFAULT_STATUS_TOPIC = "solve.status"


# Must stay below the producer's message.max.bytes (librdkafka default 1,000,000): the key,
# headers and record overhead need room on top of the serialized result.
_RESULT_MAX_BYTES_RANGE = (1024, 990_000)
_MIN_MEMORY_MB = 64
_MAX_NUM_WORKERS = 64
_INT32 = (-(2**31), 2**31 - 1)  # CP-SAT random_seed is an int32 parameter
_TOPIC_NAME = re.compile(r"[A-Za-z0-9._-]{1,249}")  # Kafka's legal topic names


class ConfigError(ValueError):
    """Invalid worker configuration; the message names variables, never their values."""

    @classmethod
    def not_a(cls, name: str, what: str) -> ConfigError:
        return cls(f"{name} must be {what}")


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    bootstrap_servers: str
    group_id: str = "lad-solver"
    build_topic: str = DEFAULT_BUILD_TOPIC
    status_topic: str = DEFAULT_STATUS_TOPIC
    build_dlq_topic: str = f"{DEFAULT_BUILD_TOPIC}.dlq"
    min_time_limit_s: int = 10
    max_time_limit_s: int = 180
    num_workers: int = 2  # ADR 2026-11-04 §3 as amended by ADR 2026-11-08 (cpus 1.5)
    seed: int = 0
    heartbeat_file: Path = Path("/tmp/lad-solver.alive")  # noqa: S108 -- container-local file
    stop_grace_s: float = 20.0  # wait for the engine to return its best solution after SIGTERM
    # Cap of the completed flush and of the close flush once stopping:
    # stop_grace_s + 2 * stop_flush_timeout_s + 1 s (signal noticed) <= shutdown_budget_s.
    stop_flush_timeout_s: float = 4.0
    shutdown_budget_s: float = 29.0  # below compose stop_grace_period (30 s)
    watchdog_margin_s: float = 60.0  # os._exit(1) if a solve runs longer than limit + margin
    max_poll_interval_ms: int = 300_000
    session_timeout_ms: int = 10_000
    heartbeat_interval_ms: int = 3_000
    started_flush_timeout_s: float = 10.0
    delivery_timeout_ms: int = 30_000
    result_max_bytes: int = 900 * 1024  # below the 1 MB broker message limit, with envelope room
    # CP-SAT max_memory_in_mb (~0.7 x compose mem_limit 768m; measured peak RSS 222 MB for a
    # 45-class school with 4 workers).
    max_memory_mb: int = 512
    # No-improvement stop of phase A (ADR 2026-11-08): 0 = off. Stays off until timetable
    # validates against the contracts version that knows stop_reason "no_improvement".
    stall_min_s: float = 0.0
    stall_ratio: float = 1.0

    @property
    def completed_flush_timeout_s(self) -> float:
        return self.delivery_timeout_ms / 1000 + 5

    def validate(self) -> None:
        problems = [
            *self._naming_problems(),
            *self._limit_problems(),
            *self._kafka_timing_problems(),
            *self._stall_problems(),
        ]
        if problems:
            raise ConfigError("; ".join(problems))

    def _naming_problems(self) -> list[str]:
        problems: list[str] = []
        if not self.bootstrap_servers.strip():
            problems.append("KAFKA_BOOTSTRAP_SERVERS is required")
        names = {
            "SOLVER_GROUP_ID": self.group_id,
            "SOLVER_BUILD_TOPIC": self.build_topic,
            "SOLVER_STATUS_TOPIC": self.status_topic,
            "SOLVER_BUILD_DLQ_TOPIC": self.build_dlq_topic,
        }
        problems.extend(f"{name} must not be empty" for name, v in names.items() if not v.strip())
        topics = {name: v for name, v in names.items() if name != "SOLVER_GROUP_ID"}
        problems.extend(
            f"{name} must be a legal Kafka topic name ([A-Za-z0-9._-], at most 249 characters)"
            for name, v in topics.items()
            if v.strip() and _TOPIC_NAME.fullmatch(v) is None
        )
        seen: dict[str, str] = {}
        for name, value in topics.items():
            if value in seen:
                # A DLQ equal to the build topic would feed every poison message back forever.
                problems.append(f"{name} must differ from {seen[value]}")
            else:
                seen[value] = name
        return problems

    def _stall_problems(self) -> list[str]:
        problems: list[str] = []
        if not (math.isfinite(self.stall_min_s) and self.stall_min_s >= 0):
            problems.append("SOLVER_STALL_MIN_S must be a number >= 0 (0 = off)")
        if not (math.isfinite(self.stall_ratio) and self.stall_ratio > 0):
            problems.append("SOLVER_STALL_RATIO must be a number > 0")
        return problems

    def _limit_problems(self) -> list[str]:
        problems: list[str] = []
        if not 1 <= self.min_time_limit_s <= self.max_time_limit_s:
            problems.append("need 1 <= SOLVER_MIN_TIME_LIMIT_S <= SOLVER_MAX_TIME_LIMIT_S")
        if not 1 <= self.num_workers <= _MAX_NUM_WORKERS:
            problems.append(f"SOLVER_NUM_WORKERS must be within [1, {_MAX_NUM_WORKERS}]")
        if not _INT32[0] <= self.seed <= _INT32[1]:
            problems.append("SOLVER_SEED must fit in int32")
        if self.stop_grace_s <= 0 or self.watchdog_margin_s <= 0 or self.stop_flush_timeout_s <= 0:
            problems.append(
                "SOLVER_STOP_GRACE_S, SOLVER_WATCHDOG_MARGIN_S and SOLVER_STOP_FLUSH_TIMEOUT_S "
                "must be > 0"
            )
        if self.stop_grace_s + 2 * self.stop_flush_timeout_s + 1 > self.shutdown_budget_s:
            problems.append(
                "need SOLVER_STOP_GRACE_S + 2 * SOLVER_STOP_FLUSH_TIMEOUT_S + 1 "
                "<= SOLVER_SHUTDOWN_BUDGET_S (below compose stop_grace_period)"
            )
        if self.max_memory_mb < _MIN_MEMORY_MB:
            problems.append(f"SOLVER_MAX_MEMORY_MB must be >= {_MIN_MEMORY_MB}")
        low, high = _RESULT_MAX_BYTES_RANGE
        if not low <= self.result_max_bytes <= high:
            problems.append(f"SOLVER_RESULT_MAX_BYTES must be within [{low}, {high}]")
        return problems

    def _kafka_timing_problems(self) -> list[str]:
        problems: list[str] = []
        # The consumer does not poll while solving: a longer solve than this gets the worker
        # kicked out of the group and its partitions reassigned mid-solve.
        floor_ms = (self.max_time_limit_s + self.watchdog_margin_s) * 1000
        if self.max_poll_interval_ms <= floor_ms:
            problems.append(
                "KAFKA_MAX_POLL_INTERVAL_MS must exceed "
                "(SOLVER_MAX_TIME_LIMIT_S + SOLVER_WATCHDOG_MARGIN_S) * 1000",
            )
        if self.heartbeat_interval_ms < 1 or self.session_timeout_ms < 1:
            problems.append("KAFKA_HEARTBEAT_INTERVAL_MS and KAFKA_SESSION_TIMEOUT_MS must be >= 1")
        if self.heartbeat_interval_ms * 3 > self.session_timeout_ms:
            problems.append("KAFKA_HEARTBEAT_INTERVAL_MS must be <= KAFKA_SESSION_TIMEOUT_MS / 3")
        # librdkafka refuses to create the consumer otherwise (and says so only in its own log).
        if self.session_timeout_ms > self.max_poll_interval_ms:
            problems.append("KAFKA_SESSION_TIMEOUT_MS must be <= KAFKA_MAX_POLL_INTERVAL_MS")
        return problems

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> WorkerConfig:
        def text(name: str, default: str) -> str:
            return env.get(name, default)

        def integer(name: str, default: int) -> int:
            raw = env.get(name)
            if raw is None or raw == "":
                return default
            try:
                return int(raw)
            except ValueError:
                raise ConfigError.not_a(name, "an integer") from None

        def number(name: str, default: float) -> float:
            raw = env.get(name)
            if raw is None or raw == "":
                return default
            try:
                value = float(raw)
            except ValueError:
                raise ConfigError.not_a(name, "a number") from None
            if not math.isfinite(value):  # NaN compares False: it would disable every bound
                raise ConfigError.not_a(name, "a finite number")
            return value

        defaults = cls(bootstrap_servers="")
        config = cls(
            bootstrap_servers=text("KAFKA_BOOTSTRAP_SERVERS", ""),
            group_id=text("SOLVER_GROUP_ID", defaults.group_id),
            build_topic=text("SOLVER_BUILD_TOPIC", defaults.build_topic),
            status_topic=text("SOLVER_STATUS_TOPIC", defaults.status_topic),
            build_dlq_topic=text("SOLVER_BUILD_DLQ_TOPIC", defaults.build_dlq_topic),
            min_time_limit_s=integer("SOLVER_MIN_TIME_LIMIT_S", defaults.min_time_limit_s),
            max_time_limit_s=integer("SOLVER_MAX_TIME_LIMIT_S", defaults.max_time_limit_s),
            num_workers=integer("SOLVER_NUM_WORKERS", defaults.num_workers),
            seed=integer("SOLVER_SEED", defaults.seed),
            heartbeat_file=Path(text("SOLVER_HEARTBEAT_FILE", str(defaults.heartbeat_file))),
            stop_grace_s=number("SOLVER_STOP_GRACE_S", defaults.stop_grace_s),
            watchdog_margin_s=number("SOLVER_WATCHDOG_MARGIN_S", defaults.watchdog_margin_s),
            stop_flush_timeout_s=number(
                "SOLVER_STOP_FLUSH_TIMEOUT_S", defaults.stop_flush_timeout_s
            ),
            shutdown_budget_s=number("SOLVER_SHUTDOWN_BUDGET_S", defaults.shutdown_budget_s),
            max_poll_interval_ms=integer(
                "KAFKA_MAX_POLL_INTERVAL_MS", defaults.max_poll_interval_ms
            ),
            session_timeout_ms=integer("KAFKA_SESSION_TIMEOUT_MS", defaults.session_timeout_ms),
            heartbeat_interval_ms=integer(
                "KAFKA_HEARTBEAT_INTERVAL_MS", defaults.heartbeat_interval_ms
            ),
            result_max_bytes=integer("SOLVER_RESULT_MAX_BYTES", defaults.result_max_bytes),
            max_memory_mb=integer("SOLVER_MAX_MEMORY_MB", defaults.max_memory_mb),
            stall_min_s=number("SOLVER_STALL_MIN_S", defaults.stall_min_s),
            stall_ratio=number("SOLVER_STALL_RATIO", defaults.stall_ratio),
        )
        config.validate()
        return config
