"""JSON logs to stdout with an allow-list of fields. Payloads and exception texts are never logged.

Only identifiers, statuses, sizes and durations pass (spec §13, personal-data rules): a field that
is not in ``ALLOWED_FIELDS`` is dropped, so a careless call cannot leak a payload.

The handler sits on the package logger ``lad_solver``: every ``lad_solver.*`` logger goes through
the same allow-list (never Python's lastResort on stderr). Inside ``job_context`` each line gets
the job's ids, also from the solve thread when it was started with ``contextvars.copy_context()``.
"""

import json
import logging
import sys
import traceback
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Final, TextIO, override

ROOT_LOGGER: Final = "lad_solver"
LOGGER_NAME: Final = "lad_solver.worker"

ALLOWED_FIELDS: Final = frozenset(
    {
        "job_id",
        "school_id",
        "correlation_id",
        "status",
        "error_code",
        "error_class",
        # librdkafka error names (e.g. "_TRANSPORT") or fixed literals only, never payload text
        "reason",
        "queue_wait_ms",
        "wall_time_ms",
        "time_limit_s",
        "objective",
        "best_bound",
        "stop_reason",
        "quality_relaxed",
        "solver_version",
        "model_version",
        "topic",
        "topics",  # consumed topic names from the config (worker.ready)
        "partition",
        "offset",
        "size_bytes",
        "attempt",
        "signal",
        "where",
        "count",
        "phase_a_status",
        # timings and search telemetry (ADR 2026-11-08): numbers only
        "handle_ms",
        "first_solution_ms",
        "last_improvement_ms",
        "gap",
        "num_workers",
        "stall_min_s",
        "stall_ratio",
        "max_memory_mb",
        "group_id",
        # coverage step and phase B summary (ADR 2026-11-14): counts and enum values, no ids
        "coverage_verdict",
        "coverage_status",
        "zero_day_before",
        "zero_day_after",
        "phase_b_retry_days",
    }
)

_PACKAGE: Final = "lad_solver"
_WHERE_FRAMES: Final = 3

_EXTRA_KEY: Final = "lad_fields"

_JOB: ContextVar[Mapping[str, object] | None] = ContextVar("lad_solver_job", default=None)


class JsonFormatter(logging.Formatter):
    @override
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.msg if isinstance(record.msg, str) else "?",
        }
        # The job context first, then the call's own fields (a non-None value overrides the
        # context). Both pass the same allow-list; None is never written.
        for source in (_JOB.get(), getattr(record, _EXTRA_KEY, None)):
            if not isinstance(source, Mapping):
                continue
            for key, value in source.items():  # pyright: ignore[reportUnknownVariableType]
                if key in ALLOWED_FIELDS and value is not None:
                    entry[str(key)] = value  # pyright: ignore[reportUnknownArgumentType]
        # Deliberately no record.args (``getMessage()``) and no record.exc_info / exc_text:
        # formatting arguments and exception messages may quote input.
        return json.dumps(entry, ensure_ascii=False, default=str)


class _StdoutHandler(logging.StreamHandler[TextIO]):
    """Writes to whatever ``sys.stdout`` is at emit time (test capture, redirected stdout)."""

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout  # no setStream(): it would flush a possibly closed stream
        super().emit(record)


def configure(stream: TextIO | None = None, level: int = logging.INFO) -> logging.Logger:
    """One JSON handler on ``lad_solver``; idempotent (a second call replaces the first)."""
    root = logging.getLogger(ROOT_LOGGER)
    root.handlers.clear()
    handler = logging.StreamHandler(stream) if stream is not None else _StdoutHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    worker = logging.getLogger(LOGGER_NAME)
    worker.handlers.clear()  # a handler left here by an older configure() would duplicate lines
    worker.setLevel(logging.NOTSET)
    worker.propagate = True
    return worker


@contextmanager
def job_context(*, job_id: str, school_id: str, correlation_id: str | None) -> Generator[None]:
    """Every line logged inside the block (this thread, or a copied context) carries the ids."""
    token = _JOB.set({"job_id": job_id, "school_id": school_id, "correlation_id": correlation_id})
    try:
        yield
    finally:
        _JOB.reset(token)


def log(level: int, event: str, **fields: object) -> None:
    """``event`` is a fixed message such as ``"job.completed"``; values go in allowed fields."""
    logging.getLogger(LOGGER_NAME).log(level, event, extra={_EXTRA_KEY: fields})


def where(exc: BaseException) -> str | None:
    """Last frames of ``exc`` inside ``lad_solver`` as ``module:function:lineno`` (outer -> inner).

    Diagnosable without personal data: no source lines, no locals, no exception text.
    """
    frames: list[str] = []
    for frame in traceback.extract_tb(exc.__traceback__):
        parts = PurePath(frame.filename).with_suffix("").parts
        if _PACKAGE not in parts:
            continue
        module = ".".join(parts[len(parts) - 1 - parts[::-1].index(_PACKAGE) :])
        frames.append(f"{module}:{frame.name}:{frame.lineno}")
    return " > ".join(frames[-_WHERE_FRAMES:]) or None
