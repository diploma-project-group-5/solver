"""JSON logs to stdout with an allow-list of fields. Payloads and exception texts are never logged.

Only identifiers, statuses, sizes and durations pass (spec §13, personal-data rules): a field that
is not in ``ALLOWED_FIELDS`` is dropped, so a careless call cannot leak a payload.
"""

import json
import logging
import sys
import traceback
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Final, TextIO, override

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
    }
)

_PACKAGE: Final = "lad_solver"
_WHERE_FRAMES: Final = 3

_EXTRA_KEY: Final = "lad_fields"


class JsonFormatter(logging.Formatter):
    @override
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.msg if isinstance(record.msg, str) else "?",
        }
        fields: object = getattr(record, _EXTRA_KEY, None)
        if isinstance(fields, dict):
            for key, value in fields.items():  # pyright: ignore[reportUnknownVariableType]
                if key in ALLOWED_FIELDS and value is not None:
                    entry[str(key)] = value  # pyright: ignore[reportUnknownArgumentType]
        # Deliberately no record.exc_info / exc_text: exception messages may quote input.
        return json.dumps(entry, ensure_ascii=False, default=str)


class _StdoutHandler(logging.StreamHandler[TextIO]):
    """Writes to whatever ``sys.stdout`` is at emit time (test capture, redirected stdout)."""

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout  # no setStream(): it would flush a possibly closed stream
        super().emit(record)


def configure(stream: TextIO | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream) if stream is not None else _StdoutHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


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
