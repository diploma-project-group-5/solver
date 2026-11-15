"""The log allow-list is the main personal-data guarantee of the worker logs (spec §13)."""

import io
import json
import logging
from collections.abc import Iterator

import pytest

from lad_solver.adapters.kafka import jsonlog

MARKER = "Іваненко-MARKER-7f3a"
BASE_KEYS = frozenset({"time", "level", "logger", "msg"})


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    buffer = io.StringIO()
    jsonlog.configure(buffer)
    try:
        yield buffer
    finally:
        jsonlog.configure()  # back to stdout, as tests/adapters/conftest.py sets it


def _entries(buffer: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buffer.getvalue().splitlines()]


def test_fields_outside_the_allow_list_are_dropped(stream: io.StringIO) -> None:
    jsonlog.log(
        logging.INFO, "job.completed", job_id="j-1", payload=MARKER, name=MARKER, signature=MARKER
    )

    assert MARKER not in stream.getvalue()
    [entry] = _entries(stream)
    assert entry["job_id"] == "j-1"
    assert set(entry) <= BASE_KEYS | jsonlog.ALLOWED_FIELDS


def _raise_marker() -> None:
    raise RuntimeError(MARKER)


def test_exception_text_and_traceback_are_never_logged(stream: io.StringIO) -> None:
    logger = logging.getLogger(jsonlog.LOGGER_NAME)
    try:
        _raise_marker()
    except RuntimeError:
        logger.exception("worker.crashed")  # ERROR with exc_info: the case under test

    assert MARKER not in stream.getvalue()
    assert "Traceback" not in stream.getvalue()
    [entry] = _entries(stream)
    assert entry["msg"] == "worker.crashed"
    assert entry["level"] == "error"


def test_engine_loggers_reach_the_json_handler_not_stderr(
    stream: io.StringIO, capsys: pytest.CaptureFixture[str]
) -> None:
    # Any lad_solver.* logger goes through the same allow-list, never Python's lastResort (stderr).
    logging.getLogger("lad_solver.app.x").warning("engine.event")

    [entry] = _entries(stream)
    assert (entry["msg"], entry["level"], entry["logger"]) == (
        "engine.event",
        "warning",
        "lad_solver.app.x",
    )
    assert capsys.readouterr().err == ""


def test_job_context_adds_ids_only_inside_the_block(stream: io.StringIO) -> None:
    with jsonlog.job_context(job_id="j-1", school_id="s-1", correlation_id="req-1"):
        jsonlog.log(logging.INFO, "inside")
    jsonlog.log(logging.INFO, "outside")

    inside, outside = _entries(stream)
    assert (inside["job_id"], inside["school_id"], inside["correlation_id"]) == (
        "j-1",
        "s-1",
        "req-1",
    )
    assert {"job_id", "school_id", "correlation_id"}.isdisjoint(outside)


def test_explicit_field_wins_over_the_job_context(stream: io.StringIO) -> None:
    with jsonlog.job_context(job_id="j-1", school_id="s-1", correlation_id=None):
        jsonlog.log(logging.INFO, "x", job_id="j-2")

    [entry] = _entries(stream)
    assert (entry["job_id"], entry["school_id"]) == ("j-2", "s-1")
    assert "correlation_id" not in entry  # None is never written


def test_job_context_reaches_engine_loggers(stream: io.StringIO) -> None:
    with jsonlog.job_context(job_id="j-1", school_id="s-1", correlation_id="req-1"):
        logging.getLogger("lad_solver.model.y").info("engine.event")

    [entry] = _entries(stream)
    assert entry["job_id"] == "j-1"
    assert set(entry) <= BASE_KEYS | jsonlog.ALLOWED_FIELDS


def test_configure_twice_does_not_duplicate_lines() -> None:
    buffer = io.StringIO()
    try:
        jsonlog.configure(buffer)
        jsonlog.configure(buffer)
        jsonlog.log(logging.INFO, "once")
        logging.getLogger("lad_solver.app.z").info("once")
    finally:
        jsonlog.configure()

    assert [e["msg"] for e in _entries(buffer)] == ["once", "once"]


def test_message_arguments_are_never_formatted(stream: io.StringIO) -> None:
    # %-arguments may carry input data: only the fixed message template is written.
    logging.getLogger("lad_solver.app.x").info("engine.event %s", MARKER)

    assert MARKER not in stream.getvalue()
    [entry] = _entries(stream)
    assert entry["msg"] == "engine.event %s"
