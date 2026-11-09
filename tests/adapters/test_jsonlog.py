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
