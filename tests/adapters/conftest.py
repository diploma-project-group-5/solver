import pytest

from lad_solver.adapters.kafka import jsonlog


@pytest.fixture(autouse=True)
def json_logs() -> None:
    """Route the worker's JSON logs to the current (captured) stdout, as in the container."""
    jsonlog.configure()
