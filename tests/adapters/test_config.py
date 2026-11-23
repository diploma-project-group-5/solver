"""Worker config: fail fast at startup on values that would only fail later, per job or per poll."""

import pytest

from lad_solver.adapters.kafka.config import ConfigError, WorkerConfig

BOOTSTRAP = {"KAFKA_BOOTSTRAP_SERVERS": "kafka:9092"}


def _load(**env: str) -> WorkerConfig:
    return WorkerConfig.from_env(BOOTSTRAP | env)


@pytest.mark.parametrize(
    ("env", "variable"),
    [
        ({"SOLVER_NUM_WORKERS": "65"}, "SOLVER_NUM_WORKERS"),
        ({"SOLVER_NUM_WORKERS": str(2**31)}, "SOLVER_NUM_WORKERS"),
        ({"SOLVER_SEED": str(2**31)}, "SOLVER_SEED"),
        ({"SOLVER_SEED": str(-(2**31) - 1)}, "SOLVER_SEED"),
    ],
)
def test_cp_sat_knobs_must_fit_their_ranges(env: dict[str, str], variable: str) -> None:
    with pytest.raises(ConfigError, match=variable) as caught:
        _load(**env)

    assert next(iter(env.values())) not in str(caught.value)


def test_cp_sat_knobs_at_their_bounds_are_accepted() -> None:
    assert _load(SOLVER_NUM_WORKERS="64", SOLVER_SEED=str(2**31 - 1)).num_workers == 64
    assert _load(SOLVER_NUM_WORKERS="1", SOLVER_SEED=str(-(2**31))).seed == -(2**31)


@pytest.mark.parametrize(
    ("env", "variable"),
    [
        ({"SOLVER_BUILD_DLQ_TOPIC": "solve.build.requested"}, "SOLVER_BUILD_DLQ_TOPIC"),
        ({"SOLVER_STATUS_TOPIC": "solve.build.requested"}, "SOLVER_STATUS_TOPIC"),
        ({"SOLVER_STATUS_TOPIC": "solve status"}, "SOLVER_STATUS_TOPIC"),
        ({"SOLVER_BUILD_TOPIC": "a" * 250}, "SOLVER_BUILD_TOPIC"),
        ({"SOLVER_BUILD_DLQ_TOPIC": "dlq/1"}, "SOLVER_BUILD_DLQ_TOPIC"),
        ({"SOLVER_SUBSTITUTION_TOPIC": "solve.build.requested"}, "SOLVER_SUBSTITUTION_TOPIC"),
        ({"SOLVER_SUBSTITUTION_TOPIC": "solve.status"}, "SOLVER_SUBSTITUTION_TOPIC"),
        ({"SOLVER_SUBSTITUTION_TOPIC": "solve status"}, "SOLVER_SUBSTITUTION_TOPIC"),
        # A DLQ equal to a consumed topic would feed every poison message back forever.
        (
            {"SOLVER_SUBSTITUTION_DLQ_TOPIC": "solve.substitution.requested"},
            "SOLVER_SUBSTITUTION_DLQ_TOPIC",
        ),
        (
            {"SOLVER_SUBSTITUTION_DLQ_TOPIC": "solve.build.requested"},
            "SOLVER_SUBSTITUTION_DLQ_TOPIC",
        ),
        ({"SOLVER_BUILD_DLQ_TOPIC": "solve.substitution.requested"}, "SOLVER_BUILD_DLQ_TOPIC"),
        ({"SOLVER_SUBSTITUTION_DLQ_TOPIC": "dlq/1"}, "SOLVER_SUBSTITUTION_DLQ_TOPIC"),
    ],
)
def test_topics_must_be_distinct_legal_names(env: dict[str, str], variable: str) -> None:
    with pytest.raises(ConfigError, match=variable) as caught:
        _load(**env)

    assert "solve status" not in str(caught.value)
    assert "dlq/1" not in str(caught.value)


@pytest.mark.parametrize(
    "env",
    [
        {"KAFKA_HEARTBEAT_INTERVAL_MS": "0"},
        {"KAFKA_SESSION_TIMEOUT_MS": "0", "KAFKA_HEARTBEAT_INTERVAL_MS": "0"},
        {"KAFKA_SESSION_TIMEOUT_MS": "-3", "KAFKA_HEARTBEAT_INTERVAL_MS": "-1"},
        # librdkafka refuses session.timeout.ms > max.poll.interval.ms at client creation
        {"KAFKA_SESSION_TIMEOUT_MS": "600000", "KAFKA_MAX_POLL_INTERVAL_MS": "300000"},
    ],
)
def test_kafka_timings_must_be_positive_and_consistent(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match="KAFKA_"):
        _load(**env)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "NaN", "Infinity"])
@pytest.mark.parametrize(
    "variable", ["SOLVER_STOP_GRACE_S", "SOLVER_WATCHDOG_MARGIN_S", "SOLVER_STOP_FLUSH_TIMEOUT_S"]
)
def test_non_finite_numbers_are_refused(variable: str, raw: str) -> None:
    with pytest.raises(ConfigError, match=variable) as caught:
        _load(**{variable: raw})

    assert "finite" in str(caught.value)


def test_result_max_bytes_stays_below_the_producer_message_limit() -> None:
    with pytest.raises(ConfigError, match="SOLVER_RESULT_MAX_BYTES"):
        _load(SOLVER_RESULT_MAX_BYTES="1000000")
    assert _load(SOLVER_RESULT_MAX_BYTES="990000").result_max_bytes == 990_000
    assert _load().result_max_bytes == 900 * 1024


def test_substitution_topics_default_to_the_manifest_names() -> None:
    loaded = _load()

    assert loaded.substitution_topic == "solve.substitution.requested"
    assert loaded.substitution_dlq_topic == "solve.substitution.requested.dlq"
    assert loaded.consumed_topics == ("solve.build.requested", "solve.substitution.requested")


def test_substitution_topics_come_from_env() -> None:
    loaded = _load(SOLVER_SUBSTITUTION_TOPIC="subst.x", SOLVER_SUBSTITUTION_DLQ_TOPIC="subst.x.dlq")

    assert loaded.consumed_topics == ("solve.build.requested", "subst.x")
    assert loaded.dlq_for("subst.x") == "subst.x.dlq"
    assert loaded.dlq_for("solve.build.requested") == "solve.build.requested.dlq"


@pytest.mark.parametrize("variable", ["SOLVER_SUBSTITUTION_TOPIC", "SOLVER_SUBSTITUTION_DLQ_TOPIC"])
def test_empty_substitution_topic_fails_fast(variable: str) -> None:
    with pytest.raises(ConfigError, match=f"{variable} must not be empty"):
        _load(**{variable: " "})


def test_dlq_for_a_topic_the_worker_does_not_consume_is_an_error() -> None:
    with pytest.raises(KeyError):
        _load().dlq_for("solve.status")
