"""Payload <-> engine mapping, config validation and server-side clamping."""

import json
from collections.abc import Callable
from typing import Any
from uuid import UUID

import pytest
from kafka_support import (
    JOB_ID,
    NOW,
    SCHOOL_ID,
    build_request,
    encode,
    feasible_result,
    inbound,
    with_data,
)

from lad_solver.adapters.kafka.config import ConfigError, WorkerConfig
from lad_solver.adapters.kafka.mapping import (
    Accepted,
    Address,
    ErrorCode,
    Poison,
    Provenance,
    Rejected,
    address_of,
    completed_data,
    completed_message,
    dlq_message,
    event_id,
    params_for,
    parse_build_request,
)
from lad_solver.domain.problem import (
    AlertLimits,
    CurriculumRow,
    Passport,
    SchoolClass,
    Teacher,
    TimetableKind,
)

ADDRESS = Address(UUID(JOB_ID), UUID(SCHOOL_ID), "req-5f2c9a")


def _accepted(document: dict[str, object]) -> Accepted:
    outcome = parse_build_request(encode(document))
    assert isinstance(outcome, Accepted)
    return outcome


def test_main_payload_maps_to_problem() -> None:
    problem = _accepted(build_request()).problem

    assert problem.kind is TimetableKind.MAIN
    assert problem.school_days == (1, 2, 3, 4, 5)
    assert problem.capacity == Passport(passport=60)
    assert problem.classes[0] == SchoolClass("c-5a", 28, priority=True)
    assert problem.teachers[1] == Teacher("t-2", 18, frozenset({3}))
    assert problem.rows[0] == CurriculumRow("c-5a", "s-math", "t-1", 5)


def test_alert_payload_maps_to_alert_limits() -> None:
    document = build_request(kind="alert", model="1.2", capacity={"seats": 40, "passport": 60})

    problem = _accepted(document).problem

    assert problem.kind is TimetableKind.ALERT
    assert problem.capacity == AlertLimits(seats=40, passport=60)


def test_school_days_are_sorted() -> None:
    assert _accepted(build_request(school_days=[5, 1, 3])).problem.school_days == (1, 3, 5)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "1.2"},  # main must be 1.1
        {"capacity": {"seats": 10, "passport": 60}},  # main takes passport only
        {"kind": "alert", "model": "1.2"},  # alert needs seats
        {"school_days": [1, 1, 2]},
        {"school_days": []},
        {"lessons_per_day": 11},
        {"classes": [{"id": "c-5a", "students": 28, "priority": True}] * 2},
        {"classes": [{"id": "has space", "students": 1, "priority": False}]},
        {"time_limit_s": 0},
        {"queue_deadline_at": "2026-11-04T10:10:00"},  # naive time is ambiguous
    ],
)
def test_invalid_data_is_rejected_as_invalid_payload(overrides: dict[str, object]) -> None:
    outcome = parse_build_request(encode(build_request(**overrides)))

    assert isinstance(outcome, Rejected)
    assert outcome.code is ErrorCode.INVALID_PAYLOAD
    assert outcome.address == ADDRESS


def test_duplicate_curriculum_row_is_rejected() -> None:
    document = with_data(build_request(), lambda d: d["rows"].append(dict(d["rows"][0])))

    assert isinstance(parse_build_request(encode(document)), Rejected)


def test_wrong_event_type_with_job_id_is_rejected() -> None:
    document = build_request()
    document["type"] = "solve.substitution.requested"

    outcome = parse_build_request(encode(document))

    assert isinstance(outcome, Rejected)
    assert outcome.error_class == "unexpected_type"


@pytest.mark.parametrize(
    ("raw", "error_class"),
    [
        (None, "not_json_object"),
        (b"{", "not_json"),
        (b"[1, 2]", "not_json_object"),
        (b"[" * 100_000, "not_json"),
        (encode({"type": "solve.build.requested"}), "invalid_envelope"),
    ],
)
def test_unparseable_messages_are_poison(raw: bytes | None, error_class: str) -> None:
    assert parse_build_request(raw) == Poison(error_class)


def test_minor_schema_versions_are_accepted() -> None:
    document = build_request()
    document["schema_version"] = "1.7"

    assert isinstance(parse_build_request(encode(document)), Accepted)


def test_unknown_fields_are_ignored() -> None:
    document = with_data(build_request(), lambda d: d.update(future_field={"x": 1}))

    assert isinstance(parse_build_request(encode(document)), Accepted)


@pytest.mark.parametrize(("requested", "expected"), [(1, 10), (10, 10), (60, 60), (3600, 180)])
def test_time_limit_is_clamped_server_side(requested: int, expected: int) -> None:
    data = _accepted(build_request(time_limit_s=requested)).data

    params = params_for(data, min_time_limit_s=10, max_time_limit_s=180, num_workers=4, seed=3)

    assert params.time_limit_s == expected
    assert (params.num_workers, params.random_seed) == (4, 3)


def test_event_id_is_deterministic_per_job_and_type() -> None:
    job = UUID(JOB_ID)

    assert event_id(job, "solve.started") == event_id(job, "solve.started")
    assert event_id(job, "solve.started") != event_id(job, "solve.completed")
    assert event_id(job, "solve.started").version == 5


def test_input_hash_ignores_job_identity_but_not_the_problem() -> None:
    first = _accepted(build_request())
    other_job = _accepted(
        build_request(
            job_id="1a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d",
            queue_deadline_at="2026-11-04T11:00:00Z",
            time_limit_s=30,
        )
    )
    other_problem = _accepted(build_request(staff=5))

    assert first.input_sha256 == other_job.input_sha256
    assert first.input_sha256 != other_problem.input_sha256


def test_result_round_trips_into_completed_data() -> None:
    data = completed_data(
        UUID(JOB_ID), feasible_result(), Provenance(wall_time_ms=1500, input_sha256="a" * 64)
    )

    assert data["conflicts"] == [{"rule": "packing", "teacher_id": "t-1", "hint": True}]
    assert data["day_formats"] == [
        {"class_id": "c-5a", "day": 1, "in_person": True},
        {"class_id": "c-9b", "day": 2, "in_person": False},
    ]
    stats = data["stats"]
    assert isinstance(stats, dict)
    assert stats["wall_time_ms"] == 1500
    assert stats["stop_reason"] == "time_limit"
    assert stats["objective"] == 140


def test_completed_message_is_keyed_by_school_and_sized() -> None:
    data = completed_data(UUID(JOB_ID), feasible_result(), None)

    completed = completed_message(ADDRESS, data, now=NOW, topic="solve.status", max_bytes=900_000)

    assert completed.message.key == SCHOOL_ID.encode()
    assert completed.error_code is None
    assert json.loads(completed.message.value or b"")["time"] == "2026-11-04T10:00:00Z"


def test_dlq_copy_keeps_contract_fields_and_drops_unknown_ones() -> None:
    document = with_data(
        build_request(), lambda d: d["teachers"][0].update(signature="Secret S.", hours="x")
    )
    document["data"]["classes"] = "not-a-list"

    copy = dlq_message(inbound(encode(document)), topic="dlq", error_class="validation")

    projected = json.loads(copy.value or b"")["projected"]
    assert projected["data"]["teachers"][0] == {
        "id": "t-1",
        "weekly_limit": 20,
        "unavailable_days": [],
    }
    assert projected["data"]["classes"] == "<redacted:str>"
    assert projected["school_id"] == SCHOOL_ID
    assert copy.key == SCHOOL_ID.encode()


# ---------------------------------------------------------------- config


def test_config_defaults_match_the_adr() -> None:
    loaded = WorkerConfig.from_env({"KAFKA_BOOTSTRAP_SERVERS": "kafka:9092"})

    assert loaded.group_id == "lad-solver"
    assert (loaded.min_time_limit_s, loaded.max_time_limit_s) == (10, 180)
    assert (loaded.num_workers, loaded.seed) == (4, 0)
    assert loaded.max_poll_interval_ms == 300_000
    assert loaded.build_dlq_topic == "solve.build.requested.dlq"
    assert loaded.completed_flush_timeout_s == 35


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"KAFKA_BOOTSTRAP_SERVERS": "k:9092", "SOLVER_NUM_WORKERS": "four"},
        {"KAFKA_BOOTSTRAP_SERVERS": "k:9092", "SOLVER_NUM_WORKERS": "0"},
        {"KAFKA_BOOTSTRAP_SERVERS": "k:9092", "SOLVER_MIN_TIME_LIMIT_S": "200"},
        # max.poll.interval.ms must exceed the largest limit + 60 s
        {"KAFKA_BOOTSTRAP_SERVERS": "k:9092", "SOLVER_MAX_TIME_LIMIT_S": "300"},
        {"KAFKA_BOOTSTRAP_SERVERS": "k:9092", "KAFKA_MAX_POLL_INTERVAL_MS": "240000"},
    ],
)
def test_invalid_config_fails_fast(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        WorkerConfig.from_env(env)


def test_config_error_names_the_variable_not_the_value() -> None:
    with pytest.raises(ConfigError) as caught:
        WorkerConfig.from_env({"KAFKA_BOOTSTRAP_SERVERS": "k:9092", "SOLVER_SEED": "s3cr3t"})

    assert "SOLVER_SEED" in str(caught.value)
    assert "s3cr3t" not in str(caught.value)


# ---------------------------------------------------------------- addressable messages (review 5)


@pytest.mark.parametrize("version", ["1", "1.0", "1.7", "1.0.0", "1.2.3"])
def test_major_1_in_any_semver_form_is_accepted(version: str) -> None:
    document = build_request()
    document["schema_version"] = version

    assert isinstance(parse_build_request(encode(document)), Accepted)


@pytest.mark.parametrize("version", ["2", "2.0", "2.0.0", "0.9"])
def test_other_majors_are_unsupported(version: str) -> None:
    document = build_request()
    document["schema_version"] = version

    outcome = parse_build_request(encode(document))

    assert isinstance(outcome, Rejected)
    assert outcome.code is ErrorCode.UNSUPPORTED_SCHEMA_VERSION
    assert outcome.address == ADDRESS


type Mutation = Callable[[dict[str, Any]], object]

ENVELOPE_ERRORS: dict[str, Mutation] = {
    "no-event-id": lambda d: d.pop("event_id"),
    "bad-event-id": lambda d: d.update(event_id="not-a-uuid"),
    "no-version": lambda d: d.pop("schema_version"),
    "bad-version": lambda d: d.update(schema_version="v1"),
    "empty-type": lambda d: d.update(type=""),
    "no-data": lambda d: d.update(data={"job_id": JOB_ID}),
}


@pytest.mark.parametrize("case", sorted(ENVELOPE_ERRORS))
def test_addressable_envelope_errors_are_rejected_not_poison(case: str) -> None:
    document = build_request()
    ENVELOPE_ERRORS[case](document)

    outcome = parse_build_request(encode(document))

    assert isinstance(outcome, Rejected)
    assert outcome.code is ErrorCode.INVALID_PAYLOAD
    assert outcome.address == ADDRESS


@pytest.mark.parametrize("correlation_id", [None, "", 42, "x" * 129])
def test_missing_or_invalid_correlation_id_falls_back_to_the_job_id(
    correlation_id: object,
) -> None:
    document = build_request()
    if correlation_id is None:
        document.pop("correlation_id")
    else:
        document["correlation_id"] = correlation_id

    outcome = parse_build_request(encode(document))

    assert isinstance(outcome, Rejected)
    assert outcome.code is ErrorCode.INVALID_PAYLOAD
    assert outcome.address == Address(UUID(JOB_ID), UUID(SCHOOL_ID), JOB_ID)


NO_ADDRESS: dict[str, Mutation] = {
    "no-school": lambda d: d.pop("school_id"),
    "bad-school": lambda d: d.update(school_id="nope"),
    "bad-job": lambda d: d["data"].update(job_id="nope"),
    "data-not-object": lambda d: d.update(data=[]),
}


@pytest.mark.parametrize("case", sorted(NO_ADDRESS))
def test_messages_without_an_address_stay_poison(case: str) -> None:
    document = build_request()
    NO_ADDRESS[case](document)

    assert parse_build_request(encode(document)) == Poison("invalid_envelope")


def test_address_of_survives_any_input() -> None:
    assert address_of(encode(build_request())) == ADDRESS
    assert address_of(b"\xff") is None
    assert address_of(None) is None
    assert address_of(encode({"school_id": SCHOOL_ID})) is None


def test_max_memory_defaults_to_512_mb_and_reaches_params() -> None:
    loaded = WorkerConfig.from_env({"KAFKA_BOOTSTRAP_SERVERS": "kafka:9092"})
    custom = WorkerConfig.from_env(
        {"KAFKA_BOOTSTRAP_SERVERS": "kafka:9092", "SOLVER_MAX_MEMORY_MB": "300"}
    )
    data = _accepted(build_request()).data

    params = params_for(
        data,
        min_time_limit_s=10,
        max_time_limit_s=180,
        num_workers=4,
        seed=0,
        max_memory_mb=custom.max_memory_mb,
    )

    assert loaded.max_memory_mb == 512
    assert params.max_memory_mb == 300


def test_shutdown_budget_must_fit_the_stop_grace_period() -> None:
    with pytest.raises(ConfigError):
        WorkerConfig.from_env({"KAFKA_BOOTSTRAP_SERVERS": "k:9092", "SOLVER_STOP_GRACE_S": "28"})
