"""Conformance with the ``contracts`` draft (JSON Schema 2020-12).

Reads the contracts checkout from ``LAD_CONTRACTS_DIR`` (default ``../contracts`` next to this
repo) and skips when it is absent. CI checks out diploma-project-group-5/contracts at a pinned
commit and sets LAD_CONTRACTS_DIR, so there these tests always run.
"""

import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from kafka_support import (
    JOB_ID,
    NOW,
    SCHOOL_ID,
    Json,
    build_request,
    decode,
    encode,
    feasible_result,
    harness,
    with_data,
)
from referencing import Registry, Resource

import oracle
from builders import OTHER, subjects
from lad_solver.adapters.kafka.mapping import (
    Address,
    ErrorCode,
    Provenance,
    completed_data,
    completed_message,
    rejected_data,
    started_message,
)
from lad_solver.domain.problem import (
    AlertLimits,
    CurriculumRow,
    Passport,
    Problem,
    SchoolClass,
    Teacher,
    TimetableKind,
)
from lad_solver.domain.result import (
    Conflict,
    ConflictRule,
    DayFormat,
    Lesson,
    Result,
    SolveStatus,
    Stats,
)
from lad_solver.model.cpsat import SolverConfig, run
from lad_solver.model.index import build_index
from lad_solver.model.phase_a import build_phase_a

_DEFAULT = Path(__file__).resolve().parents[3] / "contracts"
CONTRACTS = Path(os.environ.get("LAD_CONTRACTS_DIR", str(_DEFAULT)))
SCHEMAS = CONTRACTS / "schemas" / "events"

pytestmark = pytest.mark.skipif(
    not SCHEMAS.is_dir(),
    reason=(
        f"contracts repo not available in CI (looked in {CONTRACTS}; set LAD_CONTRACTS_DIR) "
        "— see ADR 2026-11-04"
    ),
)

ADDRESS = Address(UUID(JOB_ID), UUID(SCHOOL_ID), "req-5f2c9a")


def _load(path: Path) -> Any:  # noqa: ANN401 -- arbitrary JSON
    return json.loads(path.read_text(encoding="utf-8"))


def _validator(schema_file: str) -> Draft202012Validator:
    registry: Registry[Any] = Registry()
    for path in sorted(SCHEMAS.glob("*.json")):
        schema = _load(path)
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    schema = _load(SCHEMAS / schema_file)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())


def _errors(schema_file: str, document: Any) -> list[str]:  # noqa: ANN401 -- arbitrary JSON
    validator = _validator(schema_file)
    found = validator.iter_errors(document)  # pyright: ignore[reportUnknownMemberType]
    return [f"{list(e.absolute_path)}: {e.message}" for e in found]


def _examples() -> list[Path]:
    return sorted((CONTRACTS / "examples").glob("*/*.json")) if SCHEMAS.is_dir() else []


@pytest.mark.parametrize("example", _examples(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_example_validates_against_its_schema(example: Path) -> None:
    assert _errors(f"{example.parent.name}.json", _load(example)) == []


def test_examples_exist_for_every_event_schema() -> None:
    folders = {p.parent.name for p in _examples()}
    assert folders >= {
        "solve.build.requested.v1",
        "solve.status.v1",
        "solve.substitution.requested.v1",
    }


def test_test_fixture_request_conforms() -> None:
    assert _errors("solve.build.requested.v1.json", build_request()) == []


def test_schema_rejects_teacher_signatures_in_the_request() -> None:
    document = with_data(build_request(), lambda d: d["teachers"][0].update(signature="X"))

    assert _errors("solve.build.requested.v1.json", document) != []


def test_schema_ties_kind_to_model_and_capacity() -> None:
    bad = build_request(kind="alert", model="1.2")  # alert without seats

    assert _errors("solve.build.requested.v1.json", bad) != []


def test_started_event_conforms() -> None:
    message = started_message(ADDRESS, NOW, "solve.status")

    assert _errors("solve.status.v1.json", decode(message)) == []


@pytest.mark.parametrize(
    "data",
    [
        completed_data(
            UUID(JOB_ID), feasible_result(), Provenance(wall_time_ms=12, input_sha256="0" * 64)
        ),
        completed_data(UUID(JOB_ID), Result(status=SolveStatus.EXPIRED), None),
        rejected_data(UUID(JOB_ID), ErrorCode.INVALID_PAYLOAD),
        rejected_data(UUID(JOB_ID), ErrorCode.INTERNAL),
        completed_data(
            UUID(JOB_ID),
            Result(
                SolveStatus.MODEL_INVALID,
                conflicts=(Conflict(ConflictRule.INVALID_INPUT, class_id="c-5a"),),
                stats=Stats(num_workers=4),
            ),
            Provenance(wall_time_ms=3, input_sha256="0" * 64),
        )
        | {"error": {"code": "invalid_payload"}},
        completed_data(
            UUID(JOB_ID),
            Result(SolveStatus.MODEL_INVALID, stats=Stats(num_workers=4)),
            Provenance(wall_time_ms=3, input_sha256="0" * 64),
        )
        | {"error": {"code": "internal"}},
    ],
    ids=[
        "feasible",
        "expired",
        "invalid_payload",
        "internal",
        "precheck-invalid-input",
        "engine-model-invalid",
    ],
)
def test_completed_event_conforms(data: dict[str, object]) -> None:
    completed = completed_message(ADDRESS, data, now=NOW, topic="solve.status", max_bytes=900_000)

    assert _errors("solve.status.v1.json", decode(completed.message)) == []


def test_result_too_large_event_conforms() -> None:
    data = completed_data(UUID(JOB_ID), feasible_result(), None)
    completed = completed_message(ADDRESS, data, now=NOW, topic="solve.status", max_bytes=600)

    assert completed.error_code is ErrorCode.RESULT_TOO_LARGE
    assert _errors("solve.status.v1.json", decode(completed.message)) == []


def test_every_event_the_worker_publishes_conforms() -> None:
    h = harness()
    invalid = with_data(build_request(), lambda d: d.update(staff=-1))

    h.run(encode(build_request()), encode(invalid))

    events = [decode(m) for m in h.producer.sent if m.topic == "solve.status"]
    assert [e["type"] for e in events] == ["solve.started", "solve.completed", "solve.completed"]
    for event in events:
        assert _errors("solve.status.v1.json", event) == []


def test_error_is_only_allowed_with_model_invalid() -> None:
    event: Json = decode(
        completed_message(
            ADDRESS,
            rejected_data(UUID(JOB_ID), ErrorCode.INTERNAL) | {"status": "FEASIBLE"},
            now=NOW,
            topic="solve.status",
            max_bytes=900_000,
        ).message
    )

    assert _errors("solve.status.v1.json", event) != []


# ---------------------------------------------------------------- capacity vectors


def _vectors() -> list[Path]:
    return sorted((CONTRACTS / "vectors" / "capacity").glob("*.json")) if SCHEMAS.is_dir() else []


@pytest.mark.parametrize("vector", _vectors(), ids=lambda p: p.stem)
def test_capacity_vector_is_internally_consistent(vector: Path) -> None:
    """Guards the shared data itself; the formula is checked by the engine tests and
    validator-lib (cross-repo rule: one formula, shared vectors)."""
    case = _load(vector)
    conditions = case["expected"]["conditions"]
    assert set(conditions) == ({"passport"} if case["model"] == "1.1" else {"passport", "seats"})
    for condition in conditions.values():
        assert condition["ok"] == (condition["load"] <= condition["limit"])
    assert case["expected"]["ok"] == all(c["ok"] for c in conditions.values())


def _vector_day(case: Json) -> tuple[Problem, Result]:
    """The vector's day as a one-day Problem plus a Result with its formats and lessons."""
    day = case["day"]
    pairs = [(x["class_id"], x["teacher_id"]) for x in day["lessons"]]
    counts = {pair: pairs.count(pair) for pair in dict.fromkeys(pairs)}
    teacher_ids = sorted({t for _, t in pairs})
    kind = TimetableKind.MAIN if case["model"] == "1.1" else TimetableKind.ALERT
    capacity = case["capacity"]
    problem = Problem(
        kind=kind,
        school_days=(1,),
        lessons_per_day=max(1, *counts.values(), len(pairs)),
        staff=case["staff"],
        capacity=(
            Passport(capacity["passport"])
            if kind is TimetableKind.MAIN
            else AlertLimits(capacity["seats"], capacity["passport"])
        ),
        classes=tuple(SchoolClass(c["id"], c["students"], priority=False) for c in day["classes"]),
        subjects=subjects(),
        teachers=tuple(Teacher(t, 60) for t in teacher_ids),
        rows=tuple(CurriculumRow(c, OTHER, t, n) for (c, t), n in counts.items()),
    )
    lessons = tuple(
        Lesson(c, 1, i, OTHER, t) for i, (c, t) in enumerate(pairs, start=1)
    )  # distinct lesson numbers: the capacity rules ignore them
    formats = tuple(DayFormat(c["id"], 1, in_person=c["in_person"]) for c in day["classes"])
    return problem, Result(SolveStatus.FEASIBLE, lessons, formats)


@pytest.mark.parametrize("vector", _vectors(), ids=lambda p: p.stem)
def test_capacity_vector_loads_match_the_oracle(vector: Path) -> None:
    case = _load(vector)
    problem, result = _vector_day(case)

    load = oracle.day_load(problem, result, 1)

    expected = case["expected"]
    assert (load.pupils, load.teachers) == (
        expected["students_in_person"],
        expected["teachers_on_site"],
    )
    assert expected["conditions"]["passport"]["load"] == load.people
    if "seats" in expected["conditions"]:
        assert expected["conditions"]["seats"]["load"] == load.pupils
    assert (oracle.violations(problem, result) == []) == expected["ok"]


@pytest.mark.parametrize("vector", _vectors(), ids=lambda p: p.stem)
def test_phase_a_capacity_matches_the_vector(vector: Path) -> None:
    """Formats fixed by the vector: phase A (incl. the z linearisation) is feasible iff ok."""
    case = _load(vector)
    problem, _ = _vector_day(case)
    index = build_index(problem)
    built = build_phase_a(index, index.row_caps(2, relaxed=True))
    in_person = {c["id"]: c["in_person"] for c in case["day"]["classes"]}
    for (c, d), f in built.f.items():
        built.model.add(f == int(in_person[problem.classes[c].id]))
        assert d == 1

    _, info = run(built.model, SolverConfig(time_limit_s=10, num_workers=1))

    feasible = info.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert feasible == case["expected"]["ok"], info.status


def test_vectors_cover_both_models_and_both_outcomes() -> None:
    cases = [_load(v) for v in _vectors()]
    assert {(c["model"], c["expected"]["ok"]) for c in cases} == {
        ("1.1", True),
        ("1.1", False),
        ("1.2", True),
        ("1.2", False),
    }
    seats_only = [
        c
        for c in cases
        if c["model"] == "1.2"
        and not c["expected"]["conditions"]["seats"]["ok"]
        and c["expected"]["conditions"]["passport"]["ok"]
    ]
    passport_only = [
        c
        for c in cases
        if c["model"] == "1.2"
        and c["expected"]["conditions"]["seats"]["ok"]
        and not c["expected"]["conditions"]["passport"]["ok"]
    ]
    assert seats_only
    assert passport_only


# ---------------------------------------------------------------- example objectives (review 16)


def test_completed_examples_state_objective_as_w_primary_plus_secondary() -> None:
    """W = |priority classes| x |school days| + 1, from the request with the same job_id."""
    requests = {
        doc["data"]["job_id"]: doc["data"]
        for doc in map(_load, sorted((CONTRACTS / "examples").glob("solve.build.requested.v1/*")))
    }
    checked = 0
    for path in sorted((CONTRACTS / "examples" / "solve.status.v1").glob("completed-*.json")):
        data = _load(path)["data"]
        stats: Json = data.get("stats") or {}
        if stats.get("objective") is None:
            continue
        request = requests[data["job_id"]]  # an objective needs its request to be checkable
        weight = sum(c["priority"] for c in request["classes"]) * len(request["school_days"]) + 1
        assert stats["objective"] == weight * stats["primary"] + stats["secondary"], path.name
        assert stats["best_bound"] is None or stats["best_bound"] >= stats["objective"]
        checked += 1
    assert checked >= 1
