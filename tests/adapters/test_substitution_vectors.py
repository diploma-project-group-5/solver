"""contracts ``vectors/substitution-solve`` (0.5.0): the engine meets every expectation.

Each vector's request goes through the adapter (schema, ids, the baseline week) into the engine and
its answer through the solve.status schema. Besides ``expect``, every answer keeps the invariants of
the vectors README: each lesson exactly once, on the request's date; no ``cancelled``; ``step`` =
max(proposals[].step); a step 2 move stays on the date, a step 3 move lands inside the window; no
rows outside ``lessons``; the same request gives a byte-identical answer; and the effective week has
no new conflicts (``lad-validate --mode new-conflicts``, ``tests/substitution_golden.py``).

The week builder of the golden check is itself checked first against every ``given -> input``
vector of ``vectors/substitution`` and every pair of ``vectors/substitution-approve``.
"""

import copy
import datetime as dt
import json
from functools import cache
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from contracts_support import SKIP_WITHOUT_CONTRACTS, load, schema_errors, vectors
from kafka_support import NOW, SCHOOL_ID, decode, encode

import substitution_golden
from lad_solver.adapters.kafka.mapping import (
    Address,
    Provenance,
    SubstitutionAccepted,
    completed_message,
    parse_substitution_request,
    substitution_data,
    substitution_params,
)
from lad_solver.app.substitute import substitute
from lad_solver.domain.result import StopReason
from lad_solver.domain.substitution import SubstitutionResult

pytestmark = SKIP_WITHOUT_CONTRACTS

type Json = dict[str, Any]

SOLVE_VECTORS = vectors("substitution-solve")
WEEK_VECTORS = [p for p in vectors("substitution") if "given" in load(p)]
APPROVE_VECTORS = vectors("substitution-approve")

# The interpretations the engine implements (vectors README "params"; Roman, 15.11.2026).
OUR_PARAMS = {
    "works_that_day": "has_lesson_that_day",
    "off_profile": "no_row_with_subject",
    "equal_distance": "later_date",
    "step2_scope": "per_lesson",
    "self_study_step": "last_step_that_ran",
}


def _envelope(request: Json) -> Json:
    return {
        "event_id": "6f1d2c3b-4a5e-4f60-8b71-0c2d3e4f5a6d",
        "type": "solve.substitution.requested",
        "schema_version": "1.0",
        "source": "timetable-service",
        "time": "2026-11-16T08:00:00+02:00",
        "correlation_id": "req-vector",
        "school_id": SCHOOL_ID,
        "data": request,
    }


def answer(request: Json) -> Json:
    """``data`` of solve.completed for a request, through the production mapping."""
    outcome = parse_substitution_request(encode(_envelope(request)))
    assert isinstance(outcome, SubstitutionAccepted), outcome
    params = substitution_params(outcome.data, min_time_limit_s=10, max_time_limit_s=180)
    result = substitute(outcome.request, params)
    provenance = Provenance(wall_time_ms=0, input_sha256=outcome.input_sha256)  # no clock here
    return substitution_data(UUID(request["job_id"]), result, provenance)


@cache
def _vector(name: str) -> Json:
    return cast("Json", load(next(p for p in SOLVE_VECTORS if p.stem == name)))


@cache
def _answer(name: str) -> str:
    return json.dumps(answer(_vector(name)["given"]["request"]), sort_keys=True)


def _solved(name: str) -> Json:
    return cast("Json", json.loads(_answer(name)))


_NAMES = [p.stem for p in SOLVE_VECTORS]


def test_all_engine_vectors_are_here() -> None:
    assert len(_NAMES) >= 23
    assert {_vector(n)["name"] for n in _NAMES} == set(_NAMES)
    # the golden builder's own checks must not pass on an empty or renamed folder
    assert (len(WEEK_VECTORS), len(APPROVE_VECTORS)) >= (30, 7)


@cache
def _engine(name: str) -> SubstitutionResult:
    request = _vector(name)["given"]["request"]
    outcome = parse_substitution_request(encode(_envelope(request)))
    assert isinstance(outcome, SubstitutionAccepted), outcome
    params = substitution_params(outcome.data, min_time_limit_s=10, max_time_limit_s=180)
    return substitute(outcome.request, params)


@pytest.mark.parametrize("name", _NAMES)
def test_engine_runs_each_vector_to_the_end_without_an_anomaly(name: str) -> None:
    """A step model that fails (an engine bug) would only show as more self-study."""
    stats = _engine(name).stats

    assert (stats.stop_reason, stats.anomalies) == (StopReason.COMPLETED, ())
    assert (stats.absences_completed, stats.overlays_ignored) == (0, 0)  # consistent requests


@pytest.mark.parametrize("name", _NAMES)
def test_vectors_use_the_interpretations_the_engine_implements(name: str) -> None:
    assert _vector(name)["params"] == OUR_PARAMS


@pytest.mark.parametrize("name", _NAMES)
def test_vector_request_is_a_valid_event(name: str) -> None:
    event = _envelope(_vector(name)["given"]["request"])

    assert schema_errors("solve.substitution.requested.v1.json", event) == []


@pytest.mark.parametrize("name", _NAMES)
def test_engine_meets_the_vector_expectation(name: str) -> None:
    vector = _vector(name)
    request, expect = vector["given"]["request"], vector["expect"]

    data = _solved(name)

    _check_expect(request, expect, data)


@pytest.mark.parametrize("name", _NAMES)
def test_engine_answer_keeps_the_invariants(name: str) -> None:
    request = _vector(name)["given"]["request"]

    data = _solved(name)

    _check_invariants(request, data)


@pytest.mark.parametrize("name", _NAMES)
def test_engine_answer_conforms_to_the_status_schema(name: str) -> None:
    address = Address(UUID(_vector(name)["given"]["request"]["job_id"]), UUID(SCHOOL_ID), "c-1")
    data = _solved(name)

    completed = completed_message(address, data, now=NOW, topic="solve.status", max_bytes=900_000)

    event = decode(completed.message)
    assert event["data"]["status"] in {"OPTIMAL", "FEASIBLE"}
    assert schema_errors("solve.status.v1.json", event) == []


@pytest.mark.parametrize("name", _NAMES)
def test_engine_answer_is_byte_identical_for_the_same_request(name: str) -> None:
    request = _vector(name)["given"]["request"]

    again = json.dumps(answer(copy.deepcopy(request)), sort_keys=True)

    assert again == _answer(name)


@cache
def _golden() -> dict[str, Json]:
    cases = {
        name: substitution_golden.weeks(
            _vector(name)["given"]["request"], _solved(name)["substitution"]["proposals"]
        )
        for name in _NAMES
    }
    return substitution_golden.new_conflicts(cases)


@pytest.mark.parametrize("name", _NAMES)
def test_engine_answer_adds_no_conflict_to_the_week(name: str) -> None:
    """Invariant 5: lad-validate --mode new-conflicts gives added == [] (validator-lib 0.4.0)."""
    assert substitution_golden.added(_golden()[name]) == []


def test_support_moves_requested_are_answered_as_without_them() -> None:
    """Conditional deviation (contracts README item 9): allow_support_moves=true is not
    implemented before the defence, so the engine answers as with false (vector 12's
    ``after_defence`` alternative is not normative yet)."""
    vector = _vector("12-no-swap-chains-without-support-moves")
    [alternative] = vector["alternatives"]
    assert alternative["stage"] == "after_defence"
    request = vector["given"]["request"] | alternative["given_patch"]
    assert request["allow_support_moves"] is True

    data = answer(request)

    assert data["substitution"] == _solved(vector["name"])["substitution"]


@pytest.mark.parametrize(
    "name", [n for n in _NAMES if "alternatives" in _vector(n)], ids=lambda n: n
)
def test_alternatives_are_other_interpretations_or_after_the_defence(name: str) -> None:
    """Alternatives document rejected readings or later features; none is the engine's."""
    for alternative in _vector(name)["alternatives"]:
        params = OUR_PARAMS | alternative.get("params", {})
        assert params != OUR_PARAMS or alternative.get("stage") == "after_defence"


# ---------------------------------------------------------------- expectations


def _rows(data: Json) -> dict[tuple[str, int], Json]:
    proposals = cast("list[Json]", data["substitution"]["proposals"])
    return {(p["class_id"], p["lesson_no"]): p for p in proposals}


def _check_expect(request: Json, expect: Json, data: Json) -> None:
    assert data["status"] in expect["status_in"]
    assert data["substitution"]["step"] == expect["step"]
    rows = _rows(data)
    for lesson in expect["lessons"]:
        row = rows[lesson["class_id"], lesson["lesson_no"]]
        assert ("kind" not in lesson) or row["kind"] == lesson["kind"], row
        assert ("kind_in" not in lesson) or row["kind"] in lesson["kind_in"], row
        assert ("teacher_id" not in lesson) or row.get("teacher_id") == lesson["teacher_id"], row
        assert ("teacher_id_in" not in lesson) or row.get("teacher_id") in lesson["teacher_id_in"]
        assert ("off_profile" not in lesson) or row.get("off_profile") == lesson["off_profile"]
        if "moved_to" in lesson:
            target, moved = lesson["moved_to"], row["moved_to"]
            assert ("date" not in target) or moved["date"] == target["date"], row
            assert ("lesson_no_in" not in target) or moved["lesson_no"] in target["lesson_no_in"]
        assert row["step"] == lesson["step"], row
    for row in rows.values():
        assert row.get("teacher_id") not in expect.get("never_teachers", []), row
        assert row.get("moved_to", {}).get("date") not in expect.get("never_dates", []), row
    if expect.get("extra_rows") == []:
        assert set(rows) == {(x["class_id"], x["lesson_no"]) for x in request["lessons"]}


def _window(request: Json) -> set[str]:
    """Invariant 7's window: school days of iso_week, >= not_before, not non-school, not date."""
    year, week = request["iso_week"].split("-W")
    dates = [dt.date.fromisocalendar(int(year), int(week), d) for d in range(1, 8)]
    return {
        x.isoformat()
        for x in dates
        if x.isoweekday() in request["school_days"]
        and x.isoformat() >= request["not_before"]
        and x.isoformat() not in request.get("non_school_dates", [])
        and x.isoformat() != request["date"]
    }


def _check_invariants(request: Json, data: Json) -> None:
    assert (data["lessons"], data["day_formats"], data["conflicts"]) == ([], [], [])
    proposals = cast("list[Json]", data["substitution"]["proposals"])
    keys = sorted((p["class_id"], p["lesson_no"]) for p in proposals)
    assert keys == sorted((x["class_id"], x["lesson_no"]) for x in request["lessons"])
    assert all(p["date"] == request["date"] for p in proposals)
    assert {p["kind"] for p in proposals} <= {"teacher", "moved", "self_study"}
    assert data["substitution"]["step"] == max(p["step"] for p in proposals)
    window = _window(request)
    for p in proposals:
        if p["kind"] == "moved" and p["step"] == 2:
            assert p["moved_to"]["date"] == request["date"]
        elif p["kind"] == "moved":
            assert p["step"] == 3
            assert p["moved_to"]["date"] in window
        elif p["kind"] == "self_study":
            assert p["step"] == (3 if "non_school_dates" in request else 2)
        else:
            assert p["step"] == 1


# ---------------------------------------------------------------- the golden week builder


@pytest.mark.parametrize("path", WEEK_VECTORS, ids=lambda p: p.stem)
def test_week_builder_reproduces_the_contract_week(path: Path) -> None:
    vector = load(path)
    given = vector["given"]

    built = substitution_golden.week_input(
        given["snapshot"],
        given["template"],
        holidays=given["holidays"],
        absences=given["absences"],
        overlays=given["overlays"],
    )

    assert built == vector["input"]


def test_new_conflicts_mode_answers_the_contract_approve_pairs() -> None:
    pairs = {p.stem: (load(p)["baseline"], load(p)["effective"]) for p in APPROVE_VECTORS}
    expected = {p.stem: load(p)["expected"]["added"] for p in APPROVE_VECTORS}

    outcomes = substitution_golden.new_conflicts(pairs)

    assert {k: substitution_golden.added(o) for k, o in outcomes.items()} == expected


def test_golden_check_reports_what_bad_proposals_add() -> None:
    """The harness itself: proposals that break a rule must show up in ``added`` (a harness that
    dropped the proposals would pass every engine answer vacuously)."""
    request = _vector("01-step1-free-teacher")["given"]["request"]
    date = request["date"]
    bad = {
        # t-2 teaches c-2 in L1 already
        "teacher_double_booked": {
            "kind": "teacher",
            "teacher_id": "t-2",
            "off_profile": True,
            "step": 1,
        },
        # c-1 has its Ukrainian lesson in L2
        "class_double_booked": {
            "kind": "moved",
            "moved_to": {"date": date, "lesson_no": 2},
            "step": 2,
        },
        # t-1 is absent all day
        "teacher_unavailable_day": {
            "kind": "moved",
            "moved_to": {"date": date, "lesson_no": 5},
            "step": 2,
        },
    }
    cases = {
        rule: substitution_golden.weeks(
            request, [{"date": date, "class_id": "c-1", "lesson_no": 1} | row]
        )
        for rule, row in bad.items()
    }
    control = [{"date": date, "class_id": "c-1", "lesson_no": 1, "kind": "self_study", "step": 2}]
    cases["control"] = substitution_golden.weeks(request, control)

    outcomes = substitution_golden.new_conflicts(cases)

    for rule in bad:
        assert rule in {c["rule"] for c in substitution_golden.added(outcomes[rule])}, rule
    assert substitution_golden.added(outcomes["control"]) == []


def test_golden_weeks_follow_the_request() -> None:
    """How a request becomes the two weeks: dates to weekdays, a whole-day absence lists every
    lesson, a non-school school day is a holiday, a missing absences is the absent teacher all
    of the absence dates of the week, and an overlay without kind is a teacher overlay."""
    request = copy.deepcopy(_vector("16-step3-non-school-dates")["given"]["request"])
    request["approved_substitutions"] = [
        {"date": "2026-11-16", "class_id": "c-3", "lesson_no": 2, "teacher_id": "t-5"}
    ]
    without = {k: v for k, v in request.items() if k != "absences"}

    baseline, effective = substitution_golden.weeks(request, [])
    implied, _ = substitution_golden.weeks(without, [])

    assert baseline["holidays"] == [3]
    assert 3 not in baseline["snapshot"]["school_days"]
    assert baseline["absences"] == [
        {"teacher_id": "t-1", "day": 2, "lesson_nos": [1, 2, 3, 4, 5, 6]}
    ]
    assert implied["absences"] == baseline["absences"]
    [substitute] = [
        x for x in baseline["timetable"]["lessons"] if x.get("origin") == "substitution"
    ]
    assert (substitute["day"], substitute["class_id"], substitute["teacher_id"]) == (
        1,
        "c-3",
        "t-5",
    )
    assert effective == baseline
