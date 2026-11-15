"""Properties of the substitution engine (ADR 2026-11-15-solver-substitution-engine).

On random school weeks, stale ones included (``substitution_builders``):

- there is always a solution: OPTIMAL or FEASIBLE, every lesson to cover has exactly one
  proposal, on the request's date, and there is no other row (allow_support_moves is off);
- the steps are monotone: running further never adds self-study, and what step k fixed stays;
- the answer is deterministic: the same request gives the same proposals;
- a job that sees an earlier job's proposals as approved substitutions still answers;
- every answer adds no conflict to its week (one ``lad-validate --mode new-conflicts`` batch).

Requests go through the production mapping (schema, ids, the baseline week) into the engine.
"""

import os
import random
import subprocess
import sys
from dataclasses import replace
from functools import cache
from typing import Any, cast
from uuid import UUID

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from kafka_support import encode

import substitution_golden
from lad_solver.adapters.kafka.mapping import (
    SubstitutionAccepted,
    parse_substitution_request,
    substitution_data,
)
from lad_solver.app.substitute import substitute
from lad_solver.domain.result import SolveStatus
from lad_solver.domain.substitution import (
    ProposalKind,
    SubstitutionParams,
    SubstitutionRequest,
    SubstitutionResult,
)
from substitution_builders import date_of, envelope, random_request

type Json = dict[str, Any]

PARAMS = SubstitutionParams(time_limit_s=30.0)
SETTINGS = settings(max_examples=40, deadline=None, derandomize=True)
SEEDS = st.integers(min_value=0, max_value=2**32 - 1)
KINDS = st.sampled_from(["main", "alert"])


def parse(data: Json) -> SubstitutionRequest:
    outcome = parse_substitution_request(encode(envelope(data)))
    assert isinstance(outcome, SubstitutionAccepted), outcome
    return outcome.request


def request_of(seed: int, kind: str) -> tuple[Json, SubstitutionRequest]:
    data = random_request(seed, kind=kind)
    assume(data["lessons"])  # a week without a lesson of the absent teacher is not a job
    return data, parse(data)


def _self_study(result: SubstitutionResult) -> int:
    return sum(p.kind is ProposalKind.SELF_STUDY for p in result.proposals)


@SETTINGS
@given(seed=SEEDS, kind=KINDS)
def test_there_is_always_a_solution_with_one_proposal_per_lesson(seed: int, kind: str) -> None:
    _, request = request_of(seed, kind)

    result = substitute(request, PARAMS)

    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert [p.slot for p in result.proposals] == list(request.lessons)
    assert result.step == max(p.step for p in result.proposals)
    for p in result.proposals:
        assert (p.kind is ProposalKind.TEACHER) == (p.step == 1)
        if p.kind is ProposalKind.SELF_STUDY:
            assert p.step == request.last_step
        if p.kind is ProposalKind.MOVED:
            assert p.moved_to is not None
            assert (p.moved_to.date == request.date) == (p.step == 2)
            assert p.step == 2 or p.moved_to.date in request.window()


@SETTINGS
@given(seed=SEEDS, kind=KINDS)
def test_steps_never_add_self_study_and_keep_what_they_fixed(seed: int, kind: str) -> None:
    _, request = request_of(seed, kind)

    runs = [substitute(request, replace(PARAMS, last_step=k)) for k in (1, 2, 3)]

    self_study = [_self_study(r) for r in runs]
    assert self_study == sorted(self_study, reverse=True)
    for k, run in enumerate(runs):
        fixed = {p.slot: p for p in run.proposals if p.kind is not ProposalKind.SELF_STUDY}
        for later in runs[k + 1 :]:
            assert {p.slot: p for p in later.proposals if p.slot in fixed} == fixed


@SETTINGS
@given(seed=SEEDS, kind=KINDS)
def test_the_same_request_gives_the_same_answer(seed: int, kind: str) -> None:
    _, request = request_of(seed, kind)

    first, second = substitute(request, PARAMS), substitute(request, PARAMS)

    assert (first.status, first.proposals) == (second.status, second.proposals)


def proposal_rows(data: Json, result: SubstitutionResult) -> list[Json]:
    """The answer's proposals as they travel in solve.completed."""
    completed = substitution_data(UUID(data["job_id"]), result, None)
    return cast("list[Json]", cast("Json", completed["substitution"])["proposals"])


def chained(seed: int, kind: str) -> Json:
    """An earlier job of the same week (another absent teacher) is solved and approved first."""
    data = random_request(seed, kind=kind)
    rng = random.Random(seed)  # noqa: S311 - test data
    teaching = sorted(
        {
            (x["teacher_id"], x["day"])
            for x in data["template"]["lessons"]
            if x["teacher_id"] != data["absent_teacher_id"]
            and date_of(x["day"]) not in data.get("non_school_dates", [])
        }
    )
    if not teaching:
        return data
    other, day = rng.choice(teaching)
    date = date_of(day)
    absences = [a for a in data["absences"] if (a["teacher_id"], a["date"]) != (other, date)]
    absences.append({"teacher_id": other, "date": date, "lesson_nos": None})
    first = data | {
        "date": date,
        "absence_dates": [date],
        "absent_teacher_id": other,
        "lessons": [
            {"class_id": x["class_id"], "lesson_no": x["lesson_no"]}
            for x in data["template"]["lessons"]
            if (x["teacher_id"], x["day"]) == (other, day)
        ],
        "not_before": date_of(1),
        "absences": absences,
    }
    proposals = proposal_rows(first, substitute(parse(first), PARAMS))
    # timetable stores approved proposals as overlays: the same fields without step/off_profile
    approved = [{k: v for k, v in p.items() if k not in {"step", "off_profile"}} for p in proposals]
    covered = {(r["class_id"], r["lesson_no"]) for r in approved if r["date"] == data["date"]}
    lessons = [x for x in data["lessons"] if (x["class_id"], x["lesson_no"]) not in covered]
    return data | {"approved_substitutions": approved, "absences": absences, "lessons": lessons}


@settings(max_examples=25, deadline=None, derandomize=True)
@given(seed=SEEDS, kind=KINDS)
def test_a_job_after_approved_substitutions_still_answers(seed: int, kind: str) -> None:
    data = chained(seed, kind)
    assume(data["lessons"])
    request = parse(data)

    result = substitute(request, PARAMS)

    approved = {(a["date"], a["class_id"], a["lesson_no"]) for a in data["approved_substitutions"]}
    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    for p in result.proposals:
        assert (p.slot.date.isoformat(), p.slot.class_id, p.slot.lesson_no) not in approved


# ---------------------------------------------------------------- golden, one batch

GOLDEN_SEEDS = range(60)


@cache
def _golden_cases() -> dict[str, tuple[Json, Json]]:
    cases: dict[str, tuple[Json, Json]] = {}
    for seed in GOLDEN_SEEDS:
        for kind in ("main", "alert"):
            for name, data in (
                (f"{kind}-{seed}", random_request(seed, kind=kind)),
                (f"{kind}-{seed}-chained", chained(seed, kind)),
            ):
                if not data["lessons"]:
                    continue
                result = substitute(parse(data), PARAMS)
                cases[name] = substitution_golden.weeks(data, proposal_rows(data, result))
    return cases


def test_random_weeks_cover_moves_and_capacity_pressure() -> None:
    """The batch below is not vacuous: it holds every kind of proposal of every step."""
    kinds: set[tuple[str, int]] = set()
    for seed in GOLDEN_SEEDS:
        data = random_request(seed)
        if data["lessons"]:
            kinds |= {(p.kind.value, p.step) for p in substitute(parse(data), PARAMS).proposals}
    assert kinds >= {("teacher", 1), ("moved", 2), ("moved", 3), ("self_study", 3)}


def test_every_answer_on_random_weeks_adds_no_conflict() -> None:
    outcomes = substitution_golden.new_conflicts(_golden_cases())

    added = {name: substitution_golden.added(o) for name, o in outcomes.items()}
    assert {name: a for name, a in added.items() if a} == {}
    assert len(added) >= 200


@pytest.mark.parametrize("kind", ["main", "alert"])
def test_random_weeks_parse(kind: str) -> None:
    for seed in range(20):
        data = random_request(seed, kind=kind)
        if data["lessons"]:
            parse(data)


# ---------------------------------------------------------------- Europe/Kyiv dates

_IN_ZONE = """
import json, sys, uuid
from lad_solver.adapters.kafka.mapping import parse_substitution_request, substitution_data
from lad_solver.app.substitute import substitute
from lad_solver.domain.substitution import SubstitutionParams
outcome = parse_substitution_request(sys.stdin.buffer.read())
result = substitute(outcome.request, SubstitutionParams(time_limit_s=30.0))
print(json.dumps(substitution_data(uuid.uuid4(), result, None)["substitution"], sort_keys=True))
"""


@pytest.mark.slow
def test_the_answer_does_not_depend_on_the_server_time_zone() -> None:
    """The engine never reads the clock or the zone: dates are Kyiv dates from timetable, and the
    same request gives the same bytes in a process east of UTC+14 or west of UTC-8."""
    data = random_request(4, kind="main") | {"not_before": "2026-11-17"}
    raw = encode(envelope(data))
    answers = {
        zone: subprocess.run(  # noqa: S603 - fixed interpreter and script
            [sys.executable, "-c", _IN_ZONE],
            input=raw,
            capture_output=True,
            env=os.environ | {"TZ": zone},
            check=True,
            timeout=120,
        ).stdout
        for zone in ("UTC", "Pacific/Kiritimati", "America/Los_Angeles", "Europe/Kyiv")
    }

    assert len(set(answers.values())) == 1, answers
    assert b'"proposals"' in answers["UTC"]
