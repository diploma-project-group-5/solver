"""Realistic archetype: a large city lyceum with 50 classes (performance test, main timetable 1.1).

Kyiv averages 25.6 pupils per class; schools of 1500+ pupils are rare, so this is the upper end
of what the demo must handle: grades 1-6 x 5 parallels + grades 7-11 x 4 = 50 classes, about
1300 pupils, 75 teachers, about 1400 lesson-hours a week, 5 days x 7 lessons. The shelter
passport (700) holds about half of the ~1400 people, so capacity is the binding constraint.

The problem (``_lyceum``) is built deterministically (fixed seed); one 90 s solve on 2 workers is
shared by all checks, and every result is verified by the validator-lib CLI.
"""

from collections import Counter

import pytest

import golden
from lad_solver.app.solve import solve
from lad_solver.domain.prechecks import precheck
from lad_solver.domain.problem import Params, Problem
from lad_solver.domain.result import Result, SolveStatus, StopReason
from realistic._lyceum import DAYS, PASSPORT, PRIORITY_CLASSES, STAFF, build_lyceum

TIME_LIMIT_S = 90.0
WALL_SLACK_S = 5.0  # phase B retries and Python around CP-SAT, as in the other scenarios
PARAMS = Params(time_limit_s=TIME_LIMIT_S, num_workers=2, random_seed=0)
FIRST_SOLUTION_MAX_S = 2.0  # prototype: 0.62 s
MIN_IN_PERSON_SHARE = 0.40  # prototype: 0.48; the passport allows at most ~0.52


@pytest.fixture(scope="module")
def lyceum() -> Problem:
    return build_lyceum()


@pytest.fixture(scope="module")
def solved(lyceum: Problem) -> Result:
    return solve(lyceum, PARAMS)


def _in_person_days(result: Result) -> Counter[str]:
    return Counter(f.class_id for f in result.day_formats if f.in_person)


def test_scenario_matches_archetype_and_passes_prechecks(lyceum: Problem) -> None:
    pupils = sum(c.students for c in lyceum.classes)
    lesson_hours = sum(r.hours for r in lyceum.rows)
    people = pupils + STAFF + len(lyceum.teachers)
    assert len(lyceum.classes) == 50
    assert 1250 <= pupils <= 1350
    assert 74 <= len(lyceum.teachers) <= 80
    assert 1350 <= lesson_hours <= 1450
    assert 0.45 <= PASSPORT / people <= 0.55  # shelter for about half of the building
    assert sum(c.priority for c in lyceum.classes) == PRIORITY_CLASSES
    assert 5 <= sum(bool(t.unavailable_days) for t in lyceum.teachers) <= 10
    assert precheck(lyceum) == ()


@pytest.mark.slow
def test_large_build_is_feasible_valid_and_not_relaxed(lyceum: Problem, solved: Result) -> None:
    assert solved.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert golden.violations(lyceum, solved) == []
    assert not solved.stats.quality_relaxed
    expected_stop = (
        StopReason.COMPLETED if solved.status is SolveStatus.OPTIMAL else StopReason.TIME_LIMIT
    )
    assert solved.stats.stop_reason is expected_stop
    recomputed = golden.objective(lyceum, solved)
    assert (solved.stats.primary, solved.stats.secondary, solved.stats.objective) == (
        recomputed.primary,
        recomputed.secondary,
        recomputed.value,
    )


@pytest.mark.slow
def test_passport_holds_every_day_and_is_actually_used(lyceum: Problem, solved: Result) -> None:
    pupils = sum(c.students for c in lyceum.classes)
    in_person_pupils = 0
    for day in DAYS:
        load = golden.day_load(lyceum, solved, day)
        assert load.people <= PASSPORT, f"day {day}: {load}"
        in_person_pupils += load.pupils
    share = in_person_pupils / (pupils * len(DAYS))
    assert share >= MIN_IN_PERSON_SHARE, f"in-person share {share:.3f}"


@pytest.mark.slow
def test_wall_time_and_first_solution_within_budget(solved: Result) -> None:
    assert solved.stats.wall_time_s <= TIME_LIMIT_S + WALL_SLACK_S
    assert solved.stats.first_solution_s is not None
    assert solved.stats.first_solution_s < FIRST_SOLUTION_MAX_S
    assert solved.stats.num_workers == 2


@pytest.mark.slow
def test_unavailable_days_and_priority_flag_are_honoured(lyceum: Problem, solved: Result) -> None:
    away = {t.id: t.unavailable_days for t in lyceum.teachers if t.unavailable_days}
    assert away
    assert not [x for x in solved.lessons if x.day in away.get(x.teacher_id, frozenset())]
    # The priority flag is only a tie-break below W * primary, so "flagged classes get more
    # in-person days on average" is not a guaranteed property; check that it is counted instead.
    days = _in_person_days(solved)
    flagged_days = sum(days[c.id] for c in lyceum.classes if c.priority)
    assert solved.stats.secondary == flagged_days
    assert flagged_days > 0
