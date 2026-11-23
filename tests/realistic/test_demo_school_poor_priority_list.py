"""Realistic school: the demo lyceum with an empty list of priority areas (Trello #72).

The demo school is ``lad-solver generate --classes 30 --shelter-ratio 0.55 --seed 0`` (platform
``seed/solver-problem.json``): 30 classes (9 flagged), 808 pupils, 48 teachers, 74 staff,
``Passport(512)``, Mon-Fri, 7 lessons. With no priority subject phase A rewards only the flagged
classes, so before the coverage step (ADR 2026-11-14) 7-9 of the other 21 classes got no
in-person day at all, and with a 2000-person shelter still 1-2. One worker and a deterministic
budget keep the build reproducible; every result passes the validator-lib CLI.

On a deterministic budget the wall time only measures the machine: ~8 s on a quiet laptop, four
times that under load. So the cost is checked deterministically instead: the coverage run must
prove its optimum (every class has a day) within its 15% share of the budget.
"""

from dataclasses import dataclass, replace

import pytest

import golden
from lad_solver.app import solve as solve_module
from lad_solver.app.solve import solve
from lad_solver.cli.generate import generate_problem
from lad_solver.domain.prechecks import precheck
from lad_solver.domain.problem import Params, Passport, Problem
from lad_solver.domain.result import Result, SolveStatus, StopReason
from lad_solver.model import phase_a
from lad_solver.model.phase_a import PhaseAOutcome

TIME_LIMIT_S = 60.0  # the wall clock is only the safety net of the deterministic budget
DETERMINISTIC_S = 10.0  # the coverage step gets 15% of it: 1.5
PARAMS = Params(
    time_limit_s=TIME_LIMIT_S,
    num_workers=1,
    random_seed=0,
    max_deterministic_time=DETERMINISTIC_S,
)
FLAGGED = 9
DAYS = 5


@dataclass(frozen=True, slots=True)
class Solved:
    problem: Problem
    result: Result
    coverage: tuple[SolveStatus, ...]  # status of every coverage run


def demo_school() -> Problem:
    return generate_problem(30, 0.55, seed=0)


def without_priority_areas(p: Problem) -> Problem:
    return replace(p, subjects=tuple(replace(s, priority_area=False) for s in p.subjects))


def _solve(p: Problem) -> Solved:
    real = phase_a.solve_coverage
    statuses: list[SolveStatus] = []

    def spy(*args: object, **kwargs: object) -> PhaseAOutcome:
        outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        statuses.append(outcome.info.status)
        return outcome

    with pytest.MonkeyPatch.context() as m:
        m.setattr(solve_module, "solve_coverage", spy)
        result = solve(p, PARAMS)
    return Solved(p, result, tuple(statuses))


@pytest.fixture(scope="module")
def empty_list() -> Solved:
    return _solve(without_priority_areas(demo_school()))


@pytest.fixture(scope="module")
def big_shelter() -> Solved:
    return _solve(replace(without_priority_areas(demo_school()), capacity=Passport(2000)))


def test_scenario_is_the_demo_school() -> None:
    p = demo_school()
    pupils = sum(c.students for c in p.classes)
    assert (len(p.classes), pupils, len(p.teachers), p.staff) == (30, 808, 48, 74)
    assert p.capacity == Passport(512)
    assert sum(c.priority for c in p.classes) == FLAGGED
    assert p.school_days == tuple(range(1, DAYS + 1))
    assert precheck(p) == ()


def _admissible(solved: Solved) -> Result:
    p, result = solved.problem, solved.result
    assert result.status is SolveStatus.OPTIMAL  # the flagged classes bound the objective
    assert result.stats.stop_reason is StopReason.COMPLETED
    assert golden.violations(p, result) == []
    recomputed = golden.objective(p, result)
    assert (result.stats.objective, result.stats.primary) == (recomputed.value, 0)
    # Coverage is a tie-break: every flagged class keeps all its in-person days.
    assert result.stats.objective == FLAGGED * DAYS
    assert golden.classes_without_in_person_days(p, result) == []
    assert result.stats.wall_time_s < TIME_LIMIT_S
    return result


@pytest.mark.slow
def test_empty_list_gives_every_class_an_in_person_day(empty_list: Solved) -> None:
    _admissible(empty_list)
    assert empty_list.coverage == (SolveStatus.OPTIMAL,)  # converged within its share


@pytest.mark.slow
def test_big_shelter_gives_every_class_an_in_person_day(big_shelter: Solved) -> None:
    # Every class in person every day (150 of 150) is not the goal, one day each is.
    _admissible(big_shelter)
    assert set(big_shelter.coverage) <= {SolveStatus.OPTIMAL}  # phase A may cover all itself
