"""Realistic scenario: the large city lyceum with a poor priority list, alert mode 1.2 (Trello #72).

The lyceum of ``_lyceum`` (50 classes, ~1300 pupils) with mathematics as its only priority area
(s-math, s-alg, s-geom): every class has those lessons, so no class is a static orphan, yet only
300 equipped seats for the pupils leave some classes without an in-person day after the ladder
("dynamic orphans"). That is where the coverage step used to get no time on the wall clock: the
ladder kept the whole phase A budget and the step never ran (ADR 2026-11-14, addendum 2026-11-15:
8 such classes on 90 s of wall clock, 6 -> 2 with the reserve). This scenario runs on a
deterministic budget, where the step already had its share before the fix, so it is a regression
gate for the step on a large school, not for the wall-clock bug; the frozen-clock unit tests of
``test_engine_coverage`` are that gate. On the 1.1
timetable the 700-person passport covers every class even with this list, so 1.2 it is.

A deterministic budget and one worker keep the result reproducible (wall-clock counts would make
the test flaky); every result passes the validator-lib CLI. The budget split itself is pinned by
the unit tests of ``test_engine_coverage``; this scenario checks that the step runs on the
dynamic-orphan path of a large school and improves the ladder's plan.
"""

from dataclasses import dataclass, replace

import pytest

import golden
from lad_solver.app import solve as solve_module
from lad_solver.app.solve import solve
from lad_solver.domain.prechecks import precheck
from lad_solver.domain.problem import AlertLimits, Params, Problem, TimetableKind
from lad_solver.domain.result import Result, SolveStatus
from lad_solver.model import phase_a
from lad_solver.model.index import Index, build_index
from lad_solver.model.phase_a import DayPlan, PhaseAOutcome, static_orphans, zero_day_classes
from realistic._lyceum import build_lyceum

TIME_LIMIT_S = 600.0  # only the safety net of the deterministic budget (180 s seen under load)
DETERMINISTIC_S = 10.0  # the coverage step gets 15% of it: 1.5
PARAMS = Params(
    time_limit_s=TIME_LIMIT_S,
    num_workers=1,
    random_seed=0,
    max_deterministic_time=DETERMINISTIC_S,
)
PRIORITY_AREA = frozenset({"s-math", "s-alg", "s-geom"})
SEATS, PASSPORT = 300, 700
# OR-Tools 9.15.6755, det 10, seed 0: macOS arm64 11 -> 10, linux x86_64 (CI) 15 -> 14.
# The bound is the larger of the two, without slack; recalibrate on an OR-Tools update.
MAX_WITHOUT_IN_PERSON = 14


@dataclass(frozen=True, slots=True)
class Solved:
    problem: Problem
    result: Result
    coverage: tuple[tuple[int, int | None], ...]  # zero-day classes before / after every step run


def poor_lyceum() -> Problem:
    p = build_lyceum()
    return replace(
        p,
        kind=TimetableKind.ALERT,
        capacity=AlertLimits(seats=SEATS, passport=PASSPORT),
        subjects=tuple(replace(s, priority_area=s.id in PRIORITY_AREA) for s in p.subjects),
    )


@pytest.fixture(scope="module")
def solved() -> Solved:
    p = poor_lyceum()
    real = phase_a.solve_coverage
    runs: list[tuple[int, int | None]] = []

    def spy(
        index: Index, caps: object, plan: DayPlan, *args: object, **kwargs: object
    ) -> PhaseAOutcome:
        outcome = real(index, caps, plan, *args, **kwargs)  # pyright: ignore[reportArgumentType]
        after = None if outcome.plan is None else len(zero_day_classes(index, outcome.plan))
        runs.append((len(zero_day_classes(index, plan)), after))
        return outcome

    with pytest.MonkeyPatch.context() as m:
        m.setattr(solve_module, "solve_coverage", spy)
        result = solve(p, PARAMS)
    return Solved(p, result, tuple(runs))


def test_poor_list_leaves_no_static_orphan() -> None:
    p = poor_lyceum()
    assert {s.id for s in p.subjects if s.priority_area} == PRIORITY_AREA
    assert precheck(p) == ()
    assert static_orphans(build_index(p)) == []  # the dynamic-orphan path, not the static one


@pytest.mark.slow
def test_coverage_step_improves_a_large_school_without_static_orphans(solved: Solved) -> None:
    assert solved.result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert golden.violations(solved.problem, solved.result) == []
    assert not solved.result.stats.quality_relaxed
    (run,) = solved.coverage  # the step ran although no class is a static orphan
    before, after = run
    assert after is not None, run
    assert after < before, run  # ... and its plan was taken
    zero_day = golden.classes_without_in_person_days(solved.problem, solved.result)
    assert len(zero_day) == after <= MAX_WITHOUT_IN_PERSON, (run, zero_day)
    wall = solved.result.stats.wall_time_s
    assert wall < TIME_LIMIT_S / 2, (
        f"wall clock hit ({wall:.1f} s): the result depends on the machine"
    )
