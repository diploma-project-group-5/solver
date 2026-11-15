"""Realistic scenario: the large city lyceum in alert mode 1.2 with tight seats (Trello #72).

The lyceum of ``_lyceum`` (50 classes, 1281 pupils, 75 teachers, 30 staff) with its own priority
list, so no class is a static orphan, and an air-raid alert shelter of 300 equipped seats and a
370-person passport: seats bind (under a quarter of the pupils fit a day), and some classes the
objective does reward end up without an in-person day after the ladder ("dynamic orphans").

What this gate checks, for the quality of 1.2: zero ``seats`` and zero ``passport`` violations
(separately, as spec testing requires), the seats really bind, and the coverage step runs on the
dynamic-orphan path and strictly improves the ladder's plan. What it does not: on a
deterministic budget it cannot see the wall-clock reserve, and a regression that gave the step
more det work would only improve it; the budget split is pinned by the frozen-clock unit tests in
``test_engine_coverage`` (ADR 2026-11-14, addendum 2026-11-15).

One worker and a deterministic budget keep the result reproducible; every result passes the
validator-lib CLI, and capacity conflicts are only filtered from its report by rule and condition.
"""

from dataclasses import dataclass, replace

import pytest

import golden
from lad_solver.app import solve as solve_module
from lad_solver.app.solve import solve
from lad_solver.domain.prechecks import precheck
from lad_solver.domain.problem import AlertLimits, Params, Problem, TimetableKind
from lad_solver.domain.result import DayFormat, Result, SolveStatus
from lad_solver.model import phase_a
from lad_solver.model.index import Index, build_index
from lad_solver.model.phase_a import DayPlan, PhaseAOutcome, static_orphans, zero_day_classes
from realistic._lyceum import DAYS, PRIORITY_CLASSES, STAFF, build_lyceum

TIME_LIMIT_S = 600.0  # only the safety net of the deterministic budget (180 s seen under load)
# The smallest budget at which the step takes a better plan: 14 -> 13 already at 10 (and at 15,
# 20, 30); the coverage step gets 15% of it: 1.5.
DETERMINISTIC_S = 10.0
PARAMS = Params(
    time_limit_s=TIME_LIMIT_S,
    num_workers=1,
    random_seed=0,
    max_deterministic_time=DETERMINISTIC_S,
)
SEATS, PASSPORT = 300, 370
# OR-Tools 9.15.6755, det 10, seed 0: macOS arm64 14 -> 13, linux x86_64 (CI) 15 -> 14.
# The bound is the larger of the two, without slack; recalibrate on an OR-Tools update.
MAX_WITHOUT_IN_PERSON = 14
FOUND = frozenset({SolveStatus.OPTIMAL, SolveStatus.FEASIBLE})


@dataclass(frozen=True, slots=True)
class Solved:
    problem: Problem
    result: Result
    coverage: tuple[tuple[SolveStatus, int, int | None], ...]  # status, zero-day before / after


def lyceum() -> Problem:
    return replace(
        build_lyceum(),
        kind=TimetableKind.ALERT,
        capacity=AlertLimits(seats=SEATS, passport=PASSPORT),
    )


@pytest.fixture(scope="module")
def solved() -> Solved:
    p = lyceum()
    real = phase_a.solve_coverage
    runs: list[tuple[SolveStatus, int, int | None]] = []

    def spy(
        index: Index, caps: object, plan: DayPlan, *args: object, **kwargs: object
    ) -> PhaseAOutcome:
        outcome = real(index, caps, plan, *args, **kwargs)  # pyright: ignore[reportArgumentType]
        after = None if outcome.plan is None else len(zero_day_classes(index, outcome.plan))
        runs.append((outcome.info.status, len(zero_day_classes(index, plan)), after))
        return outcome

    with pytest.MonkeyPatch.context() as m:
        m.setattr(solve_module, "solve_coverage", spy)
        result = solve(p, PARAMS)
    return Solved(p, result, tuple(runs))


def test_scenario_is_a_large_lyceum_with_tight_seats() -> None:
    p = lyceum()
    pupils = sum(c.students for c in p.classes)
    assert (len(p.classes), pupils, len(p.teachers), p.staff) == (50, 1281, 75, STAFF)
    assert sum(c.priority for c in p.classes) == PRIORITY_CLASSES
    assert p.kind is TimetableKind.ALERT
    assert pupils > 4 * SEATS  # under a quarter of the pupils fit a day
    assert precheck(p) == ()
    assert static_orphans(build_index(p)) == []  # the dynamic-orphan path, not the static one


@pytest.mark.slow
def test_alert_build_is_admissible(solved: Solved) -> None:
    assert solved.result.status in FOUND
    assert golden.violations(solved.problem, solved.result) == []
    assert not solved.result.stats.quality_relaxed
    wall = solved.result.stats.wall_time_s
    assert wall < TIME_LIMIT_S / 2, (
        f"wall clock hit ({wall:.1f} s): the result depends on the machine"
    )


@pytest.mark.slow
def test_no_seats_violations(solved: Solved) -> None:
    assert golden.capacity_conflicts(solved.problem, solved.result, "seats") == []


@pytest.mark.slow
def test_no_passport_violations(solved: Solved) -> None:
    assert golden.capacity_conflicts(solved.problem, solved.result, "passport") == []


@pytest.mark.slow
def test_seats_is_binding(solved: Solved) -> None:
    # Teeth for the two filters above and proof that seats are tight: one more class in person
    # on the busiest day is a seats conflict that validator-lib reports.
    p, result = solved.problem, solved.result
    students = {c.id: c.students for c in p.classes}
    peak = max(DAYS, key=lambda d: golden.day_load(p, result, d).pupils)
    remote = [f.class_id for f in result.day_formats if f.day == peak and not f.in_person]
    smallest = min(remote, key=lambda c: (students[c], c))
    flipped = tuple(
        DayFormat(f.class_id, f.day, in_person=True)
        if (f.class_id, f.day) == (smallest, peak)
        else f
        for f in result.day_formats
    )
    more = replace(result, day_formats=flipped)
    assert golden.capacity_conflicts(p, more, "seats") != []


@pytest.mark.slow
def test_coverage_step_improves_the_ladder(solved: Solved) -> None:
    (run,) = solved.coverage
    _status, before, after = run
    assert after is not None, run
    assert after < before, run  # the step took a better plan
    zero_day = golden.classes_without_in_person_days(solved.problem, solved.result)
    assert len(zero_day) == after, (run, zero_day)  # the engine and validator-lib agree
    assert after <= MAX_WITHOUT_IN_PERSON, run
