"""Coverage step after phase A (ADR 2026-11-14): every class gets an in-person day when it can.

Phase A rewards an in-person day only through priority lessons or the class flag, so the format
of a class with neither (an "orphan") is left to chance. The coverage step maximises the number
of classes with at least one in-person day while keeping the phase A objective at least at the
value of the plan it starts from: a class the shelter cannot take without lowering the objective
stays remote (a tie-break, never a hard minimum). Every solved result passes the validator-lib CLI.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

import golden
from builders import OTHER, PRIO, klass, problem, row, teacher
from lad_solver.app import solve as solve_module
from lad_solver.app.solve import solve
from lad_solver.cli.generate import generate_problem
from lad_solver.cli.io import Json, result_to_json
from lad_solver.domain.problem import Params, Passport, Problem
from lad_solver.domain.result import Result, SolveStatus, StopReason
from lad_solver.model import phase_a
from lad_solver.model.cpsat import RunInfo
from lad_solver.model.index import Index, build_index
from lad_solver.model.phase_a import DayPlan, PhaseAOutcome, static_orphans
from lad_solver.model.progress import Progress
from lad_solver.model.stop import StopHandle

PARAMS = Params(time_limit_s=10, num_workers=1, random_seed=0)
FOUND = {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
WALL_CLOCK_STATS = frozenset({"wall_time_s", "first_solution_s", "last_improvement_s"})


def _zero_day(p: Problem, result: Result) -> list[str]:
    """Classes without an in-person day, from validator-lib, once the result is admissible."""
    assert result.status in FOUND, result.status
    assert golden.violations(p, result) == []
    return sorted(golden.classes_without_in_person_days(p, result))


def _spy(
    monkeypatch: pytest.MonkeyPatch, after: Callable[[PhaseAOutcome], None] | None = None
) -> list[PhaseAOutcome]:
    """Record every coverage run; ``after`` runs once a run has returned."""
    real = phase_a.solve_coverage
    outcomes: list[PhaseAOutcome] = []

    def spy(*args: object, **kwargs: object) -> PhaseAOutcome:
        outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        outcomes.append(outcome)
        if after is not None:
            after(outcome)
        return outcome

    monkeypatch.setattr(solve_module, "solve_coverage", spy)
    return outcomes


def _must_not_run(*_args: object, **_kwargs: object) -> PhaseAOutcome:
    msg = "the coverage step must not run here"
    raise AssertionError(msg)


def _orphans() -> Problem:
    """A flagged class with priority lessons and three classes with neither (W = 1 * 2 + 1)."""
    return problem(
        classes=(klass("a", 20, priority=True), klass("b", 25), klass("c", 22), klass("d", 18)),
        teachers=(teacher("t1"), teacher("t2"), teacher("t3"), teacher("t4")),
        rows=(
            row("a", PRIO, "t1", 2),
            row("b", OTHER, "t2", 2),
            row("c", OTHER, "t3", 2),
            row("d", OTHER, "t4", 2),
        ),
        days=(1, 2),
    )  # the default Passport(10_000) holds everyone


ORPHANS_OBJECTIVE = 3 * 40 + 2  # a in person on both days with one priority lesson each


def _exact_capacity() -> Problem:
    """Exactly staff + the flagged class + its teacher; b alone fits, but only on a's day."""
    return problem(
        classes=(klass("a", 20, priority=True), klass("b", 15)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 2), row("b", OTHER, "t2", 2)),
        capacity=Passport(2 + 20 + 1),
        days=(1, 2),
        staff=2,
    )


def test_coverage_gives_every_orphan_an_in_person_day() -> None:
    p = _orphans()
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.OPTIMAL
    assert result.stats.stop_reason is StopReason.COMPLETED
    assert _zero_day(p, result) == []
    assert result.stats.objective == golden.objective(p, result).value == ORPHANS_OBJECTIVE


def test_coverage_spreads_days_under_tight_capacity() -> None:
    # Room for the staff and one class with its teacher a day: each class gets one of the days.
    p = problem(
        classes=(klass("a", 20), klass("b", 25)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", OTHER, "t1", 2), row("b", OTHER, "t2", 2)),
        capacity=Passport(2 + 25 + 1),
        days=(1, 2),
        staff=2,
    )
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.OPTIMAL
    assert _zero_day(p, result) == []
    assert Counter(c for c, _ in golden.in_person_days(result)) == {"a": 1, "b": 1}


def test_coverage_soft_when_capacity_exact() -> None:
    # Taking b in person costs a one in-person day (121 < 122): the objective wins, b stays
    # remote and the solve stays OPTIMAL. A hard minimum would have made this INFEASIBLE.
    p = _exact_capacity()
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.OPTIMAL
    assert result.stats.stop_reason is StopReason.COMPLETED
    assert result.stats.objective == 3 * 40 + 2
    assert _zero_day(p, result) == ["b"]


def test_coverage_skipped_without_zero_day_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every class has priority lessons (no reserve) and the shelter takes both (no zero-day class).
    monkeypatch.setattr(solve_module, "solve_coverage", _must_not_run)
    p = problem(
        classes=(klass("a", 20, priority=True), klass("b", 21)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 1), row("b", PRIO, "t2", 1)),
        days=(1,),
    )
    assert static_orphans(build_index(p)) == []
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.OPTIMAL
    assert _zero_day(p, result) == []


def test_static_orphans_are_the_classes_no_objective_term_rewards() -> None:
    p = problem(
        classes=(klass("a", 20, priority=True), klass("b", 25), klass("c", 22), klass("d", 0)),
        teachers=(teacher("t1"), teacher("t2"), teacher("t3"), teacher("t4")),
        rows=(
            row("a", OTHER, "t1", 1),  # flagged: the secondary objective rewards it
            row("b", PRIO, "t2", 1),  # priority lessons: the primary objective rewards it
            row("c", OTHER, "t3", 1),  # neither
            row("d", PRIO, "t4", 1),  # priority lessons of no pupils earn nothing
        ),
    )
    assert static_orphans(build_index(p)) == ["c", "d"]


def test_coverage_stopped_keeps_step1_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = StopHandle()
    outcomes = _spy(monkeypatch, after=lambda _outcome: handle.stop())  # SIGTERM as it ends
    p = _orphans()

    result = solve(p, PARAMS, handle)

    assert [o.plan is not None for o in outcomes] == [True]  # the step had a better plan ...
    assert _zero_day(p, result) == ["b", "c", "d"]  # ... a stopped job keeps the phase A one
    assert result.stats.stop_reason is StopReason.STOPPED
    assert result.stats.objective == ORPHANS_OBJECTIVE


def test_coverage_not_started_once_the_job_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = StopHandle()
    real = phase_a.solve_phase_a

    def stopped_after(*args: object, **kwargs: object) -> PhaseAOutcome:
        outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        handle.stop()  # SIGTERM lands as the phase A run ends
        return outcome

    monkeypatch.setattr(solve_module, "solve_phase_a", stopped_after)
    monkeypatch.setattr(solve_module, "solve_coverage", _must_not_run)
    p = _orphans()

    result = solve(p, PARAMS, handle)

    assert result.stats.stop_reason is StopReason.STOPPED
    assert _zero_day(p, result) == ["b", "c", "d"]


def test_coverage_does_not_touch_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = Progress()
    marks: list[tuple[float | None, float | None]] = []
    real = phase_a.solve_coverage

    def spy(*args: object, **kwargs: object) -> PhaseAOutcome:
        marks.append((progress.first_solution_at, progress.last_improvement_at))
        outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        marks.append((progress.first_solution_at, progress.last_improvement_at))
        assert outcome.info.status is SolveStatus.OPTIMAL
        return outcome

    monkeypatch.setattr(solve_module, "solve_coverage", spy)
    p = _orphans()

    result = solve(p, PARAMS, progress=progress)

    before, after = marks
    assert before[1] is not None  # phase A recorded its improvements ...
    assert after == before  # ... the coverage objective (another scale) reported none
    assert _zero_day(p, result) == []


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_coverage_runs_after_the_stall_rule_fired(monkeypatch: pytest.MonkeyPatch) -> None:
    # A run that reported to the stalled progress would return UNKNOWN at once (ADR 2026-11-08).
    real = phase_a.solve_phase_a
    clock = _Clock()

    def stalled(*args: object, progress: Progress | None = None, **kwargs: object) -> PhaseAOutcome:
        outcome = real(*args, progress=progress, **kwargs)  # pyright: ignore[reportArgumentType]
        assert progress is not None
        clock.now += 100.0  # a long silence after the probe's solutions
        assert progress.check()  # the no-improvement rule fires during the probe
        return replace(outcome, info=replace(outcome.info, status=SolveStatus.FEASIBLE))

    monkeypatch.setattr(solve_module, "solve_phase_a", stalled)
    outcomes = _spy(monkeypatch)
    p = _orphans()
    params = Params(time_limit_s=10, num_workers=1, random_seed=0, stall_min_s=5.0)

    result = solve(p, params, progress=Progress(clock=clock))

    assert [o.info.status for o in outcomes] == [SolveStatus.OPTIMAL]
    assert result.status is SolveStatus.FEASIBLE  # status and stop reason stay phase A's
    assert result.stats.stop_reason is StopReason.NO_IMPROVEMENT
    assert result.stats.objective == ORPHANS_OBJECTIVE
    assert _zero_day(p, result) == []


def _capped_infeasible() -> Problem:
    """t2 comes on day 2 only and fills it, so a's three priority lessons all fall on day 1.

    That is over the row-day cap (2): only the relaxed run finds a timetable. b is an orphan.
    """
    return problem(
        classes=(klass("a", 20), klass("b", 25)),
        teachers=(teacher("t1"), teacher("t2", unavailable=(1,)), teacher("t3")),
        rows=(row("a", PRIO, "t1", 3), row("a", OTHER, "t2", 3), row("b", OTHER, "t3", 2)),
        days=(1, 2),
        lessons_per_day=3,
    )


def test_coverage_uses_relaxed_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = _spy(monkeypatch)
    p = _capped_infeasible()

    result = solve(p, PARAMS)

    assert result.stats.quality_relaxed
    # Under the capped caps the relaxed plan is no solution: the floor run would be INFEASIBLE.
    assert [o.info.status for o in outcomes] == [SolveStatus.OPTIMAL]
    assert result.status is SolveStatus.OPTIMAL
    assert result.stats.objective == 3 * 20  # W = 1: no flagged class
    assert _zero_day(p, result) == []


def _no_classes(*_args: object) -> list[str]:
    return []


def _fake_coverage(
    make: Callable[[Index, DayPlan], DayPlan],
) -> Callable[..., PhaseAOutcome]:
    """A coverage run that "finds" ``make(index, plan)`` as OPTIMAL."""

    def fake(index: Index, _caps: object, plan: DayPlan, *_: object, **__: object) -> PhaseAOutcome:
        return PhaseAOutcome(RunInfo(SolveStatus.OPTIMAL, None), make(index, plan))

    return fake


def test_coverage_plan_below_the_floor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Everyone in person on day 1 only: no zero-day class left, but a loses day 2 (61 < 122).
    def day_one(index: Index, plan: DayPlan) -> DayPlan:
        return DayPlan(plan.loads, frozenset((c, 1) for c in range(len(index.students))))

    monkeypatch.setattr(solve_module, "solve_coverage", _fake_coverage(day_one))
    p = _orphans()
    result = solve(p, PARAMS)
    assert _zero_day(p, result) == ["b", "c", "d"]
    assert result.stats.objective == ORPHANS_OBJECTIVE


def test_coverage_plan_without_fewer_zero_day_classes_keeps_step1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _exact_capacity()
    with monkeypatch.context() as m:  # phase A alone: no reserve and no coverage step
        m.setattr(solve_module, "static_orphans", _no_classes)
        m.setattr(solve_module, "zero_day_classes", _no_classes)
        step1 = solve(p, PARAMS)

    def b_elsewhere(_index: Index, plan: DayPlan) -> DayPlan:
        # The same objective and the same zero-day class (b): only b's lessons sit elsewhere.
        b_row = 1
        moved = {(b_row, 1): 2} if plan.loads.get((b_row, 1)) != 2 else {(b_row, 2): 2}
        loads = {k: n for k, n in plan.loads.items() if k[0] != b_row} | moved
        return DayPlan(loads, plan.in_person)

    monkeypatch.setattr(solve_module, "solve_coverage", _fake_coverage(b_elsewhere))
    result = solve(p, PARAMS)

    assert result.lessons == step1.lessons
    assert result.day_formats == step1.day_formats


def _without_priority_areas(p: Problem) -> Problem:
    return replace(p, subjects=tuple(replace(s, priority_area=False) for s in p.subjects))


def _timeless(result: Result) -> Json:
    """The CLI result JSON without the wall-clock stats."""
    data = result_to_json(result)
    stats = cast("Json", data["stats"])
    data["stats"] = {k: v for k, v in stats.items() if k not in WALL_CLOCK_STATS}
    return data


def test_coverage_is_reproducible_on_a_deterministic_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Phase A alone leaves 3 of these 6 classes without an in-person day.
    p = _without_priority_areas(generate_problem(6, 0.55, seed=2))
    params = Params(
        time_limit_s=60, num_workers=2, max_deterministic_time=2.0, interleave_search=True
    )
    outcomes = _spy(monkeypatch)

    first, second = solve(p, params), solve(p, params)

    assert len(outcomes) == 2  # the coverage step ran in both solves
    assert _timeless(first) == _timeless(second)
    assert _zero_day(p, first) == []
