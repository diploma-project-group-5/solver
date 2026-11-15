"""Engine telemetry in ``Stats`` for the worker's ``job.completed`` log: numbers and enums only.

The coverage step (ADR 2026-11-14) reports one of five verdicts, the status of its run and the
number of classes without an in-person day before / after it; phase B reports on how many days
the feasibility-only retry ran. No class or teacher ids: the engine has no logger of its own.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

import golden
from builders import OTHER, PRIO, klass, problem, row, teacher
from lad_solver.app import solve as solve_module
from lad_solver.app.solve import solve
from lad_solver.domain.problem import Params, Problem
from lad_solver.domain.result import CoverageVerdict, Result, SolveStatus, StopReason
from lad_solver.model import phase_a
from lad_solver.model.cpsat import RunInfo, SolverConfig
from lad_solver.model.index import Index
from lad_solver.model.phase_a import DayPlan, PhaseAOutcome
from lad_solver.model.phase_b import DayAssignment, DayDemand, assign_day
from lad_solver.model.progress import Progress
from lad_solver.model.stop import StopHandle

PARAMS = Params(time_limit_s=10, num_workers=1, random_seed=0)
FOUND = {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}


def _orphans() -> Problem:
    """A flagged class with priority lessons and three classes with neither (static orphans)."""
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
    )


def _two_days() -> Problem:
    """Every class in person on both days: nothing for the coverage step; phase B on two days."""
    return problem(
        classes=(klass("a", 10, priority=True), klass("b", 12)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 3), row("a", OTHER, "t2", 2), row("b", PRIO, "t1", 2)),
        days=(1, 2),
        lessons_per_day=3,
    )


def _summary(result: Result) -> tuple[object, ...]:
    s = result.stats
    return (s.coverage_verdict, s.coverage_status, s.zero_day_before, s.zero_day_after)


def _coverage_returns(
    monkeypatch: pytest.MonkeyPatch,
    status: SolveStatus,
    make: Callable[[Index, DayPlan], DayPlan] | None = None,
) -> None:
    """The coverage run ends with ``status`` and ``make(index, plan)`` (None: no plan)."""

    def fake(index: Index, _caps: object, plan: DayPlan, *_: object, **__: object) -> PhaseAOutcome:
        return PhaseAOutcome(RunInfo(status, None), None if make is None else make(index, plan))

    monkeypatch.setattr(solve_module, "solve_coverage", fake)


# ------------------------------------------------------------------ coverage step verdicts


def test_accepted_coverage_plan_reports_fewer_zero_day_classes() -> None:
    p = _orphans()

    result = solve(p, PARAMS)

    assert result.status in FOUND
    verdict, status, before, after = _summary(result)
    assert verdict is CoverageVerdict.ACCEPTED
    assert status in FOUND
    assert isinstance(before, int)
    assert isinstance(after, int)
    assert before > after
    assert after == len(golden.classes_without_in_person_days(p, result)) == 0


def test_no_zero_day_class_needs_no_coverage_step() -> None:
    result = solve(_two_days(), PARAMS)

    assert result.status in FOUND
    assert _summary(result) == (CoverageVerdict.NOT_NEEDED, None, 0, None)


def test_coverage_run_without_a_plan_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _coverage_returns(monkeypatch, SolveStatus.UNKNOWN)

    result = solve(_orphans(), PARAMS)

    verdict, status, before, after = _summary(result)
    assert (verdict, status, after) == (CoverageVerdict.KEPT, SolveStatus.UNKNOWN, None)
    assert isinstance(before, int)
    assert before > 0


def test_coverage_plan_below_the_floor_is_kept_with_both_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Everyone in person on day 1 only: no zero-day class left, but the objective drops.
    def day_one(index: Index, plan: DayPlan) -> DayPlan:
        return DayPlan(plan.loads, frozenset((c, 1) for c in range(len(index.students))))

    _coverage_returns(monkeypatch, SolveStatus.OPTIMAL, day_one)

    result = solve(_orphans(), PARAMS)

    verdict, status, before, after = _summary(result)
    assert (verdict, status, after) == (CoverageVerdict.KEPT, SolveStatus.OPTIMAL, 0)
    assert isinstance(before, int)
    assert before > 0


def test_coverage_stopped_while_running(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = StopHandle()
    real = phase_a.solve_coverage

    def stopped_at_end(*args: object, **kwargs: object) -> PhaseAOutcome:
        outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        handle.stop()  # SIGTERM as the run ends: its plan is not taken
        return outcome

    monkeypatch.setattr(solve_module, "solve_coverage", stopped_at_end)

    result = solve(_orphans(), PARAMS, handle)

    verdict, status, before, after = _summary(result)
    assert (verdict, after) == (CoverageVerdict.STOPPED, None)
    assert status in FOUND
    assert isinstance(before, int)
    assert before > 0


def test_coverage_not_started_once_the_job_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = StopHandle()
    real = phase_a.solve_phase_a

    def stopped_after(*args: object, **kwargs: object) -> PhaseAOutcome:
        outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        handle.stop()
        return outcome

    monkeypatch.setattr(solve_module, "solve_phase_a", stopped_after)

    result = solve(_orphans(), PARAMS, handle)

    assert _summary(result) == (CoverageVerdict.STOPPED, None, None, None)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_coverage_without_time_left(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    fake_time = SimpleNamespace(monotonic=clock)
    monkeypatch.setattr(solve_module, "time", fake_time)
    monkeypatch.setattr(phase_a, "time", fake_time)
    t = PARAMS.time_limit_s
    reserve_b = min(solve_module.PHASE_B_RESERVE_MAX_S, solve_module.PHASE_B_RESERVE_SHARE * t)
    deadline = clock.now + t - reserve_b  # the end of phase A
    real = phase_a.solve_phase_a
    ladder: list[PhaseAOutcome] = []

    def late(*args: object, **kwargs: object) -> PhaseAOutcome:
        if not ladder:  # one real plan, reported FEASIBLE for every ladder run
            outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
            ladder.append(replace(outcome, info=replace(outcome.info, status=SolveStatus.FEASIBLE)))
        clock.now = deadline  # each run used the whole phase A budget
        return ladder[0]

    monkeypatch.setattr(solve_module, "solve_phase_a", late)

    result = solve(_orphans(), PARAMS, progress=Progress(clock=clock))

    verdict, status, before, after = _summary(result)
    assert (verdict, status, after) == (CoverageVerdict.NO_TIME, None, None)
    assert isinstance(before, int)
    assert before > 0


def test_no_phase_a_plan_has_no_coverage_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    def nothing(*_args: object, **_kwargs: object) -> PhaseAOutcome:
        return PhaseAOutcome(RunInfo(SolveStatus.UNKNOWN, None), None)

    monkeypatch.setattr(solve_module, "solve_phase_a", nothing)

    result = solve(_orphans(), PARAMS)

    assert result.status is SolveStatus.UNKNOWN
    assert _summary(result) == (None, None, None, None)
    assert result.stats.phase_b_retry_days == 0


# ------------------------------------------------------------------ phase B retry days


def _phase_b_first_attempt(
    monkeypatch: pytest.MonkeyPatch, first: SolveStatus, retry: SolveStatus | None
) -> None:
    """The compact attempt ends with ``first``; the retry is real when ``retry`` is None."""

    def fake(  # noqa: PLR0913 -- mirrors assign_day
        day: int,
        lessons: Sequence[DayDemand],
        lessons_per_day: int,
        *,
        config: SolverConfig,
        stop: StopHandle | None = None,  # noqa: ARG001
        compact: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> DayAssignment:
        if compact:
            return DayAssignment(None, first)
        if retry is None:
            return assign_day(day, lessons, lessons_per_day, config=config, compact=False, **kwargs)
        return DayAssignment(None, retry)

    monkeypatch.setattr(solve_module, "assign_day", fake)


def test_phase_b_without_retries_reports_zero() -> None:
    result = solve(_two_days(), PARAMS)

    assert result.status in FOUND
    assert result.stats.phase_b_retry_days == 0


def test_phase_b_counts_the_days_that_needed_the_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _phase_b_first_attempt(monkeypatch, SolveStatus.UNKNOWN, None)
    p = _two_days()

    result = solve(p, PARAMS)

    assert result.status in FOUND
    assert golden.violations(p, result) == []
    assert result.stats.phase_b_retry_days == len(p.school_days) == 2


def test_phase_b_failure_keeps_the_retry_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _phase_b_first_attempt(monkeypatch, SolveStatus.UNKNOWN, SolveStatus.UNKNOWN)

    result = solve(_two_days(), PARAMS)

    assert result.status is SolveStatus.UNKNOWN
    assert result.stats.stop_reason is StopReason.TIME_LIMIT
    assert result.stats.phase_b_retry_days == 1  # the first day failed even with the retry


def test_phase_b_after_a_stop_retries_every_day(monkeypatch: pytest.MonkeyPatch) -> None:
    # A stop before phase B skips the compact model: each day gets the feasibility retry only.
    handle = StopHandle()
    real = phase_a.solve_phase_a

    def stopped_after(*args: object, **kwargs: object) -> PhaseAOutcome:
        outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        handle.stop()
        return outcome

    monkeypatch.setattr(solve_module, "solve_phase_a", stopped_after)
    p = _two_days()

    result = solve(p, PARAMS, handle)

    assert result.status in FOUND
    assert result.stats.stop_reason is StopReason.STOPPED
    assert result.stats.phase_b_retry_days == len(p.school_days)
