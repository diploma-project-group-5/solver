"""Phase B failures and phase A follow-up anomalies are visible, never silent (review 12, 13)."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest
from ortools.sat.python import cp_model

import oracle
from builders import OTHER, PRIO, klass, problem, row, teacher
from lad_solver.app import solve as solve_module
from lad_solver.domain.problem import Params, Problem
from lad_solver.domain.result import SolveStatus, StopReason
from lad_solver.model import phase_a, phase_b
from lad_solver.model.cpsat import RunInfo, SolverConfig
from lad_solver.model.index import Index
from lad_solver.model.phase_a import DayPlan, PhaseAOutcome
from lad_solver.model.phase_b import DayAssignment, DayDemand, assign_day
from lad_solver.model.progress import Progress
from lad_solver.model.stop import StopHandle

PARAMS = Params(time_limit_s=10, num_workers=1, random_seed=0)


def _problem() -> Problem:
    return problem(
        classes=(klass("a", 10, priority=True), klass("b", 12)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 3), row("a", OTHER, "t2", 2), row("b", PRIO, "t1", 2)),
        days=(1, 2),
        lessons_per_day=3,
    )


type AssignCall = tuple[float, StopHandle | None, bool]


def _patch_phase_b(
    monkeypatch: pytest.MonkeyPatch, first: SolveStatus, retry: SolveStatus | None
) -> list[AssignCall]:
    """First (compact) attempt ends with ``first``; the retry is real when ``retry`` is None."""
    calls: list[AssignCall] = []

    def fake(  # noqa: PLR0913 -- mirrors assign_day
        day: int,
        lessons: Sequence[DayDemand],
        lessons_per_day: int,
        *,
        config: SolverConfig,
        stop: StopHandle | None = None,
        compact: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> DayAssignment:
        calls.append((config.time_limit_s, stop, compact))
        if compact:
            return DayAssignment(None, first)
        if retry is None:
            return assign_day(day, lessons, lessons_per_day, config=config, compact=False, **kwargs)
        return DayAssignment(None, retry)

    monkeypatch.setattr(solve_module, "assign_day", fake)
    return calls


def test_phase_b_timeout_retries_each_day_and_keeps_the_phase_a_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_phase_b(monkeypatch, SolveStatus.UNKNOWN, None)
    p = _problem()

    result = solve_module.solve(p, PARAMS, StopHandle())

    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(p, result) == []
    assert result.stats.objective == oracle.objective(p, result).value
    retries = [c for c in calls if not c[2]]
    assert retries == [(solve_module.PHASE_B_RETRY_S, None, False)] * 2  # not interruptible


def test_tiny_phase_b_budget_still_returns_a_full_grid() -> None:
    p = _problem()

    result = solve_module.solve(p, replace(PARAMS, phase_b_reserve_s=0.0))

    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(p, result) == []


def test_phase_b_that_never_finishes_is_unknown_without_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_phase_b(monkeypatch, SolveStatus.UNKNOWN, SolveStatus.UNKNOWN)

    result = solve_module.solve(_problem(), PARAMS)

    assert result.status is SolveStatus.UNKNOWN
    assert result.lessons == ()
    stats = result.stats
    assert (stats.objective, stats.best_bound, stats.primary, stats.secondary) == (None,) * 4
    assert stats.stop_reason is StopReason.TIME_LIMIT  # not phase A's "completed": no grid


def test_phase_b_day_runs_cp_sat_at_most_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    # Compact model, then one feasibility retry: never the compact model's own fallback on top,
    # so a day costs at most day_limit + PHASE_B_RETRY_S and the reserve holds.
    runs: list[bool] = []

    def unknown(*_args: object, has_objective: bool = True, **_kwargs: object) -> object:
        runs.append(has_objective)
        return cp_model.CpSolver(), RunInfo(SolveStatus.UNKNOWN, None)

    monkeypatch.setattr(phase_b, "run", unknown)

    result = solve_module.solve(_problem(), PARAMS)

    assert result.status is SolveStatus.UNKNOWN
    assert runs == [True, False]  # the first day gave up: compact, then the retry


def test_assign_day_alone_keeps_its_feasibility_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[bool] = []

    def unknown(*_args: object, has_objective: bool = True, **_kwargs: object) -> object:
        runs.append(has_objective)
        return cp_model.CpSolver(), RunInfo(SolveStatus.UNKNOWN, None)

    monkeypatch.setattr(phase_b, "run", unknown)
    demands = [DayDemand(0, "a", "t1", 1)]

    assigned = assign_day(1, demands, 2, config=SolverConfig(time_limit_s=1))

    assert assigned == DayAssignment(None, SolveStatus.UNKNOWN)
    assert runs == [True, False]


@pytest.mark.parametrize("failure", [SolveStatus.INFEASIBLE, SolveStatus.MODEL_INVALID])
def test_infeasible_phase_b_is_a_bug_reported_as_model_invalid(
    monkeypatch: pytest.MonkeyPatch, failure: SolveStatus
) -> None:
    calls = _patch_phase_b(monkeypatch, failure, None)

    result = solve_module.solve(_problem(), PARAMS)

    assert result.status is SolveStatus.MODEL_INVALID
    assert result.conflicts == ()  # no invalid_input: the worker reports error.code=internal
    assert (result.stats.objective, result.stats.best_bound) == (None, None)
    assert all(compact for _, _, compact in calls)  # a proven failure is not retried


# ---------------------------------------------------------------- 13. follow-up anomaly


def test_follow_up_model_invalid_keeps_the_probe_plan_and_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = phase_a.solve_phase_a
    runs: list[int] = []

    def fake(  # noqa: PLR0913 - mirrors solve_phase_a
        index: Index,
        caps: tuple[int, ...],
        config: SolverConfig,
        stop: StopHandle | None = None,
        *,
        hint: DayPlan | None = None,
        progress: Progress | None = None,
        deadline: float | None = None,
    ) -> PhaseAOutcome:
        runs.append(1)
        outcome = real(index, caps, config, stop, hint=hint, progress=progress, deadline=deadline)
        if len(runs) == 1:  # the probe: pretend it was not proven optimal
            return replace(outcome, info=replace(outcome.info, status=SolveStatus.FEASIBLE))
        return PhaseAOutcome(replace(outcome.info, status=SolveStatus.MODEL_INVALID), None)

    monkeypatch.setattr(solve_module, "solve_phase_a", fake)
    p = _problem()

    result = solve_module.solve(p, PARAMS)

    assert len(runs) == 2
    assert result.status is SolveStatus.FEASIBLE
    assert oracle.violations(p, result) == []
    assert result.stats.phase_a_status is not None
    assert "MODEL_INVALID" in result.stats.phase_a_status
