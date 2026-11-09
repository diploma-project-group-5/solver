"""External stop (SIGTERM path): the best grid comes back quickly with stop_reason=stopped."""

import threading
import time

import pytest

import oracle
from lad_solver.app import solve as solve_module
from lad_solver.app.solve import solve
from lad_solver.cli.generate import generate_problem
from lad_solver.domain.problem import Params
from lad_solver.domain.result import SolveStatus, StopReason
from lad_solver.model import phase_a
from lad_solver.model.cpsat import RunInfo
from lad_solver.model.phase_a import DayPlan, PhaseAOutcome
from lad_solver.model.stop import StopHandle


def test_stop_during_phase_a_returns_best_full_timetable() -> None:
    problem = generate_problem(8, 0.55, seed=1)
    handle = StopHandle()
    timer = threading.Timer(1.0, handle.stop)
    timer.start()
    started = time.monotonic()
    try:
        result = solve(problem, Params(time_limit_s=60, num_workers=2), handle)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 10  # far below the 60 s limit
    assert result.stats.stop_reason is StopReason.STOPPED
    assert result.status is SolveStatus.FEASIBLE
    assert oracle.violations(problem, result) == []


def test_time_limit_runs_probe_then_hinted_follow_up(monkeypatch: pytest.MonkeyPatch) -> None:
    real = phase_a.solve_phase_a
    calls: list[bool] = []

    def recording(*args: object, hint: DayPlan | None = None, **kwargs: object) -> PhaseAOutcome:
        calls.append(hint is not None)
        return real(*args, hint=hint, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(solve_module, "solve_phase_a", recording)
    problem = generate_problem(8, 0.55, seed=1)
    params = Params(time_limit_s=4, num_workers=2)

    result = solve(problem, params)

    assert calls == [False, True]  # the capped probe, then its hinted follow-up
    assert result.status is SolveStatus.FEASIBLE
    stats = result.stats
    assert stats.stop_reason is StopReason.TIME_LIMIT  # stall off: never no_improvement
    assert stats.first_solution_s is not None
    assert stats.first_solution_s < params.time_limit_s
    assert stats.last_improvement_s is not None
    assert stats.best_bound is not None
    assert stats.objective is not None
    assert stats.objective <= stats.best_bound
    # Phase B per-day retries are feasibility-only and not interruptible.
    assert (
        stats.wall_time_s
        < params.time_limit_s + solve_module.PHASE_B_RETRY_S * len(problem.school_days) + 1
    )
    assert oracle.violations(problem, result) == []


def test_stop_right_after_an_infeasible_capped_probe_is_not_a_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The capped probe proves only that the *capped* model is infeasible; the relaxed run that
    # could still find a grid never starts once the job is stopped.
    handle = StopHandle()
    calls: list[bool] = []

    def fake(*_args: object, hint: DayPlan | None = None, **_kwargs: object) -> PhaseAOutcome:
        calls.append(hint is not None)
        handle.stop()  # SIGTERM lands while the probe ends
        return PhaseAOutcome(RunInfo(SolveStatus.INFEASIBLE, None), None)

    monkeypatch.setattr(solve_module, "solve_phase_a", fake)

    result = solve(generate_problem(4, 0.5, seed=1), Params(time_limit_s=10, num_workers=1), handle)

    assert calls == [False]  # the probe only
    assert result.status is SolveStatus.UNKNOWN
    assert result.conflicts == ()  # no packing verdict
    assert result.stats.stop_reason is StopReason.STOPPED
