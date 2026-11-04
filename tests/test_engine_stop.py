"""External stop (SIGTERM path): the best grid comes back quickly with stop_reason=stopped."""

import threading
import time

import oracle
from lad_solver.app.solve import solve
from lad_solver.cli.generate import generate_problem
from lad_solver.domain.problem import Params
from lad_solver.domain.result import SolveStatus, StopReason
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


def test_time_limit_runs_probe_then_hinted_follow_up() -> None:
    problem = generate_problem(8, 0.55, seed=1)
    result = solve(problem, Params(time_limit_s=4, num_workers=2))
    assert result.status is SolveStatus.FEASIBLE
    assert result.stats.stop_reason is StopReason.TIME_LIMIT
    assert result.stats.first_solution_s is not None
    assert result.stats.first_solution_s < 1
    assert result.stats.best_bound is not None
    assert result.stats.objective is not None
    assert result.stats.objective <= result.stats.best_bound
    assert result.stats.wall_time_s < 5
    assert oracle.violations(problem, result) == []
