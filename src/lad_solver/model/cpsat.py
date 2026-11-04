"""Thin, typed wrapper around one CpSolver run: parameters, stop handle, first-solution time."""

import time
from dataclasses import dataclass, replace

from ortools.sat.python import cp_model

from lad_solver.domain.result import SolveStatus
from lad_solver.model.stop import StopHandle

_STATUS = {
    cp_model.OPTIMAL: SolveStatus.OPTIMAL,
    cp_model.FEASIBLE: SolveStatus.FEASIBLE,
    cp_model.INFEASIBLE: SolveStatus.INFEASIBLE,
    cp_model.UNKNOWN: SolveStatus.UNKNOWN,
    cp_model.MODEL_INVALID: SolveStatus.MODEL_INVALID,
}


@dataclass(frozen=True, slots=True)
class SolverConfig:
    time_limit_s: float
    num_workers: int = 1
    random_seed: int = 0
    max_deterministic_time: float | None = None
    interleave_search: bool = False
    max_memory_mb: int | None = None  # CP-SAT stops gracefully (best solution) at this RSS

    def with_time(self, time_limit_s: float, deterministic: float | None = None) -> SolverConfig:
        return replace(self, time_limit_s=time_limit_s, max_deterministic_time=deterministic)


@dataclass(frozen=True, slots=True)
class RunInfo:
    status: SolveStatus
    wall_time_s: float
    first_solution_at: float | None  # time.monotonic() of the first solution
    best_bound: float | None


class _FirstSolution(cp_model.CpSolverSolutionCallback):
    def __init__(self) -> None:
        super().__init__()
        self.at: float | None = None

    def on_solution_callback(self) -> None:
        if self.at is None:
            self.at = time.monotonic()


def new_solver(config: SolverConfig) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    params = solver.parameters
    params.max_time_in_seconds = max(config.time_limit_s, 0.01)
    params.num_workers = config.num_workers
    params.random_seed = config.random_seed
    params.log_search_progress = False
    params.log_to_stdout = False
    if config.max_deterministic_time is not None:
        params.max_deterministic_time = config.max_deterministic_time
    if config.interleave_search:
        params.interleave_search = True
    if config.max_memory_mb is not None:
        params.max_memory_in_mb = config.max_memory_mb
    return solver


def run(
    model: cp_model.CpModel,
    config: SolverConfig,
    stop: StopHandle | None = None,
    *,
    has_objective: bool = True,
) -> tuple[cp_model.CpSolver, RunInfo]:
    """Solve ``model``; an already requested stop returns UNKNOWN without searching."""
    solver = new_solver(config)
    if stop is not None and stop.stopped:
        return solver, RunInfo(SolveStatus.UNKNOWN, 0.0, None, None)
    started = time.monotonic()
    callback = _FirstSolution()
    if stop is not None:
        stop.attach(solver)
    try:
        status = _STATUS[solver.solve(model, callback)]
    finally:
        if stop is not None:
            stop.detach()
    found = status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    bound = solver.best_objective_bound if found and has_objective else None
    return solver, RunInfo(status, time.monotonic() - started, callback.at, bound)
