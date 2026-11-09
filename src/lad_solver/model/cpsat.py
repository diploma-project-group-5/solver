"""Thin, typed wrapper around one CpSolver run: parameters, stop handle, progress."""

from dataclasses import dataclass, replace

from ortools.sat.python import cp_model

from lad_solver.domain.result import SolveStatus
from lad_solver.model.progress import Progress
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
    best_bound: float | None


class _Solutions(cp_model.CpSolverSolutionCallback):
    """Reports every solution of a phase A run (a model with an objective) to ``progress``."""

    def __init__(self, progress: Progress) -> None:
        super().__init__()
        self._progress = progress

    def on_solution_callback(self) -> None:
        self._progress.record(self.objective_value)


def new_solver(config: SolverConfig) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    params = solver.parameters
    params.max_time_in_seconds = max(config.time_limit_s, 0.01)
    params.num_workers = config.num_workers
    params.random_seed = config.random_seed
    params.log_search_progress = False
    params.log_to_stdout = False
    # CP-SAT's own SIGINT handler would abort the worker mid-solve and stay installed after it;
    # the worker owns SIGINT/SIGTERM and stops a solve through StopHandle.
    params.catch_sigint_signal = False
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
    progress: Progress | None = None,
) -> tuple[cp_model.CpSolver, RunInfo]:
    """Solve ``model``; an already requested stop returns UNKNOWN without searching.

    With ``progress`` (phase A, always a model with an objective), solutions are reported to it
    and its stall check may stop this solver; ``StopHandle`` stays reserved for the external stop.
    """
    if progress is not None and not has_objective:
        msg = "progress tracks the objective: a model without one cannot report to it"
        raise ValueError(msg)
    solver = new_solver(config)
    if (stop is not None and stop.stopped) or (progress is not None and progress.stalled):
        return solver, RunInfo(SolveStatus.UNKNOWN, None)
    callback = None if progress is None else _Solutions(progress)
    if stop is not None:
        stop.attach(solver)
    if progress is not None:
        progress.attach(solver)
    try:
        status = _STATUS[solver.solve(model, callback)]
    finally:
        if progress is not None:
            progress.detach()
        if stop is not None:
            stop.detach()
    found = status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    bound = solver.best_objective_bound if found and has_objective else None
    return solver, RunInfo(status, bound)
