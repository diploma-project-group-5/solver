"""Solver parameters reach every CpSolver run (review 11: memory cap)."""

import pytest

from builders import PRIO, klass, problem, row, teacher
from lad_solver.app.solve import solve
from lad_solver.domain.problem import Params
from lad_solver.model import cpsat
from lad_solver.model.cpsat import SolverConfig, new_solver


def test_new_solver_sets_the_memory_cap() -> None:
    assert (
        new_solver(SolverConfig(time_limit_s=1, max_memory_mb=256)).parameters.max_memory_in_mb
        == 256
    )


def test_new_solver_leaves_sigint_to_the_worker() -> None:
    # CP-SAT's own SIGINT handler would replace the worker's graceful stop (and outlive the solve).
    assert not new_solver(SolverConfig(time_limit_s=1)).parameters.catch_sigint_signal


def test_every_engine_run_gets_the_memory_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int | None] = []
    real = cpsat.new_solver

    def recording(config: SolverConfig) -> object:
        seen.append(config.max_memory_mb)
        return real(config)

    monkeypatch.setattr(cpsat, "new_solver", recording)
    p = problem(classes=(klass("a", 10),), teachers=(teacher("t"),), rows=(row("a", PRIO, "t", 3),))

    solve(p, Params(time_limit_s=5, num_workers=1, max_memory_mb=300))

    assert seen
    assert set(seen) == {300}
