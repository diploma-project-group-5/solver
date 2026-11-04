"""Toolchain smoke tests: package installs, entry point runs, CP-SAT works on this platform."""

import re
from typing import TYPE_CHECKING

from ortools.sat.python import cp_model

import lad_solver
from lad_solver.__main__ import main

if TYPE_CHECKING:
    import pytest


def test_package_exposes_semver_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", lad_solver.__version__)


def test_entry_point_reports_version_and_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert lad_solver.__version__ in capsys.readouterr().out


def test_cp_sat_solves_a_tiny_model_to_optimality() -> None:
    # Two lessons, two slots, no clash; maximise the slot index of lesson ``a`` -> a=1, b=0.
    model = cp_model.CpModel()
    a = model.new_int_var(0, 1, "a")
    b = model.new_int_var(0, 1, "b")
    model.add(a != b)
    model.maximize(a)

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 0
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = 5.0

    assert solver.solve(model) == cp_model.OPTIMAL
    assert (solver.value(a), solver.value(b)) == (1, 0)
