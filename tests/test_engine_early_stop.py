"""Early stop of phase A on no improvement, and the complete follow-up hint (ADR 2026-11-08)."""

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace

import pytest
from ortools.sat.python import cp_model

import oracle
from lad_solver.app import solve as solve_module
from lad_solver.app.solve import solve, stop_reason
from lad_solver.cli.generate import generate_problem
from lad_solver.domain.problem import Params
from lad_solver.domain.result import SolveStatus, StopReason
from lad_solver.model import phase_a
from lad_solver.model.cpsat import SolverConfig
from lad_solver.model.index import Index, build_index
from lad_solver.model.phase_a import DayPlan, PhaseAOutcome, build_phase_a, score, solve_phase_a
from lad_solver.model.progress import Progress, StallPolicy
from lad_solver.model.stop import StopHandle


@contextmanager
def _ticking(progress: Progress, every_s: float = 0.05) -> Generator[None]:
    """What the worker's main loop does while it waits for the solve."""
    done = threading.Event()

    def loop() -> None:
        while not done.wait(every_s):
            progress.check()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _FirstObjective(cp_model.CpSolverSolutionCallback):
    def __init__(self) -> None:
        super().__init__()
        self.first: float | None = None

    def on_solution_callback(self) -> None:
        if self.first is None:
            self.first = self.objective_value
            self.stop_search()  # only the first objective matters


def test_follow_up_with_the_probe_hint_starts_at_the_probe_objective() -> None:
    index = build_index(generate_problem(10, 0.55, seed=3))
    caps = index.row_caps(2, relaxed=False)
    probe = solve_phase_a(index, caps, SolverConfig(time_limit_s=1, num_workers=1))
    assert probe.plan is not None
    model = build_phase_a(index, caps, hint=probe.plan).model
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = 5
    first = _FirstObjective()
    assert solver.solve(model, first) in {cp_model.OPTIMAL, cp_model.FEASIBLE}
    assert first.first is not None
    assert first.first >= score(index, probe.plan).objective


def test_every_phase_a_variable_is_hinted_on_follow_up() -> None:
    index = build_index(generate_problem(6, 0.5, seed=1))
    caps = index.row_caps(2, relaxed=False)
    probe = solve_phase_a(index, caps, SolverConfig(time_limit_s=1, num_workers=1))
    assert probe.plan is not None
    proto = build_phase_a(index, caps, hint=probe.plan).model.proto
    hinted = set(proto.solution_hint.vars)
    assert len(hinted) == len(proto.variables)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


@pytest.mark.slow
def test_stall_stops_a_plateaued_school_long_before_the_limit() -> None:
    problem = generate_problem(8, 0.55, seed=1)
    params = Params(time_limit_s=60, num_workers=2, stall_min_s=2.0, stall_ratio=1.0)
    progress = Progress()
    handle = StopHandle()
    started = time.monotonic()
    with _ticking(progress):
        result = solve(problem, params, handle, progress=progress)
    elapsed = time.monotonic() - started

    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(problem, result) == []
    assert not handle.stopped  # the stall never goes through the SIGTERM handle
    if result.status is SolveStatus.FEASIBLE:
        assert result.stats.stop_reason is StopReason.NO_IMPROVEMENT
        assert elapsed < 30  # far below the 60 s limit
    stats = result.stats
    assert stats.first_solution_s is not None
    assert stats.last_improvement_s is not None
    assert stats.first_solution_s <= stats.last_improvement_s <= stats.wall_time_s


def test_sigterm_wins_over_a_configured_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    real = phase_a.solve_phase_a
    clock = _Clock()
    handle = StopHandle()

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
        outcome = real(index, caps, config, stop, hint=hint, progress=progress, deadline=deadline)
        assert progress is not None
        clock.now += 100.0
        assert progress.check()  # the stall rule has fired ...
        handle.stop()  # ... and SIGTERM arrives as well
        return outcome

    monkeypatch.setattr(solve_module, "solve_phase_a", fake)
    params = Params(time_limit_s=4, num_workers=1, stall_min_s=5.0)

    result = solve(generate_problem(6, 0.5, seed=1), params, handle, progress=Progress(clock=clock))

    assert result.stats.stop_reason is StopReason.STOPPED


@pytest.mark.parametrize(
    ("status", "stopped", "stalled", "expected"),
    [
        (SolveStatus.OPTIMAL, False, True, StopReason.COMPLETED),
        (SolveStatus.INFEASIBLE, False, True, StopReason.COMPLETED),
        (SolveStatus.FEASIBLE, False, True, StopReason.NO_IMPROVEMENT),
        (SolveStatus.FEASIBLE, False, False, StopReason.TIME_LIMIT),
        (SolveStatus.OPTIMAL, True, True, StopReason.STOPPED),
        (SolveStatus.UNKNOWN, False, False, StopReason.TIME_LIMIT),
    ],
)
def test_stop_reason_precedence(
    status: SolveStatus, *, stopped: bool, stalled: bool, expected: StopReason
) -> None:
    assert stop_reason(status, stopped=stopped, stalled=stalled) is expected


def test_configured_stall_requires_a_ticked_progress() -> None:
    params = Params(time_limit_s=10, stall_min_s=8.0)
    with pytest.raises(ValueError, match="progress"):
        solve(generate_problem(3, 0.5, seed=1), params)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stall_min_s": 8.0, "max_deterministic_time": 5.0},
        {"stall_min_s": 0.0},
        {"stall_min_s": 8.0, "stall_ratio": 0.0},
        {"stall_min_s": float("nan")},
    ],
)
def test_params_reject_invalid_stall_settings(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="stall"):
        Params(time_limit_s=10, **kwargs)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"time_limit_s": float("nan")}, "time_limit_s"),
        ({"time_limit_s": float("inf")}, "time_limit_s"),
        ({"time_limit_s": 0.0}, "time_limit_s"),
        ({"time_limit_s": -1.0}, "time_limit_s"),
        ({"time_limit_s": 10.0, "phase_b_reserve_s": float("nan")}, "phase_b_reserve_s"),
        ({"time_limit_s": 10.0, "phase_b_reserve_s": -1.0}, "phase_b_reserve_s"),
        ({"time_limit_s": 10.0, "max_deterministic_time": float("inf")}, "max_deterministic_time"),
        ({"time_limit_s": 10.0, "max_deterministic_time": -0.5}, "max_deterministic_time"),
    ],
)
def test_params_reject_invalid_time_settings(kwargs: dict[str, float], name: str) -> None:
    # A NaN limit would otherwise reach CP-SAT and come back as MODEL_INVALID (an "engine bug").
    with pytest.raises(ValueError, match=name):
        Params(**kwargs)  # pyright: ignore[reportArgumentType]


def test_params_accept_a_zero_phase_b_reserve() -> None:
    assert Params(time_limit_s=10, phase_b_reserve_s=0.0).phase_b_reserve_s == 0.0


def test_a_phase_a_run_after_the_stall_fired_does_not_search() -> None:
    from lad_solver.model.cpsat import run  # noqa: PLC0415

    model = cp_model.CpModel()
    x = model.new_int_var(0, 10**6, "x")
    model.maximize(x)
    clock = _Clock()
    progress = Progress(clock=clock)
    progress.begin(StallPolicy(min_s=1.0, ratio=1.0))
    progress.record(1)
    clock.now += 5.0
    assert progress.check()

    started = time.monotonic()
    _, info = run(model, SolverConfig(time_limit_s=30, num_workers=1), progress=progress)
    assert time.monotonic() - started < 1
    assert info.status is SolveStatus.UNKNOWN


def test_hint_fixed_as_equalities_keeps_the_model_feasible() -> None:
    index = build_index(generate_problem(10, 0.55, seed=3))
    caps = index.row_caps(2, relaxed=False)
    probe = solve_phase_a(index, caps, SolverConfig(time_limit_s=1, num_workers=1))
    assert probe.plan is not None
    model = build_phase_a(index, caps, hint=probe.plan).model
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.fix_variables_to_their_hinted_value = True
    assert solver.solve(model) in {cp_model.OPTIMAL, cp_model.FEASIBLE}
    assert solver.objective_value == score(index, probe.plan).objective


def test_probe_stopped_by_the_stall_rule_skips_the_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = phase_a.solve_phase_a
    clock = _Clock()
    calls: list[bool] = []

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
        calls.append(hint is not None)
        outcome = real(index, caps, config, stop, hint=hint, progress=progress, deadline=deadline)
        assert progress is not None
        clock.now += 100.0  # a long silence after the probe's solutions
        assert progress.check()  # the rule fires during the probe
        return replace(outcome, info=replace(outcome.info, status=SolveStatus.FEASIBLE))

    monkeypatch.setattr(solve_module, "solve_phase_a", fake)
    problem = generate_problem(6, 0.5, seed=1)
    progress = Progress(clock=clock)

    result = solve(
        problem, Params(time_limit_s=4, num_workers=1, stall_min_s=5.0), progress=progress
    )

    assert calls == [False]  # the probe only, no hinted follow-up
    assert result.stats.stop_reason is StopReason.NO_IMPROVEMENT
    assert oracle.violations(problem, result) == []


def test_default_params_stay_reproducible_on_a_deterministic_budget() -> None:
    problem = generate_problem(8, 0.55, seed=2)
    params = Params(
        time_limit_s=60, num_workers=2, max_deterministic_time=0.5, interleave_search=True
    )
    first, second = solve(problem, params), solve(problem, params)
    assert first.lessons == second.lessons
    assert first.day_formats == second.day_formats
    assert first.stats.objective == second.stats.objective
