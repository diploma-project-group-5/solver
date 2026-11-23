"""Phase A progress and the no-improvement stop (ADR 2026-11-08), on a fake clock."""

import math

import pytest

from lad_solver.model.progress import Progress, StallPolicy


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class _Solver:
    def __init__(self) -> None:
        self.stops = 0

    def stop_search(self) -> None:
        self.stops += 1


def _armed(policy: StallPolicy | None = None) -> tuple[Progress, _Clock, _Solver]:
    clock = _Clock()
    progress = Progress(clock=clock)
    progress.begin(policy or StallPolicy(min_s=8.0, ratio=1.0))
    solver = _Solver()
    progress.attach(solver)
    return progress, clock, solver


def test_policy_window_is_the_larger_of_floor_and_time_to_last_improvement() -> None:
    policy = StallPolicy(min_s=8.0, ratio=1.0)
    assert not policy.due(now=107.9, started=100.0, last_improvement=100.5)
    assert policy.due(now=108.5, started=100.0, last_improvement=100.5)  # floor 8 s
    assert not policy.due(now=139.9, started=100.0, last_improvement=120.0)
    assert policy.due(now=140.0, started=100.0, last_improvement=120.0)  # 1.0 x 20 s


def test_policy_never_fires_before_the_first_solution() -> None:
    assert not StallPolicy(min_s=8.0, ratio=1.0).due(now=1e9, started=0.0, last_improvement=None)


@pytest.mark.parametrize(
    ("min_s", "ratio"), [(0.0, 1.0), (-1.0, 1.0), (8.0, 0.0), (8.0, -0.5), (math.nan, 1.0)]
)
def test_policy_rejects_non_positive_or_nan_values(min_s: float, ratio: float) -> None:
    with pytest.raises(ValueError, match="stall"):
        StallPolicy(min_s=min_s, ratio=ratio)


def test_check_stops_the_attached_solver_when_due() -> None:
    progress, clock, solver = _armed()
    progress.record(10)
    clock.now = 107.9
    assert not progress.check()
    clock.now = 108.0
    assert progress.check()
    assert progress.check()  # stays fired and repeats the stop (see the next-but-one test)
    assert solver.stops == 2
    assert progress.stalled


def test_only_a_strict_improvement_of_the_global_best_resets_the_window() -> None:
    progress, clock, solver = _armed()
    progress.record(10)
    clock.now = 105.0
    progress.record(10)  # same objective, e.g. the hinted follow-up's first solution
    progress.record(9)  # worse
    clock.now = 108.0
    assert progress.check()
    assert progress.last_improvement_at == 100.0
    assert solver.stops == 1


def test_improvement_moves_the_window() -> None:
    progress, clock, _ = _armed()
    progress.record(10)
    clock.now = 105.0
    progress.record(11)
    clock.now = 112.9
    assert not progress.check()
    clock.now = 113.0
    assert progress.check()


def test_without_a_policy_progress_only_records_telemetry() -> None:
    clock = _Clock()
    progress = Progress(clock=clock)
    progress.begin(None)
    solver = _Solver()
    progress.attach(solver)
    progress.record(5)
    clock.now = 1e6
    assert not progress.check()
    assert solver.stops == 0
    assert progress.first_solution_at == 100.0
    assert progress.last_improvement_at == 100.0


def test_check_after_phase_a_is_harmless() -> None:
    progress, clock, solver = _armed()
    progress.record(10)
    progress.detach()
    progress.finish()
    clock.now = 1e6
    assert not progress.check()
    assert solver.stops == 0


def test_solver_attached_after_firing_is_stopped_at_once() -> None:
    progress, clock, first = _armed()
    progress.record(10)
    clock.now = 200.0
    assert progress.check()
    progress.detach()
    second = _Solver()
    progress.attach(second)
    assert second.stops == 1
    assert first.stops == 1


def test_check_keeps_stopping_the_running_solver_on_every_tick() -> None:
    # CP-SAT ignores stop_search() before solve() has started; a later tick must retry.
    progress, clock, solver = _armed()
    progress.record(10)
    clock.now = 200.0
    assert progress.check()
    assert progress.check()
    assert solver.stops == 2


def test_finish_snapshots_the_outcome_of_phase_a() -> None:
    progress, clock, _ = _armed()
    progress.record(10)
    assert progress.finish() is False  # phase A ended (e.g. time limit) before the rule fired
    clock.now = 200.0
    assert not progress.check()  # a late tick cannot relabel it


def test_begin_resets_state_from_a_previous_phase() -> None:
    progress, clock, _ = _armed()
    progress.record(10)
    clock.now = 200.0
    assert progress.check()
    progress.finish()
    progress.begin(StallPolicy(min_s=8.0, ratio=1.0))
    assert not progress.stalled
    assert progress.first_solution_at is None
    assert progress.last_improvement_at is None
