"""Coverage step after phase A (ADR 2026-11-14): every class gets an in-person day when it can.

Phase A rewards an in-person day only through priority lessons or the class flag, so the format
of a class with neither (an "orphan") is left to chance. The coverage step maximises the number
of classes with at least one in-person day while keeping the phase A objective at least at the
value of the plan it starts from: a class the shelter cannot take without lowering the objective
stays remote (a tie-break, never a hard minimum). Every solved result passes the validator-lib CLI.
"""

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

import golden
from builders import OTHER, PRIO, klass, problem, row, teacher
from lad_solver.app import solve as solve_module
from lad_solver.app.solve import solve
from lad_solver.cli.generate import generate_problem
from lad_solver.cli.io import Json, result_to_json
from lad_solver.domain.problem import Params, Passport, Problem
from lad_solver.domain.result import Result, SolveStatus, StopReason
from lad_solver.model import phase_a
from lad_solver.model.cpsat import RunInfo, SolverConfig
from lad_solver.model.index import Index, build_index
from lad_solver.model.phase_a import DayPlan, PhaseAOutcome, static_orphans, zero_day_classes
from lad_solver.model.progress import Progress
from lad_solver.model.stop import StopHandle

PARAMS = Params(time_limit_s=10, num_workers=1, random_seed=0)
FOUND = {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
WALL_CLOCK_STATS = frozenset({"wall_time_s", "first_solution_s", "last_improvement_s"})


def _zero_day(p: Problem, result: Result) -> list[str]:
    """Classes without an in-person day, from validator-lib, once the result is admissible."""
    assert result.status in FOUND, result.status
    assert golden.violations(p, result) == []
    return sorted(golden.classes_without_in_person_days(p, result))


def _spy(
    monkeypatch: pytest.MonkeyPatch, after: Callable[[PhaseAOutcome], None] | None = None
) -> list[PhaseAOutcome]:
    """Record every coverage run; ``after`` runs once a run has returned."""
    real = phase_a.solve_coverage
    outcomes: list[PhaseAOutcome] = []

    def spy(*args: object, **kwargs: object) -> PhaseAOutcome:
        outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        outcomes.append(outcome)
        if after is not None:
            after(outcome)
        return outcome

    monkeypatch.setattr(solve_module, "solve_coverage", spy)
    return outcomes


def _must_not_run(*_args: object, **_kwargs: object) -> PhaseAOutcome:
    msg = "the coverage step must not run here"
    raise AssertionError(msg)


def _no_classes(*_args: object) -> list[str]:
    return []


def _step1(monkeypatch: pytest.MonkeyPatch, p: Problem, params: Params = PARAMS) -> Result:
    """Phase A alone: the ladder with the same budget, its reserve unused, no coverage step.

    The baseline a kept plan is compared with.
    """
    with monkeypatch.context() as m:
        m.setattr(solve_module, "zero_day_classes", _no_classes)
        return solve(p, params)


def _orphans() -> Problem:
    """A flagged class with priority lessons and three classes with neither (W = 1 * 2 + 1)."""
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
    )  # the default Passport(10_000) holds everyone


ORPHANS_OBJECTIVE = 3 * 40 + 2  # a in person on both days with one priority lesson each


def _exact_capacity() -> Problem:
    """Exactly staff + the flagged class + its teacher; b alone fits, but only on a's day."""
    return problem(
        classes=(klass("a", 20, priority=True), klass("b", 15)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 2), row("b", OTHER, "t2", 2)),
        capacity=Passport(2 + 20 + 1),
        days=(1, 2),
        staff=2,
    )


def test_coverage_gives_every_orphan_an_in_person_day() -> None:
    p = _orphans()
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.OPTIMAL
    assert result.stats.stop_reason is StopReason.COMPLETED
    assert _zero_day(p, result) == []
    assert result.stats.objective == golden.objective(p, result).value == ORPHANS_OBJECTIVE


def test_coverage_spreads_days_under_tight_capacity() -> None:
    # Room for the staff and one class with its teacher a day: each class gets one of the days.
    p = problem(
        classes=(klass("a", 20), klass("b", 25)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", OTHER, "t1", 2), row("b", OTHER, "t2", 2)),
        capacity=Passport(2 + 25 + 1),
        days=(1, 2),
        staff=2,
    )
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.OPTIMAL
    assert _zero_day(p, result) == []
    assert Counter(c for c, _ in golden.in_person_days(result)) == {"a": 1, "b": 1}


def test_coverage_soft_when_capacity_exact() -> None:
    # Taking b in person costs a one in-person day (121 < 122): the objective wins, b stays
    # remote and the solve stays OPTIMAL. A hard minimum would have made this INFEASIBLE.
    p = _exact_capacity()
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.OPTIMAL
    assert result.stats.stop_reason is StopReason.COMPLETED
    assert result.stats.objective == 3 * 40 + 2
    assert _zero_day(p, result) == ["b"]


def test_coverage_skipped_without_zero_day_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every class has priority lessons and the shelter takes both (no zero-day class): the ladder
    # still ends a reserve early (known only after it), and the step does not run.
    clock = _freeze(monkeypatch)
    configs, _ = _ladder_spy(monkeypatch)
    monkeypatch.setattr(solve_module, "solve_coverage", _must_not_run)
    p = problem(
        classes=(klass("a", 20, priority=True), klass("b", 21)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 1), row("b", PRIO, "t2", 1)),
        days=(1,),
    )
    assert static_orphans(build_index(p)) == []
    result = solve(p, PARAMS, progress=Progress(clock=clock))
    assert result.status is SolveStatus.OPTIMAL
    assert _zero_day(p, result) == []
    ladder = _phase_a_budget(PARAMS) - _reserve(PARAMS)  # the probe gets a share of the ladder
    assert configs[0].time_limit_s == pytest.approx(solve_module.PROBE_SHARE * ladder)


def test_static_orphans_are_the_classes_no_objective_term_rewards() -> None:
    p = problem(
        classes=(klass("a", 20, priority=True), klass("b", 25), klass("c", 22), klass("d", 0)),
        teachers=(teacher("t1"), teacher("t2"), teacher("t3"), teacher("t4")),
        rows=(
            row("a", OTHER, "t1", 1),  # flagged: the secondary objective rewards it
            row("b", PRIO, "t2", 1),  # priority lessons: the primary objective rewards it
            row("c", OTHER, "t3", 1),  # neither
            row("d", PRIO, "t4", 1),  # priority lessons of no pupils earn nothing
        ),
    )
    assert static_orphans(build_index(p)) == ["c", "d"]


def test_coverage_stopped_keeps_step1_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _orphans()
    step1 = _step1(monkeypatch, p)
    handle = StopHandle()
    outcomes = _spy(monkeypatch, after=lambda _outcome: handle.stop())  # SIGTERM as it ends

    result = solve(p, PARAMS, handle)

    better = [o.plan for o in outcomes]
    assert len(better) == 1
    assert better[0] is not None
    assert zero_day_classes(build_index(p), better[0]) == []  # the step had a better plan ...
    assert _zero_day(p, step1) != []
    assert result.day_formats == step1.day_formats  # ... a stopped job keeps the phase A one
    assert result.stats.stop_reason is StopReason.STOPPED
    assert result.stats.objective == ORPHANS_OBJECTIVE


def test_coverage_not_started_once_the_job_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _orphans()
    step1 = _step1(monkeypatch, p)
    handle = StopHandle()
    real = phase_a.solve_phase_a

    def stopped_after(*args: object, **kwargs: object) -> PhaseAOutcome:
        outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        handle.stop()  # SIGTERM lands as the phase A run ends
        return outcome

    monkeypatch.setattr(solve_module, "solve_phase_a", stopped_after)
    monkeypatch.setattr(solve_module, "solve_coverage", _must_not_run)

    result = solve(p, PARAMS, handle)

    assert result.stats.stop_reason is StopReason.STOPPED
    assert _zero_day(p, step1) != []  # there was work for the step, but the job is stopped
    assert _zero_day(p, result) == _zero_day(p, step1)
    assert result.day_formats == step1.day_formats


def test_coverage_does_not_touch_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = Progress()
    marks: list[tuple[float | None, float | None]] = []
    real = phase_a.solve_coverage

    def spy(*args: object, **kwargs: object) -> PhaseAOutcome:
        marks.append((progress.first_solution_at, progress.last_improvement_at))
        outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        marks.append((progress.first_solution_at, progress.last_improvement_at))
        assert outcome.info.status is SolveStatus.OPTIMAL
        return outcome

    monkeypatch.setattr(solve_module, "solve_coverage", spy)
    p = _orphans()

    result = solve(p, PARAMS, progress=progress)

    before, after = marks
    assert before[1] is not None  # phase A recorded its improvements ...
    assert after == before  # ... the coverage objective (another scale) reported none
    assert _zero_day(p, result) == []


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_coverage_runs_after_the_stall_rule_fired(monkeypatch: pytest.MonkeyPatch) -> None:
    # A run that reported to the stalled progress would return UNKNOWN at once (ADR 2026-11-08).
    real = phase_a.solve_phase_a
    clock = _Clock()

    def stalled(*args: object, progress: Progress | None = None, **kwargs: object) -> PhaseAOutcome:
        outcome = real(*args, progress=progress, **kwargs)  # pyright: ignore[reportArgumentType]
        assert progress is not None
        clock.now += 100.0  # a long silence after the probe's solutions
        assert progress.check()  # the no-improvement rule fires during the probe
        return replace(outcome, info=replace(outcome.info, status=SolveStatus.FEASIBLE))

    monkeypatch.setattr(solve_module, "solve_phase_a", stalled)
    outcomes = _spy(monkeypatch)
    p = _orphans()
    params = Params(time_limit_s=10, num_workers=1, random_seed=0, stall_min_s=5.0)

    result = solve(p, params, progress=Progress(clock=clock))

    assert [o.info.status for o in outcomes] == [SolveStatus.OPTIMAL]
    assert result.status is SolveStatus.FEASIBLE  # status, stop reason, bound stay phase A's
    assert result.stats.stop_reason is StopReason.NO_IMPROVEMENT
    # The probe proved its optimum (bound V*); the coverage bound counts classes, another scale.
    assert outcomes[0].info.best_bound == len(p.classes)
    assert result.stats.best_bound == ORPHANS_OBJECTIVE
    assert result.stats.objective == ORPHANS_OBJECTIVE
    assert _zero_day(p, result) == []


def _capped_infeasible() -> Problem:
    """t2 comes on day 2 only and fills it, so a's three priority lessons all fall on day 1.

    That is over the row-day cap (2): only the relaxed run finds a timetable. b is an orphan.
    """
    return problem(
        classes=(klass("a", 20), klass("b", 25)),
        teachers=(teacher("t1"), teacher("t2", unavailable=(1,)), teacher("t3")),
        rows=(row("a", PRIO, "t1", 3), row("a", OTHER, "t2", 3), row("b", OTHER, "t3", 2)),
        days=(1, 2),
        lessons_per_day=3,
    )


def test_coverage_uses_relaxed_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = _spy(monkeypatch)
    p = _capped_infeasible()

    result = solve(p, PARAMS)

    assert result.stats.quality_relaxed
    # Under the capped caps the relaxed plan is no solution: the floor run would be INFEASIBLE.
    assert [o.info.status for o in outcomes] == [SolveStatus.OPTIMAL]
    assert result.status is SolveStatus.OPTIMAL
    assert result.stats.objective == 3 * 20  # W = 1: no flagged class
    assert _zero_day(p, result) == []


def _fake_coverage(
    make: Callable[[Index, DayPlan], DayPlan],
) -> Callable[..., PhaseAOutcome]:
    """A coverage run that "finds" ``make(index, plan)`` as OPTIMAL."""

    def fake(index: Index, _caps: object, plan: DayPlan, *_: object, **__: object) -> PhaseAOutcome:
        return PhaseAOutcome(RunInfo(SolveStatus.OPTIMAL, None), make(index, plan))

    return fake


def test_coverage_plan_below_the_floor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Everyone in person on day 1 only: no zero-day class left, but a loses day 2 (61 < 122).
    def day_one(index: Index, plan: DayPlan) -> DayPlan:
        return DayPlan(plan.loads, frozenset((c, 1) for c in range(len(index.students))))

    p = _orphans()
    step1 = _step1(monkeypatch, p)
    monkeypatch.setattr(solve_module, "solve_coverage", _fake_coverage(day_one))

    result = solve(p, PARAMS)

    assert _zero_day(p, step1) != []  # the fake plan has fewer zero-day classes ...
    assert result.lessons == step1.lessons  # ... but a lower objective: step 1 is kept
    assert result.day_formats == step1.day_formats
    assert result.stats.objective == ORPHANS_OBJECTIVE


def test_coverage_plan_without_fewer_zero_day_classes_keeps_step1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _exact_capacity()
    step1 = _step1(monkeypatch, p)

    def b_elsewhere(_index: Index, plan: DayPlan) -> DayPlan:
        # The same objective and the same zero-day class (b): only b's lessons sit elsewhere.
        b_row = 1
        moved = {(b_row, 1): 2} if plan.loads.get((b_row, 1)) != 2 else {(b_row, 2): 2}
        loads = {k: n for k, n in plan.loads.items() if k[0] != b_row} | moved
        return DayPlan(loads, plan.in_person)

    monkeypatch.setattr(solve_module, "solve_coverage", _fake_coverage(b_elsewhere))
    result = solve(p, PARAMS)

    assert result.lessons == step1.lessons
    assert result.day_formats == step1.day_formats


def _without_priority_areas(p: Problem) -> Problem:
    return replace(p, subjects=tuple(replace(s, priority_area=False) for s in p.subjects))


def _timeless(result: Result) -> Json:
    """The CLI result JSON without the wall-clock stats."""
    data = result_to_json(result)
    stats = cast("Json", data["stats"])
    data["stats"] = {k: v for k, v in stats.items() if k not in WALL_CLOCK_STATS}
    return data


def test_coverage_is_reproducible_on_a_deterministic_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Phase A alone leaves 3 of these 6 classes without an in-person day.
    p = _without_priority_areas(generate_problem(6, 0.55, seed=2))
    params = Params(
        time_limit_s=60, num_workers=2, max_deterministic_time=2.0, interleave_search=True
    )
    outcomes = _spy(monkeypatch)

    first, second = solve(p, params), solve(p, params)

    assert len(outcomes) == 2  # the coverage step ran in both solves
    assert _timeless(first) == _timeless(second)
    assert _zero_day(p, first) == []


def _spans(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Wall-clock span of every coverage run, from its call to its deadline."""
    real = phase_a.solve_coverage
    spans: list[float] = []

    def spy(*args: object, deadline: float, **kwargs: object) -> PhaseAOutcome:
        spans.append(deadline - time.monotonic())
        return real(*args, deadline=deadline, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(solve_module, "solve_coverage", spy)
    return spans


def _phase_a_budget(params: Params) -> float:
    """Seconds of phase A: the time limit less the phase B reserve (no override in these tests)."""
    t = params.time_limit_s
    return t - min(solve_module.PHASE_B_RESERVE_MAX_S, solve_module.PHASE_B_RESERVE_SHARE * t)


def test_coverage_step_is_capped_by_its_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    # On a tight shelter A2 cannot prove its optimum; uncapped it ran to the phase A deadline.
    spans = _spans(monkeypatch)
    p = _orphans()

    result = solve(p, Params(time_limit_s=600, num_workers=1, random_seed=0))

    assert len(spans) == 1
    assert spans[0] <= solve_module.COVERAGE_MAX_S
    assert _zero_day(p, result) == []


def test_coverage_step_is_capped_by_its_share_of_phase_a(monkeypatch: pytest.MonkeyPatch) -> None:
    # 10 s: phase A gets 8.5 s, and 0.15 of it (1.275 s) is below the 10 s cap.
    spans = _spans(monkeypatch)
    p = _orphans()

    result = solve(p, PARAMS)

    share = solve_module.COVERAGE_SHARE * _phase_a_budget(PARAMS)
    assert share < solve_module.COVERAGE_MAX_S
    assert len(spans) == 1
    assert 0 < spans[0] <= share + 1e-6
    assert _zero_day(p, result) == []


def test_coverage_on_a_deterministic_budget_ignores_the_wall_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The step is bounded by 0.15 of max_deterministic_time; the wall clock is only the safety
    # net, so a wall-clock cap far below MIN_RUN_S neither skips the step nor changes the result.
    p = _without_priority_areas(generate_problem(6, 0.55, seed=2))
    params = Params(
        time_limit_s=60, num_workers=2, max_deterministic_time=2.0, interleave_search=True
    )
    outcomes = _spy(monkeypatch)
    usual = solve(p, params)
    monkeypatch.setattr(solve_module, "COVERAGE_MAX_S", 0.001)

    capped = solve(p, params)

    assert len(outcomes) == 2  # the coverage step ran in both solves
    assert _timeless(capped) == _timeless(usual)
    assert _zero_day(p, capped) == []


def _freeze(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """One fake clock for the use case and phase A: every budget becomes an exact number."""
    clock = _Clock()
    fake = SimpleNamespace(monotonic=clock)
    monkeypatch.setattr(solve_module, "time", fake)
    monkeypatch.setattr(phase_a, "time", fake)
    return clock


def _shelter_for_one() -> Problem:
    """Both classes earn the primary term (no static orphan), but the shelter takes only one."""
    return problem(
        classes=(klass("a", 20), klass("b", 25)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 1), row("b", PRIO, "t2", 1)),
        capacity=Passport(2 + 25 + 1),
        days=(1,),
        staff=2,
    )


def _reserve(params: Params) -> float:
    """Wall-clock reserve of the coverage step: min(10 s, 0.15 of the phase A budget)."""
    budget = _phase_a_budget(params)
    return min(solve_module.COVERAGE_MAX_S, solve_module.COVERAGE_SHARE * budget)


def _ladder_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[SolverConfig], list[float | None]]:
    """Config and ``deadline`` of every ladder run (the probe passes no deadline)."""
    real = phase_a.solve_phase_a
    configs: list[SolverConfig] = []
    deadlines: list[float | None] = []

    def spy(
        index: Index, caps: tuple[int, ...], config: SolverConfig, *args: object, **kwargs: object
    ) -> PhaseAOutcome:
        configs.append(config)
        deadlines.append(cast("float | None", kwargs.get("deadline")))
        return real(index, caps, config, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(solve_module, "solve_phase_a", spy)
    return configs, deadlines


def _coverage_configs(monkeypatch: pytest.MonkeyPatch) -> list[SolverConfig]:
    """The config every coverage run was called with (positional argument 4)."""
    real = phase_a.solve_coverage
    configs: list[SolverConfig] = []

    def spy(*args: object, **kwargs: object) -> PhaseAOutcome:
        configs.append(cast("SolverConfig", args[4]))
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(solve_module, "solve_coverage", spy)
    return configs


@pytest.mark.parametrize("det", [None, 2.0], ids=["wall-clock", "deterministic"])
@pytest.mark.parametrize("orphans", [False, True], ids=["no-static-orphans", "static-orphans"])
def test_ladder_always_leaves_the_coverage_reserve(
    monkeypatch: pytest.MonkeyPatch, *, det: float | None, orphans: bool
) -> None:
    # Classes the objective rewards can still be left without an in-person day by the plan the
    # ladder finds ("dynamic orphans", _shelter_for_one): the reserve must not depend on that.
    clock = _freeze(monkeypatch)
    start = clock.now
    configs, deadlines = _ladder_spy(monkeypatch)
    coverage = _coverage_configs(monkeypatch)
    p = _orphans() if orphans else _shelter_for_one()
    params = replace(PARAMS, max_deterministic_time=det)

    result = solve(p, params, progress=Progress(clock=clock))

    assert bool(static_orphans(build_index(p))) is orphans
    assert len(coverage) == 1  # a class without an in-person day: the step ran in every case
    budget, reserve = _phase_a_budget(params), _reserve(params)
    share, probe_share = solve_module.COVERAGE_SHARE, solve_module.PROBE_SHARE
    probe, follow_ups, step = configs[0], configs[1:], coverage[0]
    if det is None:  # wall clock: the reserve comes off the ladder's deadline
        ladder = budget - reserve
        assert probe.time_limit_s == pytest.approx(probe_share * ladder)
        assert probe.max_deterministic_time is None
        assert step.time_limit_s == pytest.approx(reserve)  # the clock is frozen
        assert step.max_deterministic_time is None
    else:  # det splits the work; the wall clock is the same safety net for ladder and step
        ladder = budget
        ladder_det = det * (1 - share)
        assert probe.time_limit_s == pytest.approx(budget)
        assert probe.max_deterministic_time == pytest.approx(probe_share * ladder_det)
        for c in follow_ups:
            assert c.max_deterministic_time == pytest.approx((1 - probe_share) * ladder_det)
        assert step.max_deterministic_time == pytest.approx(share * det)
        planned = ladder_det + cast("float", step.max_deterministic_time)
        assert planned <= det + 1e-9  # phase A never plans more work than it was given
    assert all(d == pytest.approx(start + ladder) for d in deadlines if d is not None)
    assert result.status in FOUND


def test_coverage_runs_after_a_ladder_that_used_its_whole_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The large lyceum in miniature: no static orphan, the ladder runs to its deadline and a
    # class is still without an in-person day. The step must still get its reserve.
    clock = _freeze(monkeypatch)
    start = clock.now
    real = phase_a.solve_phase_a
    ladder: list[PhaseAOutcome] = []

    def to_its_deadline(
        index: Index, caps: tuple[int, ...], config: SolverConfig, *args: object, **kwargs: object
    ) -> PhaseAOutcome:
        if not ladder:  # one real plan, reported FEASIBLE so that the follow-up runs too
            outcome = real(index, caps, config, *args, **kwargs)  # pyright: ignore[reportArgumentType]
            ladder.append(replace(outcome, info=replace(outcome.info, status=SolveStatus.FEASIBLE)))
        deadline = cast("float | None", kwargs.get("deadline"))
        clock.now = clock.now + config.time_limit_s if deadline is None else deadline
        return ladder[0]

    monkeypatch.setattr(solve_module, "solve_phase_a", to_its_deadline)
    coverage = _coverage_configs(monkeypatch)
    p = _shelter_for_one()

    result = solve(p, PARAMS, progress=Progress(clock=clock))

    assert static_orphans(build_index(p)) == []
    assert len(ladder) == 1
    assert len(coverage) == 1  # the step ran although the ladder used all of its time ...
    assert clock.now == pytest.approx(start + _phase_a_budget(PARAMS) - _reserve(PARAMS))
    # ... because the ladder's time ended a whole reserve before the phase A deadline.
    assert coverage[0].time_limit_s == pytest.approx(_reserve(PARAMS))
    assert result.status is SolveStatus.FEASIBLE
    assert golden.violations(p, result) == []


def test_coverage_skipped_when_phase_a_left_no_time(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _freeze(monkeypatch)
    deadline = clock.now + _phase_a_budget(PARAMS)
    real = phase_a.solve_phase_a
    ladder: list[PhaseAOutcome] = []

    def late(*args: object, **kwargs: object) -> PhaseAOutcome:
        if not ladder:  # one real plan, reported FEASIBLE for every ladder run
            outcome = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]
            ladder.append(replace(outcome, info=replace(outcome.info, status=SolveStatus.FEASIBLE)))
        clock.now = deadline  # the ladder overran into the reserve: no time left for the step
        return ladder[0]

    monkeypatch.setattr(solve_module, "solve_phase_a", late)
    monkeypatch.setattr(solve_module, "solve_coverage", _must_not_run)
    p = _orphans()

    result = solve(p, PARAMS, progress=Progress(clock=clock))

    plan = ladder[0].plan
    assert plan is not None
    assert zero_day_classes(build_index(p), plan) != []  # work for the step, but no time left
    kept = {(p.classes[c].id, d) for c, d in plan.in_person}
    assert golden.in_person_days(result) == kept
    assert result.status is SolveStatus.FEASIBLE
    assert golden.violations(p, result) == []


_seconds = st.floats(0.0, 10_000.0, allow_nan=False)


@given(
    phase_a_s=_seconds,
    elapsed=_seconds,
    det=st.none() | st.floats(0.0, 1_000.0, allow_nan=False),
)
def test_coverage_budgets_split_phase_a_without_overrun(
    phase_a_s: float, elapsed: float, det: float | None
) -> None:
    # Pure split: the ladder and the step together never get more than phase A was given.
    now = 1000.0
    deadline = now + phase_a_s - elapsed  # may already be past
    params = replace(PARAMS, max_deterministic_time=det)
    split = solve_module._coverage_budgets  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    ladder, step = split(params, deadline, now)

    assert step.deadline == deadline
    if det is None:
        assert step.det is None
        assert ladder.det is None
        assert step.max_s is not None
        assert 0.0 <= step.max_s <= solve_module.COVERAGE_MAX_S
        assert ladder.deadline + step.max_s == pytest.approx(deadline)
        assert ladder.deadline <= deadline
    else:
        assert ladder.deadline == deadline  # the wall clock is only the safety net
        assert ladder.det is not None
        assert step.det is not None
        assert ladder.det + step.det == pytest.approx(det)
        assert step.det == pytest.approx(solve_module.COVERAGE_SHARE * det)
