"""Use case: build a main (1.1) or alert (1.2) timetable. The single entry point of the engine.

Flow (ADR 2026-11-04 §1-§3): pre-checks -> phase A (days, formats, capacity, objective) with a
quality ladder -> coverage step (ADR 2026-11-14: more classes with an in-person day, the phase A
objective kept) -> phase B (lesson numbers, one small model per day) -> ``Result``.
The time limit covers everything; phase B keeps a reserve of ``min(15 s, 0.15 T)``.
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass, replace

from lad_solver.domain.prechecks import precheck
from lad_solver.domain.problem import Params, Problem
from lad_solver.domain.result import (
    Conflict,
    ConflictRule,
    DayFormat,
    Lesson,
    Result,
    SolveStatus,
    Stats,
    StopReason,
)
from lad_solver.model.cpsat import SolverConfig
from lad_solver.model.index import Index, build_index
from lad_solver.model.phase_a import (
    MIN_RUN_S,
    DayPlan,
    PhaseAOutcome,
    score,
    solve_coverage,
    solve_phase_a,
    zero_day_classes,
)
from lad_solver.model.phase_b import DayAssignment, DayDemand, assign_day
from lad_solver.model.progress import Progress, StallPolicy
from lad_solver.model.stop import StopHandle

_log = logging.getLogger(__name__)

PROBE_SHARE = 0.25  # no phase A solution within this share of its budget -> drop the row cap
# Coverage step (ADR 2026-11-14, addendum 2026-11-15): the ladder always ends min(10 s, 0.15 of the
# phase A budget) before the phase A deadline, or gets 0.85 of max_deterministic_time (the wall
# clock is then only the safety net, shared with the step). Whether a class ends up without an
# in-person day is known only after the ladder, so the reserve is kept even without static orphans;
# with no such class the step does not run and the reserve goes to phase B unused.
COVERAGE_MAX_S = 10.0
COVERAGE_SHARE = 0.15
PHASE_B_RESERVE_MAX_S = 15.0
PHASE_B_RESERVE_SHARE = 0.15
PHASE_B_DAY_MAX_S = 5.0
# Per day, feasibility only and not interruptible: a timed-out or stopped day still gets a grid.
PHASE_B_RETRY_S = 2.0
# With params.max_deterministic_time set, phase B days get these deterministic budgets instead of
# wall-clock ones (same magnitudes), so a seeded result does not depend on the machine.
PHASE_B_DAY_DET = PHASE_B_DAY_MAX_S
PHASE_B_RETRY_DET = PHASE_B_RETRY_S
BUSY_TEACHER_LOAD = 0.9
BUSY_TEACHERS_MAX = 3

_FOUND = frozenset({SolveStatus.OPTIMAL, SolveStatus.FEASIBLE})


@dataclass(frozen=True, slots=True)
class _PhaseA:
    status: SolveStatus
    plan: DayPlan | None
    best_bound: float | None
    relaxed: bool
    caps: tuple[int, ...]  # row-day caps of the run reported here (the coverage step keeps them)
    note: str | None = None  # anomaly worth seeing in Stats.phase_a_status
    stalled: bool = False  # stopped by the no-improvement rule (ADR 2026-11-08)
    first_solution_at: float | None = None  # time.monotonic(), from Progress
    last_improvement_at: float | None = None  # time.monotonic(), from Progress


@dataclass(frozen=True, slots=True)
class _Budget:
    deadline: float  # time.monotonic()
    det: float | None  # max_deterministic_time of the runs; None = wall clock only
    max_s: float | None = None  # wall-clock cap of a run started later, from its own start

    def end(self, now: float) -> float:
        """Wall-clock end of a run started at ``now``; with ``det``, ``deadline`` (safety net)."""
        if self.det is not None or self.max_s is None:
            return self.deadline
        return min(self.deadline, now + self.max_s)


def solve(
    problem: Problem,
    params: Params,
    stop: StopHandle | None = None,
    *,
    progress: Progress | None = None,
) -> Result:
    """Pre-checks -> phase A (days, formats, capacity, objective) -> phase B (lesson numbers).

    ``params.stall_min_s`` needs a ``progress`` whose ``check()`` the caller ticks while the
    solve runs (the engine itself starts no threads); without it the rule could never fire.
    """
    if params.stall_min_s is not None and progress is None:
        msg = "params.stall_min_s is set: pass a progress and tick progress.check()"
        raise ValueError(msg)
    started = time.monotonic()
    conflicts = precheck(problem)
    if conflicts:
        invalid = any(c.rule is ConflictRule.INVALID_INPUT for c in conflicts)
        status = SolveStatus.MODEL_INVALID if invalid else SolveStatus.INFEASIBLE
        stats = _stats(params, started, stop_reason=StopReason.PRECHECK)
        return Result(status, conflicts=conflicts, stats=stats)

    index = build_index(problem)
    reserve_b = _phase_b_reserve(params)
    a = _run_phase_a(
        index,
        params,
        stop,
        progress or Progress(),
        deadline=started + params.time_limit_s - reserve_b,
    )
    stopped = _is_stopped(stop)
    if a.plan is None:
        return _no_plan(index, params, started, a, stopped=stopped)

    day_limit = min(PHASE_B_DAY_MAX_S, reserve_b / len(problem.school_days))
    end = started + params.time_limit_s
    lessons = _run_phase_b(index, a.plan, params, stop, day_limit, end=end)
    sc = score(index, a.plan)
    stats = _stats(
        params,
        started,
        stop_reason=stop_reason(a.status, stopped=_is_stopped(stop), stalled=a.stalled),
        objective=sc.objective,
        best_bound=a.best_bound,
        primary=sc.primary,
        secondary=sc.secondary,
        phase_a=a,
    )
    if isinstance(lessons, SolveStatus):
        # No grid: the phase A objective describes nothing that is returned.
        cleared = replace(stats, objective=None, best_bound=None, primary=None, secondary=None)
        if lessons is SolveStatus.UNKNOWN:  # out of time even with the feasibility retry
            reason = StopReason.STOPPED if _is_stopped(stop) else StopReason.TIME_LIMIT
            return Result(SolveStatus.UNKNOWN, stats=replace(cleared, stop_reason=reason))
        # INFEASIBLE / MODEL_INVALID cannot happen for valid loads (König): an engine bug,
        # reported as MODEL_INVALID without invalid_input (the worker sends error.code=internal).
        return Result(SolveStatus.MODEL_INVALID, stats=cleared)
    status = SolveStatus.OPTIMAL if a.status is SolveStatus.OPTIMAL else SolveStatus.FEASIBLE
    return Result(status, lessons, _day_formats(index, a.plan), stats=stats)


def _phase_b_reserve(params: Params) -> float:
    if params.phase_b_reserve_s is not None:
        return params.phase_b_reserve_s
    return min(PHASE_B_RESERVE_MAX_S, PHASE_B_RESERVE_SHARE * params.time_limit_s)


def _share(value: float | None, share: float) -> float | None:
    return None if value is None else value * share


def _run_phase_a(
    index: Index,
    params: Params,
    stop: StopHandle | None,
    progress: Progress,
    *,
    deadline: float,
) -> _PhaseA:
    """One clock for every ladder run; the no-improvement rule stops whichever is running.

    The coverage step runs once that clock has finished: its objective has another scale.
    """
    ladder, coverage = _coverage_budgets(params, deadline, time.monotonic())
    policy = (
        None
        if params.stall_min_s is None
        else StallPolicy(min_s=params.stall_min_s, ratio=params.stall_ratio)
    )
    progress.begin(policy)
    try:
        a = _phase_a_ladder(index, params, stop, progress, budget=ladder)
    finally:
        stalled = progress.finish()
    a = replace(
        a,
        stalled=stalled,
        first_solution_at=progress.first_solution_at,
        last_improvement_at=progress.last_improvement_at,
    )
    return _cover(index, a, params, stop, coverage)


def _coverage_budgets(params: Params, deadline: float, now: float) -> tuple[_Budget, _Budget]:
    """(ladder, coverage step) of a phase A that starts at ``now`` and must end by ``deadline``.

    The ladder always leaves the step its reserve: which classes the ladder's plan leaves without
    an in-person day is known only once it has run. A deterministic budget splits the work (0.15
    of it for the step, so phase A never plans more than ``max_deterministic_time``), as in the
    ladder itself: the wall clock is then just the safety net, the same ``deadline`` for both, and
    the wall-clock reserve and cap do not apply (they would make a seeded result depend on the
    machine).
    """
    det = params.max_deterministic_time
    if det is not None:
        step = det * COVERAGE_SHARE
        return _Budget(deadline, det - step), _Budget(deadline, step)
    reserve = min(COVERAGE_MAX_S, COVERAGE_SHARE * max(deadline - now, 0.0))
    return _Budget(deadline - reserve, None), _Budget(deadline, None, max_s=reserve)


def _config(params: Params, time_limit_s: float, det: float | None = None) -> SolverConfig:
    return SolverConfig(
        time_limit_s=time_limit_s,
        num_workers=params.num_workers,
        random_seed=params.random_seed,
        max_deterministic_time=det,
        interleave_search=params.interleave_search,
        max_memory_mb=params.max_memory_mb,
    )


def _phase_a_ladder(
    index: Index,
    params: Params,
    stop: StopHandle | None,
    progress: Progress,
    *,
    budget: _Budget,
) -> _PhaseA:
    """Capped model; if it is proven infeasible or finds nothing in the probe -> uncapped."""
    deadline, det = budget.deadline, budget.det
    seconds = max(deadline - time.monotonic(), MIN_RUN_S)
    base = _config(params, seconds)
    caps = index.row_caps(params.row_day_cap_min, relaxed=False)
    # The probe keeps its fixed share; later runs get what is left after building their model.
    probe = solve_phase_a(
        index,
        caps,
        # A deterministic budget splits the work; the wall clock is then just the safety net.
        base.with_time(
            seconds if det is not None else PROBE_SHARE * seconds, _share(det, PROBE_SHARE)
        ),
        stop,
        progress=progress,
    )
    rest = base.with_time(_remaining(deadline), _share(det, 1 - PROBE_SHARE))
    if probe.plan is not None:
        if probe.info.status is SolveStatus.OPTIMAL or _is_stopped(stop) or progress.stalled:
            return _PhaseA(
                probe.info.status, probe.plan, probe.info.best_bound, relaxed=False, caps=caps
            )
        follow = solve_phase_a(
            index, caps, rest, stop, hint=probe.plan, progress=progress, deadline=deadline
        )
        return _better(index, caps, probe, probe.plan, follow)
    if _is_stopped(stop):
        # INFEASIBLE here is a proof for the capped model only: the relaxed run never started.
        status = (
            SolveStatus.UNKNOWN
            if probe.info.status is SolveStatus.INFEASIBLE
            else probe.info.status
        )
        return _PhaseA(status, None, None, relaxed=False, caps=caps)
    if probe.info.status not in {SolveStatus.INFEASIBLE, SolveStatus.UNKNOWN}:
        return _PhaseA(probe.info.status, None, None, relaxed=False, caps=caps)
    relaxed_caps = index.row_caps(params.row_day_cap_min, relaxed=True)
    relaxed = solve_phase_a(index, relaxed_caps, rest, stop, progress=progress, deadline=deadline)
    return _PhaseA(
        relaxed.info.status,
        relaxed.plan,
        relaxed.info.best_bound,
        relaxed=True,
        caps=relaxed_caps,
    )


def _remaining(deadline: float) -> float:
    return max(deadline - time.monotonic(), MIN_RUN_S)


def _better(
    index: Index,
    caps: tuple[int, ...],
    probe: PhaseAOutcome,
    probe_plan: DayPlan,
    follow: PhaseAOutcome,
) -> _PhaseA:
    bounds = [b for b in (probe.info.best_bound, follow.info.best_bound) if b is not None]
    bound = min(bounds) if bounds else None
    if (
        follow.plan is not None
        and score(index, follow.plan).objective >= score(index, probe_plan).objective
    ):
        return _PhaseA(follow.info.status, follow.plan, bound, relaxed=False, caps=caps)
    note = None
    if follow.info.status in {SolveStatus.MODEL_INVALID, SolveStatus.INFEASIBLE}:
        # The hinted follow-up of a feasible model cannot be either: keep the probe, but say so.
        note = f"{SolveStatus.FEASIBLE.value}; follow-up {follow.info.status.value}"
    return _PhaseA(SolveStatus.FEASIBLE, probe_plan, bound, relaxed=False, caps=caps, note=note)


def _cover(
    index: Index, a: _PhaseA, params: Params, stop: StopHandle | None, budget: _Budget
) -> _PhaseA:
    """Coverage step (ADR 2026-11-14) on the ladder's plan; the plan it returns may replace it.

    Runs only if some class has no in-person day, the job is not stopped and time is left. The
    floor is the ladder plan's own objective: V* after OPTIMAL, the value found after FEASIBLE.
    The new plan is taken only with strictly fewer classes without an in-person day and an
    objective >= the floor; anything else (no plan, a stop) keeps the ladder's plan. Status,
    bound and stop reason stay the ladder's; ``solve`` rescores the objective from the plan.
    """
    if a.plan is None or _is_stopped(stop):
        return a
    before = zero_day_classes(index, a.plan)
    now = time.monotonic()
    end = budget.end(now)
    left = end - now
    if not before or left <= MIN_RUN_S:
        return a
    floor = score(index, a.plan).objective
    config = _config(params, left, budget.det)
    outcome = solve_coverage(index, a.caps, a.plan, floor, config, stop=stop, deadline=end)
    status = outcome.info.status
    if outcome.plan is None or _is_stopped(stop):
        _log_coverage("kept", status, len(before), None)
        return a
    after = zero_day_classes(index, outcome.plan)
    if len(after) >= len(before) or score(index, outcome.plan).objective < floor:
        _log_coverage("kept", status, len(before), len(after))
        return a
    _log_coverage("accepted", status, len(before), len(after))
    return replace(a, plan=outcome.plan)


def _log_coverage(verdict: str, status: SolveStatus, before: int, after: int | None) -> None:
    """Counts and statuses only. A hinted feasible model cannot be INFEASIBLE / MODEL_INVALID."""
    broken = status in {SolveStatus.INFEASIBLE, SolveStatus.MODEL_INVALID}
    _log.log(
        logging.WARNING if broken else logging.INFO,
        "coverage step %s (%s): classes without an in-person day %d -> %s",
        verdict,
        status.value,
        before,
        "-" if after is None else after,
    )


def _no_plan(index: Index, params: Params, started: float, a: _PhaseA, *, stopped: bool) -> Result:
    if a.status is SolveStatus.INFEASIBLE:
        stats = _stats(params, started, stop_reason=StopReason.COMPLETED, phase_a=a)
        return Result(SolveStatus.INFEASIBLE, conflicts=_packing_conflicts(index), stats=stats)
    status = (
        SolveStatus.MODEL_INVALID if a.status is SolveStatus.MODEL_INVALID else SolveStatus.UNKNOWN
    )
    reason = StopReason.STOPPED if stopped else StopReason.TIME_LIMIT
    return Result(status, stats=_stats(params, started, stop_reason=reason, phase_a=a))


def _packing_conflicts(index: Index) -> tuple[Conflict, ...]:
    """``packing`` plus up to three busiest teachers (load / available slots >= 0.9) as hints."""
    problem = index.problem
    load = Counter[int]()
    for r, t in enumerate(index.row_teacher):
        load[t] += index.row_hours[r]
    ratios: list[tuple[float, str]] = []
    for t, days in enumerate(index.teacher_days):
        slots = len(days) * index.lessons_per_day
        if load[t] and slots and load[t] / slots >= BUSY_TEACHER_LOAD:
            ratios.append((load[t] / slots, problem.teachers[t].id))
    ratios.sort(key=lambda x: (-x[0], x[1]))
    busiest = tuple(
        Conflict(ConflictRule.PACKING, teacher_id=tid, hint=True)
        for _, tid in ratios[:BUSY_TEACHERS_MAX]
    )
    return (Conflict(ConflictRule.PACKING), *busiest)


def _run_phase_b(  # noqa: PLR0913 - keyword-only overall deadline
    index: Index,
    plan: DayPlan,
    params: Params,
    stop: StopHandle | None,
    day_limit: float,
    *,
    end: float,
) -> tuple[Lesson, ...] | SolveStatus:
    """All lessons, or the status of the first day that got no assignment."""
    rows = index.problem.rows
    lessons: list[Lesson] = []
    for day in index.days:
        demands = [
            DayDemand(r, rows[r].class_id, rows[r].teacher_id, n)
            for (r, d), n in sorted(plan.loads.items())
            if d == day
        ]
        if not demands:
            continue
        assigned = _assign_day(
            day, demands, index.lessons_per_day, params, stop=stop, limit=day_limit, end=end
        )
        if assigned.periods is None:
            return assigned.status
        lessons.extend(
            Lesson(rows[r].class_id, day, p, rows[r].subject_id, rows[r].teacher_id)
            for r, ps in assigned.periods.items()
            for p in ps
        )
    return tuple(sorted(lessons, key=lambda x: (x.class_id, x.day, x.lesson_no)))


def _assign_day(  # noqa: PLR0913 - keyword-only stop/limit
    day: int,
    demands: list[DayDemand],
    lessons_per_day: int,
    params: Params,
    *,
    stop: StopHandle | None,
    limit: float,
    end: float,
) -> DayAssignment:
    """Compact model within ``limit``; on timeout (or stop) one feasibility-only retry.

    At most two CP-SAT runs per day (``fallback=False``): ``limit + PHASE_B_RETRY_S``. With a
    deterministic budget the runs get ``PHASE_B_DAY_DET`` / ``PHASE_B_RETRY_DET`` instead, and the
    wall clock only stops them at the overall deadline ``end`` (``time.monotonic()``).
    """
    deterministic = params.max_deterministic_time is not None
    config = SolverConfig(
        time_limit_s=limit,
        num_workers=params.num_workers,
        random_seed=params.random_seed,
        interleave_search=params.interleave_search,
        max_memory_mb=params.max_memory_mb,
    )
    if deterministic:
        config = config.with_time(_remaining(end), PHASE_B_DAY_DET)
    if not _is_stopped(stop):
        first = assign_day(day, demands, lessons_per_day, config=config, stop=stop, fallback=False)
        if first.periods is not None or first.status is not SolveStatus.UNKNOWN:
            return first  # found, or a proven failure that a retry cannot fix
    retry = (
        config.with_time(_remaining(end), PHASE_B_RETRY_DET)
        if deterministic
        else config.with_time(PHASE_B_RETRY_S)
    )
    return assign_day(day, demands, lessons_per_day, config=retry, compact=False)


def _is_stopped(stop: StopHandle | None) -> bool:
    return stop is not None and stop.stopped


def _day_formats(index: Index, plan: DayPlan) -> tuple[DayFormat, ...]:
    return tuple(
        DayFormat(c.id, d, (pos, d) in plan.in_person)
        for pos, c in enumerate(index.problem.classes)
        for d in index.days
    )


def stop_reason(status: SolveStatus, *, stopped: bool, stalled: bool) -> StopReason:
    """stopped > completed (a proof wins even if the stall fired) > no_improvement > time_limit."""
    if stopped:
        return StopReason.STOPPED
    if status in {SolveStatus.OPTIMAL, SolveStatus.INFEASIBLE}:
        return StopReason.COMPLETED
    if stalled:
        return StopReason.NO_IMPROVEMENT
    return StopReason.TIME_LIMIT


def _stats(  # noqa: PLR0913
    params: Params,
    started: float,
    *,
    stop_reason: StopReason,
    objective: int | None = None,
    best_bound: float | None = None,
    primary: int | None = None,
    secondary: int | None = None,
    phase_a: _PhaseA | None = None,
) -> Stats:
    return Stats(
        objective=objective,
        best_bound=best_bound,
        primary=primary,
        secondary=secondary,
        wall_time_s=time.monotonic() - started,
        first_solution_s=_since(started, None if phase_a is None else phase_a.first_solution_at),
        last_improvement_s=_since(
            started, None if phase_a is None else phase_a.last_improvement_at
        ),
        phase_a_status=None if phase_a is None else (phase_a.note or phase_a.status.value),
        quality_relaxed=False if phase_a is None else phase_a.relaxed,
        stop_reason=stop_reason,
        num_workers=params.num_workers,
        random_seed=params.random_seed,
    )


def _since(started: float, moment: float | None) -> float | None:
    return None if moment is None else moment - started
