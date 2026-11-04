"""Use case: build a main (1.1) or alert (1.2) timetable. The single entry point of the engine.

Flow (ADR 2026-11-04 §1-§3): pre-checks -> phase A (days, formats, capacity, objective) with a
quality ladder -> phase B (lesson numbers, one small model per day) -> ``Result``.
The time limit covers everything; phase B keeps a reserve of ``min(15 s, 0.15 T)``.
"""

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
from lad_solver.model.phase_a import DayPlan, PhaseAOutcome, score, solve_phase_a
from lad_solver.model.phase_b import DayAssignment, DayDemand, assign_day
from lad_solver.model.stop import StopHandle

PROBE_SHARE = 0.25  # no phase A solution within this share of its budget -> drop the row cap
PHASE_B_RESERVE_MAX_S = 15.0
PHASE_B_RESERVE_SHARE = 0.15
PHASE_B_DAY_MAX_S = 5.0
# Per day, feasibility only and not interruptible: a timed-out or stopped day still gets a grid.
PHASE_B_RETRY_S = 2.0
MIN_RUN_S = 0.05
BUSY_TEACHER_LOAD = 0.9
BUSY_TEACHERS_MAX = 3

_FOUND = frozenset({SolveStatus.OPTIMAL, SolveStatus.FEASIBLE})


@dataclass(frozen=True, slots=True)
class _PhaseA:
    status: SolveStatus
    plan: DayPlan | None
    best_bound: float | None
    first_solution_at: float | None  # time.monotonic()
    relaxed: bool
    note: str | None = None  # anomaly worth seeing in Stats.phase_a_status


def solve(problem: Problem, params: Params, stop: StopHandle | None = None) -> Result:
    """Pre-checks -> phase A (days, formats, capacity, objective) -> phase B (lesson numbers)."""
    started = time.monotonic()
    conflicts = precheck(problem, row_day_cap_min=params.row_day_cap_min)
    if conflicts:
        invalid = any(c.rule is ConflictRule.INVALID_INPUT for c in conflicts)
        status = SolveStatus.MODEL_INVALID if invalid else SolveStatus.INFEASIBLE
        stats = _stats(params, started, stop_reason=StopReason.PRECHECK)
        return Result(status, conflicts=conflicts, stats=stats)

    index = build_index(problem)
    reserve_b = _phase_b_reserve(params)
    a = _run_phase_a(index, params, stop, deadline=started + params.time_limit_s - reserve_b)
    stopped = _is_stopped(stop)
    if a.plan is None:
        return _no_plan(index, params, started, a, stopped=stopped)

    day_limit = min(PHASE_B_DAY_MAX_S, reserve_b / len(problem.school_days))
    lessons = _run_phase_b(index, a.plan, params, stop, day_limit)
    sc = score(index, a.plan)
    stats = _stats(
        params,
        started,
        stop_reason=_stop_reason(a.status, stopped=_is_stopped(stop)),
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
            return Result(SolveStatus.UNKNOWN, stats=cleared)
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
    index: Index, params: Params, stop: StopHandle | None, *, deadline: float
) -> _PhaseA:
    """Capped model; if it is proven infeasible or finds nothing in the probe -> uncapped."""
    started = time.monotonic()
    budget = max(deadline - started, MIN_RUN_S)
    base = SolverConfig(
        time_limit_s=budget,
        num_workers=params.num_workers,
        random_seed=params.random_seed,
        interleave_search=params.interleave_search,
        max_memory_mb=params.max_memory_mb,
    )
    det = params.max_deterministic_time
    caps = index.row_caps(params.row_day_cap_min, relaxed=False)
    probe = solve_phase_a(
        index, caps, base.with_time(PROBE_SHARE * budget, _share(det, PROBE_SHARE)), stop
    )
    first_at = probe.info.first_solution_at
    if probe.plan is not None:
        if probe.info.status is SolveStatus.OPTIMAL or (_is_stopped(stop)):
            return _PhaseA(
                probe.info.status, probe.plan, probe.info.best_bound, first_at, relaxed=False
            )
        rest = base.with_time(_remaining(deadline), _share(det, 1 - PROBE_SHARE))
        follow = solve_phase_a(index, caps, rest, stop, hint=probe.plan)
        return _better(index, probe, follow, first_at)
    if probe.info.status not in {SolveStatus.INFEASIBLE, SolveStatus.UNKNOWN} or (
        _is_stopped(stop)
    ):
        return _PhaseA(probe.info.status, None, None, None, relaxed=False)
    relaxed_caps = index.row_caps(params.row_day_cap_min, relaxed=True)
    rest = base.with_time(_remaining(deadline), _share(det, 1 - PROBE_SHARE))
    relaxed = solve_phase_a(index, relaxed_caps, rest, stop)
    return _PhaseA(
        relaxed.info.status,
        relaxed.plan,
        relaxed.info.best_bound,
        relaxed.info.first_solution_at,
        relaxed=True,
    )


def _remaining(deadline: float) -> float:
    return max(deadline - time.monotonic(), MIN_RUN_S)


def _better(
    index: Index, probe: PhaseAOutcome, follow: PhaseAOutcome, first_at: float | None
) -> _PhaseA:
    bounds = [b for b in (probe.info.best_bound, follow.info.best_bound) if b is not None]
    bound = min(bounds) if bounds else None
    if probe.plan is None:
        return _PhaseA(follow.info.status, follow.plan, bound, first_at, relaxed=False)
    if (
        follow.plan is not None
        and score(index, follow.plan).objective >= score(index, probe.plan).objective
    ):
        return _PhaseA(follow.info.status, follow.plan, bound, first_at, relaxed=False)
    note = None
    if follow.info.status in {SolveStatus.MODEL_INVALID, SolveStatus.INFEASIBLE}:
        # The hinted follow-up of a feasible model cannot be either: keep the probe, but say so.
        note = f"{SolveStatus.FEASIBLE.value}; follow-up {follow.info.status.value}"
    return _PhaseA(SolveStatus.FEASIBLE, probe.plan, bound, first_at, relaxed=False, note=note)


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


def _run_phase_b(
    index: Index, plan: DayPlan, params: Params, stop: StopHandle | None, day_limit: float
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
            day, demands, index.lessons_per_day, params, stop=stop, limit=day_limit
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
) -> DayAssignment:
    """Compact model within ``limit``; on timeout (or stop) one feasibility-only retry."""
    config = SolverConfig(
        time_limit_s=limit,
        num_workers=params.num_workers,
        random_seed=params.random_seed,
        interleave_search=params.interleave_search,
        max_memory_mb=params.max_memory_mb,
    )
    if not _is_stopped(stop):
        first = assign_day(day, demands, lessons_per_day, config=config, stop=stop)
        if first.periods is not None or first.status is not SolveStatus.UNKNOWN:
            return first  # found, or a proven failure that a retry cannot fix
    retry = config.with_time(PHASE_B_RETRY_S)
    return assign_day(day, demands, lessons_per_day, config=retry, compact=False)


def _is_stopped(stop: StopHandle | None) -> bool:
    return stop is not None and stop.stopped


def _day_formats(index: Index, plan: DayPlan) -> tuple[DayFormat, ...]:
    return tuple(
        DayFormat(c.id, d, (pos, d) in plan.in_person)
        for pos, c in enumerate(index.problem.classes)
        for d in index.days
    )


def _stop_reason(status: SolveStatus, *, stopped: bool) -> StopReason:
    if stopped:
        return StopReason.STOPPED
    if status in {SolveStatus.OPTIMAL, SolveStatus.INFEASIBLE}:
        return StopReason.COMPLETED
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
        phase_a_status=None if phase_a is None else (phase_a.note or phase_a.status.value),
        quality_relaxed=False if phase_a is None else phase_a.relaxed,
        stop_reason=stop_reason,
        num_workers=params.num_workers,
        random_seed=params.random_seed,
    )


def _since(started: float, moment: float | None) -> float | None:
    return None if moment is None else moment - started
