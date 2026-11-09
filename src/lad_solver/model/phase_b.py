"""Phase B: lesson numbers for one day (ADR 2026-11-04 §1).

Given the day loads of phase A (class-day and teacher-day <= L) a clash-free assignment always
exists (König). The soft objective packs each class into periods 1..k_c (no gaps, from lesson 1):
it minimises lessons placed after the class's k_c-th period.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ortools.sat.python import cp_model

from lad_solver.domain.result import SolveStatus
from lad_solver.model.cpsat import SolverConfig, run
from lad_solver.model.stop import StopHandle


@dataclass(frozen=True, slots=True)
class DayDemand:
    """``count`` lessons on this day of one curriculum row (``key``)."""

    key: int
    class_id: str
    teacher_id: str
    count: int


type Periods = dict[int, tuple[int, ...]]  # key -> lesson numbers 1..L (ascending)


@dataclass(frozen=True, slots=True)
class DayAssignment:
    """Lesson numbers of one day, or ``None`` with the status of the last CP-SAT run.

    UNKNOWN = ran out of time (or stopped); INFEASIBLE / MODEL_INVALID = a bug, since valid
    phase A loads always have an assignment (König).
    """

    periods: Periods | None
    status: SolveStatus


def assign_day(  # noqa: PLR0913 - mirrors the ADR signature plus solver settings
    day: int,
    lessons: Sequence[DayDemand],
    lessons_per_day: int,
    fixed: Sequence[tuple[int, int]] = (),
    hint: Mapping[int, Sequence[int]] | None = None,
    *,
    config: SolverConfig,
    stop: StopHandle | None = None,
    compact: bool = True,
    fallback: bool = True,
) -> DayAssignment:
    """Lesson numbers per demand key.

    ``fixed`` pins (key, lesson_no) pairs (substitutions); ``hint`` suggests lesson numbers.
    If the compact model ends UNKNOWN and ``fallback`` is set, a feasibility-only model is tried
    with the same limit; a caller with its own retry passes ``fallback=False``.
    """
    attempts = (True, False) if compact and fallback else (compact,)
    status = SolveStatus.UNKNOWN
    for with_objective in attempts:
        model, x = _build(day, lessons, lessons_per_day, fixed, hint, objective=with_objective)
        solver, info = run(model, config, stop, has_objective=with_objective)
        if info.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}:
            periods = {
                d.key: tuple(p for p in range(1, lessons_per_day + 1) if solver.value(x[d.key, p]))
                for d in lessons
            }
            return DayAssignment(periods, info.status)
        status = info.status
        if status is not SolveStatus.UNKNOWN:
            break
    return DayAssignment(None, status)


def assign_periods(  # noqa: PLR0913 - mirrors the ADR signature plus solver settings
    day: int,
    lessons: Sequence[DayDemand],
    lessons_per_day: int,
    fixed: Sequence[tuple[int, int]] = (),
    hint: Mapping[int, Sequence[int]] | None = None,
    *,
    config: SolverConfig,
    stop: StopHandle | None = None,
    compact: bool = True,
) -> Periods | None:
    """``assign_day`` without the status: lesson numbers, or ``None`` if none were found."""
    return assign_day(
        day, lessons, lessons_per_day, fixed, hint, config=config, stop=stop, compact=compact
    ).periods


def _build(  # noqa: PLR0913
    day: int,
    lessons: Sequence[DayDemand],
    lessons_per_day: int,
    fixed: Sequence[tuple[int, int]],
    hint: Mapping[int, Sequence[int]] | None,
    *,
    objective: bool,
) -> tuple[cp_model.CpModel, dict[tuple[int, int], cp_model.IntVar]]:
    model = cp_model.CpModel()
    periods = range(1, lessons_per_day + 1)
    x = {(d.key, p): model.new_bool_var(f"x_{day}_{d.key}_{p}") for d in lessons for p in periods}
    by_class: defaultdict[str, list[DayDemand]] = defaultdict(list)
    by_teacher: defaultdict[str, list[DayDemand]] = defaultdict(list)
    for d in lessons:
        model.add(sum(x[d.key, p] for p in periods) == d.count)
        by_class[d.class_id].append(d)
        by_teacher[d.teacher_id].append(d)
    for group in (*by_class.values(), *by_teacher.values()):
        for p in periods:
            model.add_at_most_one(x[d.key, p] for d in group)
    for key, p in fixed:
        model.add(x[key, p] == 1)
    if hint is not None:
        for (key, p), var in x.items():
            model.add_hint(var, p in hint.get(key, ()))
    if objective:
        late = [
            x[d.key, p]
            for group in by_class.values()
            for d in group
            for p in periods
            if p > sum(g.count for g in group)
        ]
        model.minimize(sum(late))
    return model, x
