"""Phase B: lesson numbers for one day (ADR 2026-11-04 §1).

Given the day loads of phase A (class-day and teacher-day <= L) a clash-free assignment always
exists (König). The soft objective is lexicographic per class with k_c lessons: first no gaps
inside the day (span last_c - first_c + 1 beyond k_c), then the earliest start (first_c - 1). A
class may start later (lessons 2-4) rather than get a hole; nothing is a hard constraint, so
fixed lessons (substitutions) that force a gap still give an assignment.
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
        _minimise_gaps_then_start(model, x, day, by_class, lessons_per_day)
    return model, x


def _minimise_gaps_then_start(
    model: cp_model.CpModel,
    x: Mapping[tuple[int, int], cp_model.IntVar],
    day: int,
    by_class: Mapping[str, Sequence[DayDemand]],
    lessons_per_day: int,
) -> None:
    """``W * sum(hole_c) + sum(first_c - 1)``: holes before a late start (bug 11).

    ``first_c <= p <= last_c`` for every busy period p, so at the optimum they are the first and
    last lessons; ``hole_c = last_c - first_c + 1 - k_c >= 0``. ``first_c - 1 <= L - k_c``, hence
    ``W = 1 + sum(L - k_c)`` makes one hole cost more than every late start of the day together.
    """
    weight = 1 + sum(lessons_per_day - sum(d.count for d in g) for g in by_class.values())
    bounds: list[tuple[cp_model.IntVar, cp_model.IntVar, int]] = []
    for class_id, group in by_class.items():
        k = sum(d.count for d in group)
        first = model.new_int_var(1, lessons_per_day - k + 1, f"first_{day}_{class_id}")
        last = model.new_int_var(k, lessons_per_day, f"last_{day}_{class_id}")
        for p in range(1, lessons_per_day + 1):
            busy = model.new_bool_var(f"busy_{day}_{class_id}_{p}")
            model.add(busy == sum(x[d.key, p] for d in group))  # at most one lesson per period
            model.add(first <= p).only_enforce_if(busy)  # pyright: ignore[reportUnknownMemberType]
            model.add(last >= p).only_enforce_if(busy)  # pyright: ignore[reportUnknownMemberType]
        model.add(last - first >= k - 1)  # hole_c >= 0
        bounds.append((first, last, k))
    # W * (last - first + 1 - k) + (first - 1), as weighted variables plus a constant
    variables = [v for first, last, _ in bounds for v in (last, first)]
    coefficients = [c for _ in bounds for c in (weight, 1 - weight)]
    constant = sum(weight * (1 - k) - 1 for _, _, k in bounds)
    model.minimize(sum((c * v for v, c in zip(variables, coefficients, strict=True)), constant))
