"""The day capacity rule of spec §04, shared by the build (phase A) and the substitution model.

- 1.1: pupils of in-person classes + non-teaching staff + teachers on site <= passport.
- 1.2: (1) pupils of in-person classes <= seats AND (2) everyone, as in 1.1, <= passport.

A teacher is on site when he teaches at least one lesson of an in-person class that day, and is
counted once. Staff is a constant, so it moves to the right-hand side: ``passport - staff`` is exact
in Python. The right-hand sides are clamped to what a day can hold at most, so that limits beyond
int64 (CP-SAT constants) mean "unlimited" instead of crashing the model build.
"""

from dataclasses import dataclass

from ortools.sat.python import cp_model

from lad_solver.domain.problem import AlertLimits, Capacity


@dataclass(frozen=True, slots=True)
class DayRoom:
    """Right-hand sides of one day's capacity constraints."""

    people: int  # pupils + teachers on site may reach this (the passport minus the staff)
    pupils: int | None  # model 1.2 seats; None = no seats constraint


def day_people(pupils: int, staff: int, teachers: int) -> int:
    """The passport load of a day: pupils of in-person classes + staff + teachers on site."""
    return pupils + staff + teachers


def day_room(capacity: Capacity, staff: int, *, max_pupils: int, max_teachers: int) -> DayRoom:
    """The room of a build day; ``max_pupils`` / ``max_teachers`` bound what a day can hold."""
    people = min(capacity.passport - staff, max_pupils + max_teachers)
    pupils = min(capacity.seats, max_pupils) if isinstance(capacity, AlertLimits) else None
    return DayRoom(people=people, pupils=pupils)


def baseline_room(
    capacity: Capacity, staff: int, baseline: int, *, max_pupils: int, max_teachers: int
) -> DayRoom:
    """The room of a day of a substitution: never worse than the baseline week.

    The passport becomes ``max(passport, baseline)`` (``baseline`` = that day's people load, staff
    included), so a week that is already over the limit may stay there but not grow (ADR
    2026-11-15-solver-substitution-engine). Day formats are fixed, so the pupils never change and
    the seats condition (1.2) cannot be broken by a proposal: no seats constraint.
    """
    people = min(max(capacity.passport, baseline) - staff, max_pupils + max_teachers)
    return DayRoom(people=people, pupils=None)


def add_day_capacity(
    model: cp_model.CpModel,
    day: int,
    pupils: cp_model.LinearExpr | int,
    teachers: cp_model.LinearExpr | int,
    room: DayRoom,
) -> None:
    """``pupils <= seats`` (1.2) and ``pupils + teachers <= passport - staff`` for one day."""
    if room.pupils is not None:
        model.add(pupils <= room.pupils).with_name(f"seats_{day}")
    model.add(pupils + teachers <= room.people).with_name(f"passport_{day}")
