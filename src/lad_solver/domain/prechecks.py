"""Deterministic pre-checks run before CP-SAT (ADR 2026-11-04 §2).

"Everything remote" is always admissible for capacity, so a build can only be infeasible
because of staff vs. limits or because the weekly grid cannot be packed. The rules below are
cheap necessary conditions for the latter; anything they miss is reported as ``packing``.

Malformed input (unknown ids, non-positive hours, days outside 1..6, ...) yields
``invalid_input`` conflicts only: the other rules assume referential integrity.
"""

from collections import Counter
from collections.abc import Iterator
from itertools import pairwise
from math import ceil

from lad_solver.domain.problem import AlertLimits, Problem, Teacher, TimetableKind
from lad_solver.domain.result import CapacityCondition, Conflict, ConflictRule

MIN_DAY, MAX_DAY = 1, 6
MIN_LESSONS_PER_DAY, MAX_LESSONS_PER_DAY = 1, 10
DEFAULT_ROW_DAY_CAP_MIN = 2


def row_day_cap(hours: int, available_days: int, cap_min: int = DEFAULT_ROW_DAY_CAP_MIN) -> int:
    """Max lessons of one curriculum row per day: ``max(cap_min, ceil(h / available days))``."""
    if available_days <= 0:
        return 0
    return max(cap_min, ceil(hours / available_days))


def available_days(problem: Problem, teacher: Teacher) -> tuple[int, ...]:
    """School days on which the teacher can work."""
    return tuple(d for d in problem.school_days if d not in teacher.unavailable_days)


def precheck(
    problem: Problem, *, row_day_cap_min: int = DEFAULT_ROW_DAY_CAP_MIN
) -> tuple[Conflict, ...]:
    """Return every pre-check conflict; an empty tuple means "hand over to CP-SAT"."""
    invalid = tuple(_invalid_input(problem))
    if invalid:
        return invalid
    return (
        *_capacity(problem),
        *_class_hours(problem),
        *_teacher_hours(problem),
        *_row_caps(problem, row_day_cap_min),
    )


_BAD = ConflictRule.INVALID_INPUT


def _invalid_input(problem: Problem) -> Iterator[Conflict]:
    yield from _invalid_frame(problem)
    yield from _duplicate_ids(problem)
    yield from _invalid_entities(problem)
    yield from _invalid_rows(problem)


def _invalid_frame(problem: Problem) -> Iterator[Conflict]:
    days = problem.school_days
    out_of_range = any(not MIN_DAY <= d <= MAX_DAY for d in days)
    unordered = any(a >= b for a, b in pairwise(days))
    if not days or out_of_range or unordered:
        yield Conflict(_BAD)
    if not MIN_LESSONS_PER_DAY <= problem.lessons_per_day <= MAX_LESSONS_PER_DAY:
        yield Conflict(_BAD)
    if problem.staff < 0:
        yield Conflict(_BAD)
    if isinstance(problem.capacity, AlertLimits) and problem.capacity.seats < 0:
        yield Conflict(_BAD)  # zero seats are fine: every class is remote


def _duplicate_ids(problem: Problem) -> Iterator[Conflict]:
    for class_id, n in Counter(c.id for c in problem.classes).items():
        if n > 1:
            yield Conflict(_BAD, class_id=class_id)
    for subject_id, n in Counter(s.id for s in problem.subjects).items():
        if n > 1:
            yield Conflict(_BAD, subject_id=subject_id)
    for teacher_id, n in Counter(t.id for t in problem.teachers).items():
        if n > 1:
            yield Conflict(_BAD, teacher_id=teacher_id)


def _invalid_entities(problem: Problem) -> Iterator[Conflict]:
    for c in problem.classes:
        if c.students < 0:
            yield Conflict(_BAD, class_id=c.id)
    for t in problem.teachers:
        if t.weekly_limit < 0 or any(not MIN_DAY <= d <= MAX_DAY for d in t.unavailable_days):
            yield Conflict(_BAD, teacher_id=t.id)


def _invalid_rows(problem: Problem) -> Iterator[Conflict]:
    class_ids = {c.id for c in problem.classes}
    subject_ids = {s.id for s in problem.subjects}
    teacher_ids = {t.id for t in problem.teachers}
    for r in problem.rows:
        if r.class_id not in class_ids or r.hours <= 0:
            yield Conflict(_BAD, class_id=r.class_id)
        if r.subject_id not in subject_ids:
            yield Conflict(_BAD, subject_id=r.subject_id)
        if r.teacher_id not in teacher_ids:
            yield Conflict(_BAD, teacher_id=r.teacher_id)


def _capacity(problem: Problem) -> Iterator[Conflict]:
    capacity = problem.capacity
    if problem.staff > capacity.passport:
        yield Conflict(ConflictRule.STAFF_OVER_PASSPORT, condition=CapacityCondition.PASSPORT)


def _class_hours(problem: Problem) -> Iterator[Conflict]:
    hours = Counter[str]()
    for r in problem.rows:
        hours[r.class_id] += r.hours
    slots = problem.lessons_per_day * len(problem.school_days)
    rule = (
        ConflictRule.ALERT_BELLS
        if problem.kind is TimetableKind.ALERT
        else ConflictRule.CLASS_HOURS_EXCEED_SLOTS
    )
    for c in problem.classes:
        if hours[c.id] > slots:
            yield Conflict(rule, class_id=c.id)


def _teacher_hours(problem: Problem) -> Iterator[Conflict]:
    load = Counter[str]()
    for r in problem.rows:
        load[r.teacher_id] += r.hours
    for t in problem.teachers:
        if load[t.id] == 0:
            continue
        if load[t.id] > t.weekly_limit:
            yield Conflict(ConflictRule.TEACHER_WEEKLY_LIMIT, teacher_id=t.id)
        days = available_days(problem, t)
        if not days:
            yield Conflict(ConflictRule.TEACHER_UNAVAILABLE_ALL_DAYS, teacher_id=t.id)
        elif load[t.id] > problem.lessons_per_day * len(days):
            yield Conflict(ConflictRule.TEACHER_HOURS_EXCEED_AVAILABLE_SLOTS, teacher_id=t.id)


def _row_caps(problem: Problem, cap_min: int) -> Iterator[Conflict]:
    teachers = {t.id: t for t in problem.teachers}
    for r in problem.rows:
        n_days = len(available_days(problem, teachers[r.teacher_id]))
        if n_days == 0:
            continue  # reported as teacher_unavailable_all_days
        per_day = min(row_day_cap(r.hours, n_days, cap_min), problem.lessons_per_day)
        if r.hours > per_day * n_days:
            yield Conflict(
                ConflictRule.ROW_DAY_CAP,
                class_id=r.class_id,
                subject_id=r.subject_id,
                teacher_id=r.teacher_id,
            )
