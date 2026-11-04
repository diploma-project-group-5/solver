"""Independent check of an engine ``Result`` against its ``Problem`` (test-only).

Temporary until the validator-lib CLI exists (ADR 2026-11-04 §5): production rules live in
validator-lib only; this oracle never gives a verdict to users. It is deliberately written
without any engine code so that extraction bugs (day shifts, clashes, wrong formats) surface.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

from lad_solver.domain.problem import AlertLimits, Problem, TimetableKind
from lad_solver.domain.result import Result, SolveStatus


@dataclass(frozen=True, slots=True)
class Objective:
    primary: int
    secondary: int
    value: int


def violations(problem: Problem, result: Result) -> list[str]:
    """Every broken rule as a human-readable line; empty means the result is admissible."""
    if result.status not in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}:
        return [f"status {result.status} has no timetable"]
    return [
        *_structure(problem, result),
        *_formats(problem, result),
        *_capacity(problem, result),
    ]


def objective(problem: Problem, result: Result) -> Objective:
    students = {c.id: c.students for c in problem.classes}
    flagged = {c.id for c in problem.classes if c.priority}
    priority_subjects = {s.id for s in problem.subjects if s.priority_area}
    in_person = {(f.class_id, f.day) for f in result.day_formats if f.in_person}
    primary = sum(
        students[x.class_id]
        for x in result.lessons
        if x.subject_id in priority_subjects and (x.class_id, x.day) in in_person
    )
    secondary = sum(1 for c, _ in in_person if c in flagged)
    weight = len(flagged) * len(problem.school_days) + 1
    return Objective(primary, secondary, weight * primary + secondary)


def _structure(problem: Problem, result: Result) -> list[str]:
    out: list[str] = []
    expected = Counter[tuple[str, str, str]]()
    for r in problem.rows:
        expected[r.class_id, r.subject_id, r.teacher_id] += r.hours
    actual = Counter((x.class_id, x.subject_id, x.teacher_id) for x in result.lessons)
    if actual != expected:
        out.append(f"plan mismatch: expected {dict(expected)}, got {dict(actual)}")
    class_slots = Counter((x.class_id, x.day, x.lesson_no) for x in result.lessons)
    teacher_slots = Counter((x.teacher_id, x.day, x.lesson_no) for x in result.lessons)
    out.extend(f"class clash {k}" for k, n in class_slots.items() if n > 1)
    out.extend(f"teacher clash {k}" for k, n in teacher_slots.items() if n > 1)
    unavailable = {t.id: t.unavailable_days for t in problem.teachers}
    limits = {t.id: t.weekly_limit for t in problem.teachers}
    for x in result.lessons:
        if not 1 <= x.lesson_no <= problem.lessons_per_day:
            out.append(f"lesson number out of range {x}")
        if x.day not in problem.school_days:
            out.append(f"not a school day {x}")
        if x.day in unavailable[x.teacher_id]:
            out.append(f"teacher unavailable {x}")
    load = Counter(x.teacher_id for x in result.lessons)
    out.extend(f"weekly limit {t}" for t, n in load.items() if n > limits[t])
    return out


def _formats(problem: Problem, result: Result) -> list[str]:
    out: list[str] = []
    keys = Counter((f.class_id, f.day) for f in result.day_formats)
    wanted = {(c.id, d) for c in problem.classes for d in problem.school_days}
    if set(keys) != wanted or any(n > 1 for n in keys.values()):
        out.append("day formats do not cover every class x school day exactly once")
    in_person = {(f.class_id, f.day) for f in result.day_formats if f.in_person}
    busy = {(x.class_id, x.day) for x in result.lessons}
    out.extend(f"empty in-person day {k}" for k in in_person - busy)
    formats: defaultdict[tuple[str, int], set[bool]] = defaultdict(set)
    for x in result.lessons:
        formats[x.teacher_id, x.day].add((x.class_id, x.day) in in_person)
    out.extend(f"teacher in two formats {k}" for k, v in formats.items() if len(v) > 1)
    if problem.kind is TimetableKind.ALERT:
        outside = {s.id for s in problem.subjects if s.no_shelter}
        out.extend(
            f"no-shelter lesson on in-person day {x}"
            for x in result.lessons
            if x.subject_id in outside and (x.class_id, x.day) in in_person
        )
    return out


@dataclass(frozen=True, slots=True)
class DayLoad:
    pupils: int  # students of in-person classes (the 1.2 seats load)
    teachers: int  # distinct teachers with an in-person lesson
    people: int  # pupils + staff + teachers (the passport load)


def day_load(problem: Problem, result: Result, day: int) -> DayLoad:
    students = {c.id: c.students for c in problem.classes}
    in_person = {(f.class_id, f.day) for f in result.day_formats if f.in_person}
    pupils = sum(students[c] for c, d in in_person if d == day)
    teachers = {
        x.teacher_id for x in result.lessons if (x.class_id, x.day) in in_person and x.day == day
    }
    return DayLoad(pupils, len(teachers), pupils + problem.staff + len(teachers))


def _capacity(problem: Problem, result: Result) -> list[str]:
    out: list[str] = []
    cap = problem.capacity
    for d in problem.school_days:
        load = day_load(problem, result, d)
        if isinstance(cap, AlertLimits) and load.pupils > cap.seats:
            out.append(f"day {d}: seats {load.pupils} > {cap.seats}")
        if load.people > cap.passport:
            out.append(f"day {d}: passport {load.people} > {cap.passport}")
    return out
