"""Compact builders for hand-written test problems (test-only)."""

from lad_solver.domain.problem import (
    Capacity,
    CurriculumRow,
    Passport,
    Problem,
    SchoolClass,
    Subject,
    Teacher,
    TimetableKind,
)

PRIO = "prio"  # subject of a priority educational area
OTHER = "other"  # subject outside priority areas
OUTSIDE = "outside"  # "not held in the shelter", outside priority areas


def subjects() -> tuple[Subject, ...]:
    return (
        Subject(PRIO, priority_area=True, no_shelter=False),
        Subject(OTHER, priority_area=False, no_shelter=False),
        Subject(OUTSIDE, priority_area=False, no_shelter=True),
    )


def problem(  # noqa: PLR0913 - a keyword-only builder for readable fixtures
    *,
    classes: tuple[SchoolClass, ...],
    teachers: tuple[Teacher, ...],
    rows: tuple[CurriculumRow, ...],
    capacity: Capacity | None = None,
    kind: TimetableKind = TimetableKind.MAIN,
    days: tuple[int, ...] = (1, 2),
    lessons_per_day: int = 3,
    staff: int = 0,
) -> Problem:
    return Problem(
        kind=kind,
        school_days=days,
        lessons_per_day=lessons_per_day,
        staff=staff,
        capacity=capacity if capacity is not None else Passport(10_000),
        classes=classes,
        subjects=subjects(),
        teachers=teachers,
        rows=rows,
    )


def row(class_id: str, subject_id: str, teacher_id: str, hours: int) -> CurriculumRow:
    return CurriculumRow(class_id, subject_id, teacher_id, hours)


def klass(class_id: str, students: int, *, priority: bool = False) -> SchoolClass:
    return SchoolClass(class_id, students, priority)


def teacher(teacher_id: str, limit: int = 30, unavailable: tuple[int, ...] = ()) -> Teacher:
    return Teacher(teacher_id, limit, frozenset(unavailable))
