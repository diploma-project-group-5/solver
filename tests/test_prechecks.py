"""Pre-checks: one case per rule (ADR 2026-11-04 §2)."""

from dataclasses import replace

import pytest

from builders import OTHER, OUTSIDE, PRIO, klass, problem, row, teacher
from lad_solver.domain.prechecks import precheck, row_day_cap
from lad_solver.domain.problem import AlertLimits, Passport, Problem, TimetableKind
from lad_solver.domain.result import CapacityCondition, Conflict, ConflictRule

R = ConflictRule


def _base(**overrides: object) -> Problem:
    base = problem(
        classes=(klass("a", 20),),
        teachers=(teacher("t1", limit=10),),
        rows=(row("a", PRIO, "t1", 4),),
        days=(1, 2),
        lessons_per_day=3,
        staff=5,
        capacity=Passport(100),
    )
    return replace(base, **overrides)


def test_valid_problem_has_no_conflicts() -> None:
    assert precheck(_base()) == ()


CASES: list[tuple[str, Problem, tuple[Conflict, ...]]] = [
    (
        "staff over passport (1.1)",
        _base(staff=101),
        (Conflict(R.STAFF_OVER_PASSPORT, condition=CapacityCondition.PASSPORT),),
    ),
    (
        "staff over passport (1.2)",
        _base(kind=TimetableKind.ALERT, capacity=AlertLimits(seats=50, passport=4)),
        (Conflict(R.STAFF_OVER_PASSPORT, condition=CapacityCondition.PASSPORT),),
    ),
    (
        "negative seats (1.2) are malformed input",
        _base(kind=TimetableKind.ALERT, capacity=AlertLimits(seats=-1, passport=100)),
        (Conflict(R.INVALID_INPUT),),
    ),
    (
        "zero seats (1.2) are legitimate: every class remote",
        _base(kind=TimetableKind.ALERT, capacity=AlertLimits(seats=0, passport=100)),
        (),
    ),
    (
        "class hours exceed slots",
        _base(
            rows=(row("a", PRIO, "t1", 4), row("a", OTHER, "t2", 3)),
            teachers=(teacher("t1", 10), teacher("t2", 10)),
        ),
        (Conflict(R.CLASS_HOURS_EXCEED_SLOTS, class_id="a"),),
    ),
    (
        "alert bells: class hours exceed alert slots",
        _base(
            kind=TimetableKind.ALERT,
            capacity=AlertLimits(50, 100),
            lessons_per_day=1,
            rows=(row("a", PRIO, "t1", 2), row("a", OTHER, "t2", 1)),
            teachers=(teacher("t1", 10), teacher("t2", 10)),
        ),
        (Conflict(R.ALERT_BELLS, class_id="a"),),
    ),
    (
        "teacher weekly limit",
        _base(teachers=(teacher("t1", limit=3),)),
        (Conflict(R.TEACHER_WEEKLY_LIMIT, teacher_id="t1"),),
    ),
    (
        "teacher hours exceed available slots",
        _base(
            classes=(klass("a", 20), klass("b", 20)),
            teachers=(teacher("t1", 10, unavailable=(2,)),),
            rows=(row("a", PRIO, "t1", 2), row("b", PRIO, "t1", 2)),
        ),
        (Conflict(R.TEACHER_HOURS_EXCEED_AVAILABLE_SLOTS, teacher_id="t1"),),
    ),
    (
        "teacher unavailable on all school days",
        _base(teachers=(teacher("t1", 10, unavailable=(1, 2)),)),
        (Conflict(R.TEACHER_UNAVAILABLE_ALL_DAYS, teacher_id="t1"),),
    ),
    (
        "row does not fit: one available day, more hours than lessons per day",
        _base(teachers=(teacher("t1", 10, unavailable=(2,)),)),
        (
            Conflict(R.TEACHER_HOURS_EXCEED_AVAILABLE_SLOTS, teacher_id="t1"),
            Conflict(R.ROW_DAY_CAP, class_id="a", subject_id=PRIO, teacher_id="t1"),
        ),
    ),
]


@pytest.mark.parametrize(("name", "case", "expected"), CASES, ids=[c[0] for c in CASES])
def test_rule(name: str, case: Problem, expected: tuple[Conflict, ...]) -> None:
    assert name
    assert precheck(case) == expected


INVALID: list[tuple[str, Problem]] = [
    ("unknown class in row", _base(rows=(row("zzz", PRIO, "t1", 1),))),
    ("unknown subject in row", _base(rows=(row("a", "zzz", "t1", 1),))),
    ("unknown teacher in row", _base(rows=(row("a", PRIO, "zzz", 1),))),
    ("zero hours", _base(rows=(row("a", PRIO, "t1", 0),))),
    ("day outside 1..6", _base(school_days=(1, 7))),
    ("days not ascending", _base(school_days=(2, 1))),
    ("no school days", _base(school_days=())),
    ("lessons per day 0", _base(lessons_per_day=0)),
    ("lessons per day 11", _base(lessons_per_day=11)),
    ("negative staff", _base(staff=-1)),
    ("negative students", _base(classes=(klass("a", -1),))),
    ("negative weekly limit", _base(teachers=(teacher("t1", -1),))),
    ("unavailable day outside 1..6", _base(teachers=(teacher("t1", 10, unavailable=(0,)),))),
    ("duplicate class id", _base(classes=(klass("a", 1), klass("a", 2)))),
    ("negative passport (1.1)", _base(staff=0, capacity=Passport(-1))),
    (
        "negative passport (1.2)",
        _base(staff=0, kind=TimetableKind.ALERT, capacity=AlertLimits(seats=0, passport=-1)),
    ),
    ("alert timetable with a 1.1 capacity", _base(kind=TimetableKind.ALERT)),
    ("main timetable with a 1.2 capacity", _base(capacity=AlertLimits(seats=50, passport=100))),
]


@pytest.mark.parametrize(("name", "case"), INVALID, ids=[c[0] for c in INVALID])
def test_invalid_input(name: str, case: Problem) -> None:
    conflicts = precheck(case)
    assert conflicts, name
    assert {c.rule for c in conflicts} == {R.INVALID_INPUT}


def test_no_shelter_subject_is_not_a_precheck_conflict() -> None:
    # "All remote" is always admissible, so no_shelter can never make a build infeasible.
    case = _base(
        kind=TimetableKind.ALERT, capacity=AlertLimits(50, 100), rows=(row("a", OUTSIDE, "t1", 4),)
    )
    assert precheck(case) == ()


@pytest.mark.parametrize(
    ("hours", "days", "cap_min", "expected"),
    [(1, 5, 2, 2), (5, 5, 2, 2), (7, 3, 2, 3), (9, 2, 2, 5), (3, 5, 1, 1), (4, 0, 2, 0)],
)
def test_row_day_cap(hours: int, days: int, cap_min: int, expected: int) -> None:
    assert row_day_cap(hours, days, cap_min) == expected
