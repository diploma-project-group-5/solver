"""Realistic alert timetable (model 1.2): equipped seats vs. shelter passport.

A ~280-pupil school, grades 1-11 with one class each, built for the alert bells. PE and
technologies cannot be held in the shelter, so they force remote days. Variant ``a`` makes the
seats condition binding, variant ``b`` the passport condition; variant ``c`` uses short alert
bells (6 lessons) that cannot hold the 31-33 weekly hours of grades 6-11.

Every solved result is checked by the independent ``oracle``; the per-day checks below restate
the binding condition explicitly.
"""

from collections import Counter
from functools import cache

import pytest

import oracle
from lad_solver.app.solve import solve
from lad_solver.domain.problem import (
    AlertLimits,
    CurriculumRow,
    Params,
    Problem,
    SchoolClass,
    Subject,
    Teacher,
    TimetableKind,
)
from lad_solver.domain.result import ConflictRule, Result, SolveStatus, StopReason

TIME_LIMIT_S = 30.0
PARAMS = Params(time_limit_s=TIME_LIMIT_S, num_workers=2, random_seed=0)
DAYS = (1, 2, 3, 4, 5)
STAFF = 9
TEACHER_LIMIT = 24
WALL_SLACK_S = 5.0  # process start-up, model building and extraction on a busy CI host

# Variant -> (seats, passport, lessons per day).
VARIANTS: dict[str, tuple[int, int, int]] = {
    "a": (120, 200, 7),  # seats binding
    "b": (200, 150, 7),  # passport binding
    "c": (120, 200, 6),  # short alert bells
}

# Priority areas per MON: mathematics, science, language-literature, civic & history.
PRIORITY_SUBJECTS = (
    "s-ukr",
    "s-ukrlit",
    "s-eng",
    "s-wlit",
    "s-math",
    "s-nature",
    "s-bio",
    "s-chem",
    "s-phys",
    "s-geo",
    "s-hist",
)
OTHER_SUBJECTS = ("s-inf", "s-art", "s-health", "s-defense")
NO_SHELTER_SUBJECTS = ("s-pe", "s-tech")

# Pupils per class (total 281).
STUDENTS: dict[int, int] = {
    1: 27,
    2: 26,
    3: 25,
    4: 24,
    5: 27,
    6: 26,
    7: 25,
    8: 26,
    9: 25,
    10: 23,
    11: 27,
}

# Weekly hours per grade and subject (subject -> hours).
_PRIMARY_BASE = {"s-ukr": 7, "s-nature": 3, "s-art": 2, "s-pe": 3, "s-tech": 1}
CURRICULUM: dict[int, dict[str, int]] = {
    1: {**_PRIMARY_BASE, "s-math": 4, "s-eng": 2},  # 22
    2: {**_PRIMARY_BASE, "s-math": 4, "s-eng": 3},  # 23
    3: {**_PRIMARY_BASE, "s-math": 5, "s-eng": 3, "s-inf": 1},  # 25
    4: {**_PRIMARY_BASE, "s-math": 5, "s-eng": 3, "s-inf": 1},  # 25
    5: {  # 28
        "s-ukr": 4,
        "s-ukrlit": 2,
        "s-eng": 3,
        "s-wlit": 2,
        "s-math": 5,
        "s-nature": 2,
        "s-hist": 1,
        "s-health": 1,
        "s-inf": 1,
        "s-art": 2,
        "s-tech": 2,
        "s-pe": 3,
    },
    6: {  # 31
        "s-ukr": 4,
        "s-ukrlit": 2,
        "s-eng": 3,
        "s-wlit": 2,
        "s-math": 5,
        "s-nature": 2,
        "s-hist": 2,
        "s-geo": 2,
        "s-health": 1,
        "s-inf": 1,
        "s-art": 2,
        "s-tech": 2,
        "s-pe": 3,
    },
    7: {  # 32
        "s-ukr": 3,
        "s-ukrlit": 2,
        "s-eng": 3,
        "s-wlit": 2,
        "s-math": 5,
        "s-bio": 2,
        "s-geo": 2,
        "s-phys": 2,
        "s-chem": 2,
        "s-hist": 2,
        "s-health": 1,
        "s-inf": 1,
        "s-art": 1,
        "s-tech": 1,
        "s-pe": 3,
    },
    8: {  # 33
        "s-ukr": 3,
        "s-ukrlit": 2,
        "s-eng": 3,
        "s-wlit": 2,
        "s-math": 5,
        "s-bio": 2,
        "s-geo": 2,
        "s-phys": 3,
        "s-chem": 2,
        "s-hist": 2,
        "s-health": 1,
        "s-inf": 1,
        "s-art": 1,
        "s-tech": 1,
        "s-pe": 3,
    },
    9: {  # 33
        "s-ukr": 3,
        "s-ukrlit": 2,
        "s-eng": 3,
        "s-wlit": 2,
        "s-math": 5,
        "s-bio": 2,
        "s-geo": 2,
        "s-phys": 3,
        "s-chem": 2,
        "s-hist": 3,
        "s-health": 1,
        "s-inf": 1,
        "s-tech": 1,
        "s-pe": 3,
    },
    **{
        g: {  # 33
            "s-ukr": 2,
            "s-ukrlit": 2,
            "s-eng": 3,
            "s-wlit": 2,
            "s-math": 5,
            "s-bio": 2,
            "s-geo": 2,
            "s-phys": 3,
            "s-chem": 2,
            "s-hist": 3,
            "s-defense": 2,
            "s-inf": 2,
            "s-pe": 3,
        }
        for g in (10, 11)
    },
}


def _class_id(grade: int) -> str:
    return f"c-{grade}a"


def _teacher_id(grade: int, subject_id: str) -> str:
    """Who teaches ``subject_id`` in ``grade`` (19 teachers, 4 of them homeroom)."""
    homeroom = subject_id in {"s-ukr", "s-math", "s-nature", "s-art"}
    if grade <= 4 and homeroom:
        return f"t-{grade:02d}"  # t-01..t-04: primary homeroom teachers
    split = {
        "s-ukr": ("t-05", "t-06", 8),  # grades 5-8 / 9-11
        "s-ukrlit": ("t-05", "t-06", 8),
        "s-eng": ("t-07", "t-08", 6),  # grades 1-6 / 7-11
        "s-math": ("t-10", "t-11", 8),
        "s-pe": ("t-18", "t-19", 6),
    }
    if subject_id in split:
        low, high, last_low = split[subject_id]
        return low if grade <= last_low else high
    single = {
        "s-wlit": "t-09",
        "s-nature": "t-12",  # biology & chemistry teacher also teaches science in 5-6
        "s-bio": "t-12",
        "s-chem": "t-12",
        "s-geo": "t-13",
        "s-health": "t-13",
        "s-phys": "t-14",
        "s-hist": "t-15",
        "s-defense": "t-15",
        "s-inf": "t-16",
        "s-art": "t-17",  # art & technologies
        "s-tech": "t-17",
    }
    return single[subject_id]


def _rows() -> tuple[CurriculumRow, ...]:
    return tuple(
        CurriculumRow(_class_id(g), s, _teacher_id(g, s), h)
        for g, plan in sorted(CURRICULUM.items())
        for s, h in plan.items()
    )


def build_problem(variant: str) -> Problem:
    seats, passport, lessons_per_day = VARIANTS[variant]
    rows = _rows()
    teacher_ids = sorted({r.teacher_id for r in rows})
    subjects = (
        *(Subject(s, priority_area=True, no_shelter=False) for s in PRIORITY_SUBJECTS),
        *(Subject(s, priority_area=False, no_shelter=False) for s in OTHER_SUBJECTS),
        *(Subject(s, priority_area=False, no_shelter=True) for s in NO_SHELTER_SUBJECTS),
    )
    return Problem(
        kind=TimetableKind.ALERT,
        school_days=DAYS,
        lessons_per_day=lessons_per_day,
        staff=STAFF,
        capacity=AlertLimits(seats=seats, passport=passport),
        classes=tuple(SchoolClass(_class_id(g), n, priority=False) for g, n in STUDENTS.items()),
        subjects=subjects,
        teachers=tuple(Teacher(t, TEACHER_LIMIT) for t in teacher_ids),
        rows=rows,
    )


@cache
def _solved(variant: str) -> tuple[Problem, Result]:
    p = build_problem(variant)
    return p, solve(p, PARAMS)


def test_scenario_is_realistic() -> None:
    p = build_problem("a")
    hours = Counter[str]()
    load = Counter[str]()
    for r in p.rows:
        hours[r.class_id] += r.hours
        load[r.teacher_id] += r.hours
    assert sum(c.students for c in p.classes) == 281
    assert all(23 <= c.students <= 29 for c in p.classes)
    assert len(p.teachers) == 19
    assert max(load.values()) <= TEACHER_LIMIT
    assert all(31 <= hours[_class_id(g)] <= 33 for g in range(6, 12))
    assert all(hours[_class_id(g)] <= 6 * len(DAYS) for g in range(1, 6))


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["a", "b"])
def test_alert_build_is_feasible_admissible_and_in_time(variant: str) -> None:
    p, result = _solved(variant)
    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(p, result) == []
    assert result.stats.wall_time_s <= TIME_LIMIT_S + WALL_SLACK_S
    assert result.stats.first_solution_s is not None
    assert result.stats.first_solution_s < TIME_LIMIT_S
    assert result.stats.primary is not None
    assert result.stats.primary > 0  # some priority lessons are in person, not "all remote"
    assert oracle.in_person_days(result)


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["a", "b"])
def test_both_capacity_conditions_hold_every_day(variant: str) -> None:
    p, result = _solved(variant)
    cap = p.capacity
    assert isinstance(cap, AlertLimits)
    for d in p.school_days:
        load = oracle.day_load(p, result, d)
        assert load.pupils <= cap.seats, (d, load)
        assert load.people <= cap.passport, (d, load)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("variant", "binding"),
    [("a", "seats"), ("b", "passport")],
    ids=["a-seats", "b-passport"],
)
def test_binding_condition_is_tight(variant: str, binding: str) -> None:
    """On the busiest day no further class could fit under the binding condition."""
    p, result = _solved(variant)
    cap = p.capacity
    assert isinstance(cap, AlertLimits)
    smallest = min(c.students for c in p.classes)
    loads = [oracle.day_load(p, result, d) for d in p.school_days]
    if binding == "seats":
        assert max(x.pupils for x in loads) > cap.seats - smallest
        # The passport has slack every day: it is not what limits variant a.
        assert all(x.people <= cap.passport - smallest for x in loads)
    else:
        # Adding a class adds its pupils plus at least zero new teachers.
        assert max(x.people for x in loads) > cap.passport - smallest
        assert all(x.pupils <= cap.seats - smallest for x in loads)


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["a", "b"])
def test_pe_and_tech_only_on_remote_days(variant: str) -> None:
    p, result = _solved(variant)
    in_person = oracle.in_person_days(result)
    outside = {s.id for s in p.subjects if s.no_shelter}
    assert all(
        (x.class_id, x.day) not in in_person for x in result.lessons if x.subject_id in outside
    )
    pe_days: dict[str, set[int]] = {}
    for x in result.lessons:
        if x.subject_id == "s-pe":
            pe_days.setdefault(x.class_id, set()).add(x.day)
    assert set(pe_days) == {c.id for c in p.classes}
    for c in p.classes:
        remote = [d for d in p.school_days if (c.id, d) not in in_person]
        # PE is 3 h with a row-day cap of 2, so it needs at least two (remote) days.
        assert len(pe_days[c.id]) >= 2, c.id
        assert len(remote) >= 2, c.id


def test_short_alert_bells_are_rejected_at_precheck() -> None:
    p = build_problem("c")
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.INFEASIBLE
    assert result.stats.stop_reason is StopReason.PRECHECK
    assert result.lessons == ()
    assert {c.rule for c in result.conflicts} == {ConflictRule.ALERT_BELLS}
    assert {c.class_id for c in result.conflicts} == {_class_id(g) for g in range(6, 12)}
    assert result.stats.wall_time_s < 1.0
