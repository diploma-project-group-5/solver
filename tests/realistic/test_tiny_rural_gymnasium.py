"""Realistic school: a tiny rural gymnasium, grades 1-9, one class per grade, shelter for all.

Archetype ``tiny_rural_gymnasium``: 52 pupils in 9 classes (5-8 per class), 4 homeroom teachers
for grades 1-4 and 11 subject teachers who each cover 2-3 subjects across grades 5-9 (some also
teach foreign language, PE or informatics in grades 1-4), weekly hours per the sanitary limits
20/22/23/23/28/31/32/33/33 in a 5 x 7 grid. The passport (73) fits every pupil, every staff
member and every teacher at once, so the whole school can be in person; the stress is the dense
teacher-sharing graph (one format per teacher per day couples every class) and the near-full
grid of grades 8-9 (33 lessons in 35 slots).
"""

from collections import Counter
from functools import cache

import pytest

import oracle
from lad_solver.app.solve import solve
from lad_solver.domain.problem import (
    CurriculumRow,
    Params,
    Passport,
    Problem,
    SchoolClass,
    Subject,
    Teacher,
    TimetableKind,
)
from lad_solver.domain.result import Result, SolveStatus, StopReason

pytestmark = pytest.mark.slow  # real CP-SAT solves: run with -m "" (CI scenarios job)
TIME_LIMIT_S = 30.0
PARAMS = Params(time_limit_s=TIME_LIMIT_S, num_workers=2, random_seed=0)
FULL_PASSPORT = 73  # 52 pupils + 6 staff + 15 teachers
STAFF = 6

STUDENTS = {1: 6, 2: 5, 3: 7, 4: 6, 5: 5, 6: 8, 7: 6, 8: 5, 9: 4}

PRIORITY_AREA = frozenset(
    {"ukr", "lit", "flang", "math", "alg", "geo", "explore", "sci", "bio", "geog", "phys"}
    | {"chem", "hist", "civ"}
)
ALL_SUBJECTS = (
    *sorted(PRIORITY_AREA),
    *("inf", "arts", "tech", "health", "pe", "elective"),
)

# Weekly hours per grade (sum = sanitary limit of the grade).
_G1 = {"ukr": 6, "math": 4, "explore": 2, "flang": 2, "arts": 2, "tech": 1, "pe": 3}
_G2 = {**_G1, "flang": 3, "inf": 1}
_G3 = {**_G2, "math": 5}
_G5 = {
    "ukr": 4,
    "lit": 3,
    "flang": 3,
    "math": 4,
    "sci": 2,
    "hist": 2,
    "inf": 1,
    "arts": 2,
    "tech": 2,
    "health": 1,
    "pe": 3,
    "elective": 1,
}
_G6 = {**_G5, "geog": 2, "elective": 2}
_G7 = {
    "ukr": 3,
    "lit": 3,
    "flang": 3,
    "alg": 3,
    "geo": 2,
    "bio": 2,
    "geog": 2,
    "phys": 2,
    "chem": 2,
    "hist": 2,
    "inf": 1,
    "arts": 1,
    "tech": 1,
    "health": 1,
    "pe": 3,
    "elective": 1,
}
_G8 = {**_G7, "hist": 3}
_G9 = {**_G7, "civ": 1}  # history 2 + civics 1
CURRICULUM: dict[int, dict[str, int]] = {
    1: _G1,
    2: _G2,
    3: _G3,
    4: _G3,
    5: _G5,
    6: _G6,
    7: _G7,
    8: _G8,
    9: _G9,
}
SANITARY_LIMITS = {1: 20, 2: 22, 3: 23, 4: 23, 5: 28, 6: 31, 7: 32, 8: 33, 9: 33}

HOMEROOM_SUBJECTS = frozenset({"ukr", "math", "explore", "arts", "tech"})
HOMEROOM_LIMIT = 18

# Subject teachers: (teacher, weekly limit, {(subject, grades)}); several below 18 = part-time.
_UPPER = range(5, 10)
SUBJECT_TEACHERS: tuple[tuple[str, int, tuple[tuple[str, range], ...]], ...] = (
    ("t-05", 24, (("ukr", _UPPER), ("lit", range(5, 7)))),
    ("t-06", 9, (("lit", range(7, 10)),)),
    ("t-07", 27, (("flang", range(1, 10)),)),
    ("t-08", 24, (("math", range(5, 7)), ("alg", _UPPER), ("geo", _UPPER))),
    ("t-09", 18, (("phys", _UPPER), ("inf", range(1, 10)))),
    ("t-10", 18, (("sci", _UPPER), ("bio", _UPPER), ("chem", _UPPER))),
    ("t-11", 12, (("hist", _UPPER), ("civ", _UPPER))),
    ("t-12", 9, (("geog", _UPPER),)),
    ("t-13", 27, (("pe", range(1, 10)),)),
    ("t-14", 14, (("arts", _UPPER), ("elective", _UPPER))),
    ("t-15", 12, (("tech", _UPPER), ("health", _UPPER))),
)


def _class_id(grade: int) -> str:
    return f"c-{grade}a"


def _teacher_of(grade: int, subject: str) -> str:
    if grade <= 4 and subject in HOMEROOM_SUBJECTS:
        return f"t-{grade:02d}"
    for tid, _, covers in SUBJECT_TEACHERS:
        if any(s == subject and grade in grades for s, grades in covers):
            return tid
    msg = f"no teacher for grade {grade} subject {subject}"
    raise AssertionError(msg)


def build_school(passport: int = FULL_PASSPORT) -> Problem:
    classes = tuple(SchoolClass(_class_id(g), STUDENTS[g], priority=False) for g in STUDENTS)
    subjects = tuple(
        Subject(f"s-{s}", priority_area=s in PRIORITY_AREA, no_shelter=False) for s in ALL_SUBJECTS
    )
    teachers = (
        *(Teacher(f"t-{g:02d}", HOMEROOM_LIMIT) for g in range(1, 5)),
        *(Teacher(tid, limit) for tid, limit, _ in SUBJECT_TEACHERS),
    )
    rows = tuple(
        CurriculumRow(_class_id(g), f"s-{s}", _teacher_of(g, s), h)
        for g, plan in CURRICULUM.items()
        for s, h in sorted(plan.items())
    )
    return Problem(
        kind=TimetableKind.MAIN,
        school_days=(1, 2, 3, 4, 5),
        lessons_per_day=7,
        staff=STAFF,
        capacity=Passport(passport),
        classes=classes,
        subjects=subjects,
        teachers=teachers,
        rows=rows,
    )


@cache
def _school() -> Problem:
    return build_school()


@cache
def _solved() -> Result:
    return solve(_school(), PARAMS)


def test_scenario_matches_archetype() -> None:
    p = _school()
    assert sum(c.students for c in p.classes) == 52
    assert len(p.teachers) == 15
    assert p.capacity == Passport(52 + STAFF + 15)
    hours = Counter[str]()
    for r in p.rows:
        hours[r.class_id] += r.hours
    assert hours == Counter({_class_id(g): h for g, h in SANITARY_LIMITS.items()})
    load = Counter[str]()
    for r in p.rows:
        load[r.teacher_id] += r.hours
    limits = {t.id: t.weekly_limit for t in p.teachers}
    assert all(load[t] <= limits[t] for t in limits)
    assert max(load.values()) == 27


def test_builds_optimal_admissible_timetable_within_limit() -> None:
    p, result = _school(), _solved()
    assert result.status is SolveStatus.OPTIMAL
    assert result.stats.stop_reason is StopReason.COMPLETED
    assert not result.stats.quality_relaxed
    assert oracle.violations(p, result) == []
    recomputed = oracle.objective(p, result)
    assert (result.stats.objective, result.stats.primary, result.stats.secondary) == (
        recomputed.value,
        recomputed.primary,
        recomputed.secondary,
    )
    assert result.stats.wall_time_s < TIME_LIMIT_S


def test_passport_holds_every_day() -> None:
    p, result = _school(), _solved()
    assert result.status is SolveStatus.OPTIMAL
    for day in p.school_days:
        load = oracle.day_load(p, result, day)
        assert load.people <= FULL_PASSPORT, f"day {day}: {load}"


def test_every_priority_lesson_is_in_person() -> None:
    # The passport fits everyone, so the optimum puts every priority-area lesson in person.
    p, result = _school(), _solved()
    assert result.status is SolveStatus.OPTIMAL
    students = {c.id: c.students for c in p.classes}
    priority = {s.id for s in p.subjects if s.priority_area}
    bound = sum(students[r.class_id] * r.hours for r in p.rows if r.subject_id in priority)
    assert result.stats.primary == bound
    in_person = oracle.in_person_days(result)
    carrying = {(x.class_id, x.day) for x in result.lessons if x.subject_id in priority}
    assert carrying <= in_person


def test_teacher_loads_equal_curriculum_and_respect_limits() -> None:
    p, result = _school(), _solved()
    assert result.status is SolveStatus.OPTIMAL
    planned = Counter[str]()
    for r in p.rows:
        planned[r.teacher_id] += r.hours
    taught = Counter(x.teacher_id for x in result.lessons)
    assert taught == planned
    limits = {t.id: t.weekly_limit for t in p.teachers}
    assert all(n <= limits[t] for t, n in taught.items())
    # Grades 8-9 fill 33 of 35 slots: each class sits at most one lesson per slot.
    slots = Counter((x.class_id, x.day, x.lesson_no) for x in result.lessons)
    assert max(slots.values()) == 1


def test_passport_one_short_still_admissible() -> None:
    # One seat less than everyone: the solver must keep someone home on a full day.
    p = build_school(FULL_PASSPORT - 1)
    result = solve(p, PARAMS)
    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(p, result) == []
    for day in p.school_days:
        assert oracle.day_load(p, result, day).people <= FULL_PASSPORT - 1
    assert result.stats.wall_time_s < TIME_LIMIT_S
