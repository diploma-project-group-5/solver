"""Realistic school: senior profile lyceum 10-11 with a zero-slack weekly grid.

Grades 10-11, four parallels each (8 classes, 226 students), Mon-Fri x 7 lessons = 35 slots.
Variant a: the 33-hour plan with profile algebra (5 h) gives exactly 35 h per class, so every
slot of every class is filled. Heavy teachers carry 24-27 h (1.3-1.5 rates, limit 27).
Variant b: one more physics hour (36 h) must be rejected by the pre-checks for every class.
Capacity: passport 126, about half of the 251 people (students + 10 staff + 15 teachers).
"""

from collections import Counter

import pytest

import oracle
from lad_solver.app.solve import solve
from lad_solver.domain.prechecks import precheck
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
from lad_solver.domain.result import ConflictRule, Result, SolveStatus, StopReason

DAYS = (1, 2, 3, 4, 5)
LESSONS_PER_DAY = 7
STAFF = 10
PASSPORT = 126
TEACHER_LIMIT = 27
TIME_LIMIT_S = 30.0
PARAMS = Params(time_limit_s=TIME_LIMIT_S, num_workers=2, random_seed=0)

G10 = ("c-10a", "c-10b", "c-10c", "c-10d")
G11 = ("c-11a", "c-11b", "c-11c", "c-11d")
CLASS_IDS = G10 + G11
STUDENTS = dict(zip(CLASS_IDS, (30, 29, 28, 27, 30, 28, 27, 27), strict=True))  # 226

# Subject id -> weekly hours for grades 10-11 (33-h plan, algebra raised to the profile 5 h).
PLAN_A: dict[str, int] = {
    "s-ukr-lang": 2,
    "s-ukr-lit": 2,
    "s-for-lit": 1,
    "s-eng": 3,
    "s-hist-ua": 2,
    "s-hist-world": 1,
    "s-civics": 1,
    "s-alg": 5,
    "s-geom": 2,
    "s-bio": 2,
    "s-geo": 1,
    "s-phys": 3,
    "s-chem": 2,
    "s-inf": 2,
    "s-pe": 3,
    "s-defence": 2,
    "s-arts": 1,
}
PLAN_B = {**PLAN_A, "s-phys": 4}  # 36 h: one hour more than the grid holds

PRIORITY_AREA = frozenset({"s-alg", "s-geom", "s-phys", "s-chem", "s-bio", "s-inf"})
NO_SHELTER = frozenset({"s-pe"})  # irrelevant for a main (1.1) timetable

# Teaching assignment: (teacher, subjects, classes). Every (class, subject) pair is covered once.
_UKR = ("s-ukr-lang", "s-ukr-lit", "s-for-lit")
_HIST = ("s-hist-ua", "s-hist-world", "s-civics")
_MATH = ("s-alg", "s-geom", "s-inf")
ASSIGNMENT: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("t-01", _MATH, ("c-10a", "c-10b", "c-10c")),  # 27
    ("t-02", _MATH, ("c-10d", "c-11a", "c-11b")),  # 27
    ("t-03", _MATH, ("c-11c", "c-11d")),  # 18 (+ physics below)
    ("t-03", ("s-phys",), ("c-11c", "c-11d")),  # +6 = 24 (26 in variant b)
    ("t-04", _UKR, ("c-10a", "c-10b", "c-10c", "c-10d", "c-11a")),  # 25
    ("t-05", _UKR, ("c-11b", "c-11c", "c-11d")),  # 15
    ("t-06", ("s-eng",), CLASS_IDS),  # 24
    ("t-07", _HIST, ("c-10a", "c-10b", "c-10c", "c-10d", "c-11a", "c-11b")),  # 24
    ("t-08", _HIST, ("c-11c", "c-11d")),  # 8
    ("t-09", ("s-phys",), ("c-10a", "c-10b", "c-10c", "c-10d", "c-11a", "c-11b")),  # 18
    ("t-10", ("s-bio",), CLASS_IDS),  # 16 (+ chemistry below)
    ("t-10", ("s-chem",), G10),  # +8 = 24
    ("t-11", ("s-chem",), G11),  # 8 (+ geography below)
    ("t-11", ("s-geo",), CLASS_IDS),  # +8 = 16
    ("t-12", ("s-pe",), CLASS_IDS),  # 24
    ("t-13", ("s-defence",), CLASS_IDS),  # 16
    ("t-14", ("s-arts",), CLASS_IDS),  # 8
)
TEACHER_IDS = tuple(sorted({t for t, _, _ in ASSIGNMENT} | {"t-15"}))  # t-15: on leave, no rows


def build(plan: dict[str, int]) -> Problem:
    rows = tuple(
        CurriculumRow(class_id, subject_id, teacher_id, plan[subject_id])
        for teacher_id, subject_ids, class_ids in ASSIGNMENT
        for class_id in class_ids
        for subject_id in subject_ids
    )
    return Problem(
        kind=TimetableKind.MAIN,
        school_days=DAYS,
        lessons_per_day=LESSONS_PER_DAY,
        staff=STAFF,
        capacity=Passport(PASSPORT),
        classes=tuple(SchoolClass(c, STUDENTS[c], priority=False) for c in CLASS_IDS),
        subjects=tuple(
            Subject(s, priority_area=s in PRIORITY_AREA, no_shelter=s in NO_SHELTER) for s in PLAN_A
        ),
        teachers=tuple(Teacher(t, TEACHER_LIMIT) for t in TEACHER_IDS),
        rows=rows,
    )


def test_scenario_is_the_intended_school() -> None:
    p = build(PLAN_A)
    class_hours = Counter[str]()
    teacher_hours = Counter[str]()
    for r in p.rows:
        class_hours[r.class_id] += r.hours
        teacher_hours[r.teacher_id] += r.hours
    assert dict(class_hours) == dict.fromkeys(CLASS_IDS, LESSONS_PER_DAY * len(DAYS))
    assert len({(r.class_id, r.subject_id) for r in p.rows}) == len(p.rows)  # one teacher each
    assert sum(STUDENTS.values()) == 226
    assert sorted(teacher_hours.values(), reverse=True)[:4] == [27, 27, 25, 24]
    assert max(teacher_hours.values()) <= TEACHER_LIMIT
    assert precheck(p) == ()  # 35 h in 35 slots: on the boundary, still admissible


def test_one_extra_hour_is_rejected_at_precheck_for_every_class() -> None:
    p = build(PLAN_B)
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.INFEASIBLE
    assert result.stats.stop_reason is StopReason.PRECHECK  # CP-SAT not started
    assert result.lessons == ()
    assert result.day_formats == ()
    assert {c.rule for c in result.conflicts} == {ConflictRule.CLASS_HOURS_EXCEED_SLOTS}
    assert sorted(c.class_id or "" for c in result.conflicts) == sorted(CLASS_IDS)
    assert result.stats.wall_time_s < 1.0


@pytest.fixture(scope="module")
def full_grid() -> tuple[Problem, Result]:
    p = build(PLAN_A)
    return p, solve(p, PARAMS)


@pytest.mark.slow
def test_full_grid_is_solved_within_the_limit_and_passes_the_oracle(
    full_grid: tuple[Problem, Result],
) -> None:
    p, result = full_grid
    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(p, result) == []
    assert not result.stats.quality_relaxed
    assert result.stats.wall_time_s < TIME_LIMIT_S + 2.0
    assert result.stats.first_solution_s is not None
    assert result.stats.objective == oracle.objective(p, result).value


@pytest.mark.slow
def test_every_class_has_seven_consecutive_lessons_every_day(
    full_grid: tuple[Problem, Result],
) -> None:
    _, result = full_grid
    periods: dict[tuple[str, int], list[int]] = {}
    for x in result.lessons:
        periods.setdefault((x.class_id, x.day), []).append(x.lesson_no)
    expected = list(range(1, LESSONS_PER_DAY + 1))
    assert set(periods) == {(c, d) for c in CLASS_IDS for d in DAYS}
    for key, numbers in periods.items():
        assert sorted(numbers) == expected, key  # exact packing: no gaps, no free period


@pytest.mark.slow
def test_passport_holds_every_day_and_some_classes_attend(
    full_grid: tuple[Problem, Result],
) -> None:
    p, result = full_grid
    for d in DAYS:
        load = oracle.day_load(p, result, d)
        assert load.people <= PASSPORT, (d, load)
    # Everything remote is admissible but useless; the half-size passport leaves room for some.
    share = len(oracle.in_person_days(result)) / (len(CLASS_IDS) * len(DAYS))
    assert share > 0
    assert result.stats.primary is not None
    assert result.stats.primary > 0


@pytest.mark.slow
def test_heavy_teachers_work_in_one_format_per_day_within_their_limit(
    full_grid: tuple[Problem, Result],
) -> None:
    _, result = full_grid
    in_person = oracle.in_person_days(result)
    weekly = Counter(x.teacher_id for x in result.lessons)
    heavy = {t for t, n in weekly.items() if n >= 24}
    assert heavy == {"t-01", "t-02", "t-03", "t-04", "t-06", "t-07", "t-10", "t-12"}
    assert all(n <= TEACHER_LIMIT for n in weekly.values())
    for t in sorted(heavy):
        for d in DAYS:
            day = [x for x in result.lessons if x.teacher_id == t and x.day == d]
            assert len(day) <= LESSONS_PER_DAY
            assert len({(x.class_id, d) in in_person for x in day}) <= 1, (t, d)
