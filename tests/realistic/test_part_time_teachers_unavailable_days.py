"""Realistic school: grades 5-11 with part-time teachers (сумісники) who come only Mon-Tue.

Archetype ``part_time_teachers_unavailable_days``: the share of part-timers is not in MoE data,
so the research assumes 5-15 % with 4-12 hours and 2-3 available days (teachers working in
several schools). Grades 5-11, two parallels (14 classes, 365 pupils), 30 teachers: 25 core
(limits 18-27, a quarter of them with one unavailable day) and 5 part-timers unavailable on
days 3-5. One of them is the boundary case: exactly 14 hours = 7 lessons x 2 days. Model 1.1,
Mon-Fri, 7 lessons, 11 staff, ``Passport(250)`` ~ 60 % of the 406 people.

What it stresses: unavailable-day constraints, a part-timer filling every slot of both of
their days, one format per teacher per day on the few shared days, and the boundary of the
``teacher_hours_exceed_available_slots`` pre-check (14 h passes, 15 h is rejected).

The problem is built deterministically; ids are opaque and PD-free. Every result is checked
by the independent oracle (``tests/oracle.py``), never by engine code.
"""

from collections import Counter
from dataclasses import dataclass

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
from lad_solver.domain.result import ConflictRule, Result, SolveStatus

TIME_LIMIT_S = 30.0
PARAMS = Params(time_limit_s=TIME_LIMIT_S, num_workers=2, random_seed=0)
DAYS = (1, 2, 3, 4, 5)
LESSONS_PER_DAY = 7
STAFF = 11
PASSPORT = 250
WALL_SLACK_S = 5.0  # phase B feasibility retry and extraction may run slightly past the limit

PART_TIME_OFF = frozenset({3, 4, 5})  # part-timers work only on Monday and Tuesday
PART_TIME_DAYS = frozenset({1, 2})

# MON priority areas (spec §04): maths, natural sciences, language-literature, civic-history.
PRIORITY = frozenset(
    {"s-ukr", "s-lit", "s-eng", "s-math", "s-alg", "s-geo", "s-nature", "s-bio", "s-geog"}
    | {"s-phys", "s-chem", "s-hist", "s-civ"}
)
NO_SHELTER = frozenset({"s-pe"})  # irrelevant for model 1.1, set as a school would

# Weekly hours per grade (one class), the grade 5-11 plan of the town lyceum archetype.
CURRICULUM: dict[int, dict[str, int]] = {
    5: {"s-ukr": 4, "s-lit": 2, "s-eng": 3, "s-math": 5, "s-nature": 3, "s-hist": 2}
    | {"s-inf": 2, "s-arts": 2, "s-pe": 3, "s-elective": 2},
    6: {"s-ukr": 4, "s-lit": 2, "s-eng": 3, "s-math": 5, "s-nature": 2, "s-geog": 2}
    | {"s-hist": 2, "s-inf": 1, "s-arts": 2, "s-pe": 3, "s-elective": 3},
    7: {"s-ukr": 3, "s-lit": 2, "s-eng": 3, "s-alg": 3, "s-geo": 2, "s-bio": 2, "s-geog": 2}
    | {"s-phys": 2, "s-chem": 1, "s-hist": 3, "s-inf": 2, "s-arts": 1, "s-pe": 3}
    | {"s-elective": 2},
    **{
        g: {"s-ukr": 3, "s-lit": 2, "s-eng": 3, "s-alg": 3, "s-geo": 2, "s-bio": 2}
        | {"s-geog": 2, "s-phys": 3, "s-chem": 2, "s-hist": 3, "s-inf": 2, "s-arts": 1}
        | {"s-pe": 3, "s-elective": 2}
        for g in (8, 9)
    },
    **{
        g: {"s-ukr": 2, "s-lit": 3, "s-eng": 3, "s-alg": 3, "s-geo": 2, "s-bio": 2}
        | {"s-geog": 1, "s-phys": 3, "s-chem": 2, "s-hist": 3, "s-civ": 1, "s-inf": 2}
        | {"s-arts": 1, "s-pe": 3, "s-elective": 2}
        for g in (10, 11)
    },
}

CLASS_IDS = tuple(f"c-{g}{p}" for g in range(5, 12) for p in "ab")
# Pupils per class, grade-major (a, b): 24-28 each, 365 in total.
STUDENTS = (26, 25, 27, 26, 28, 24, 25, 27, 26, 28, 24, 26, 25, 28)

# Part-timers: teacher -> (weekly limit, subject -> classes). They come only on Mon-Tue.
BOUNDARY = "t-26"  # 7 classes x 2 h of informatics = 14 h = 7 lessons x 2 available days
BOUNDARY_CLASSES = ("c-7a", "c-7b", "c-8a", "c-8b", "c-9a", "c-9b", "c-10a")
PART_TIMERS: dict[str, tuple[int, dict[str, tuple[str, ...]]]] = {
    BOUNDARY: (14, {"s-inf": BOUNDARY_CLASSES}),
    "t-27": (12, {"s-chem": ("c-7a", "c-7b", "c-8a", "c-8b", "c-9a", "c-9b")}),  # 10 h
    "t-28": (10, {"s-arts": CLASS_IDS[4:]}),  # grades 7-11, 10 h
    "t-29": (8, {"s-civ": CLASS_IDS[10:], "s-elective": ("c-10a", "c-10b")}),  # 8 h
    "t-30": (6, {"s-elective": ("c-5a", "c-5b")}),  # 4 h
}

# Core subject-teacher specialties: (subjects, number of teachers) -> t-1..t-25. A class's
# remaining load in a specialty goes to the least loaded teacher of the pool.
SPECIALTIES: tuple[tuple[frozenset[str], int], ...] = (
    (frozenset({"s-ukr", "s-lit"}), 4),
    (frozenset({"s-eng"}), 3),
    (frozenset({"s-math", "s-alg", "s-geo"}), 4),
    (frozenset({"s-nature", "s-bio"}), 2),
    (frozenset({"s-chem"}), 1),
    (frozenset({"s-geog"}), 1),
    (frozenset({"s-phys"}), 2),
    (frozenset({"s-hist"}), 2),
    (frozenset({"s-inf"}), 1),
    (frozenset({"s-arts"}), 1),
    (frozenset({"s-pe"}), 3),
    (frozenset({"s-elective"}), 1),
)
CORE_TEACHERS = 25
# A quarter of the core teachers have one unavailable day (methodical day, second job).
CORE_UNAVAILABLE = {"t-2": 1, "t-6": 2, "t-10": 3, "t-14": 4, "t-18": 5, "t-22": 3}


@dataclass(frozen=True, slots=True)
class Solved:
    problem: Problem
    result: Result


def _subjects() -> tuple[Subject, ...]:
    ids = sorted({s for plan in CURRICULUM.values() for s in plan})
    return tuple(Subject(s, priority_area=s in PRIORITY, no_shelter=s in NO_SHELTER) for s in ids)


def _grade(class_id: str) -> int:
    return int(class_id.removeprefix("c-")[:-1])


def _core_limit(load: int) -> int:
    """Weekly limit 18..27, a little above the assigned load (deputy heads keep a margin)."""
    return min(27, max(18, load + 2))


def build_school(*, boundary_hours: int = 14, boundary_limit: int = 14) -> Problem:
    """The school; ``boundary_hours`` > 14 adds hours to the boundary part-timer's first row."""
    rows: list[CurriculumRow] = []
    load = Counter[str]()
    taken: set[tuple[str, str]] = set()

    def add(class_id: str, subject_id: str, teacher_id: str, hours: int) -> None:
        rows.append(CurriculumRow(class_id, subject_id, teacher_id, hours))
        load[teacher_id] += hours
        taken.add((class_id, subject_id))

    extra = boundary_hours - 14
    for teacher_id, (_, plan) in PART_TIMERS.items():
        for subject_id, class_ids in plan.items():
            for class_id in class_ids:
                hours = CURRICULUM[_grade(class_id)][subject_id]
                if teacher_id == BOUNDARY and class_id == class_ids[0]:
                    hours += extra
                add(class_id, subject_id, teacher_id, hours)

    next_no = 1
    for subjects, count in SPECIALTIES:
        pool = [f"t-{next_no + i}" for i in range(count)]
        next_no += count
        for class_id in CLASS_IDS:
            plan = CURRICULUM[_grade(class_id)]
            mine = {s: h for s, h in plan.items() if s in subjects and (class_id, s) not in taken}
            if not mine:
                continue
            teacher_id = min(pool, key=lambda t: (load[t], int(t.removeprefix("t-"))))
            for subject_id, hours in mine.items():
                add(class_id, subject_id, teacher_id, hours)
    assert next_no == CORE_TEACHERS + 1

    core = tuple(
        Teacher(
            f"t-{n}",
            _core_limit(load[f"t-{n}"]),
            frozenset({CORE_UNAVAILABLE[f"t-{n}"]})
            if f"t-{n}" in CORE_UNAVAILABLE
            else frozenset(),
        )
        for n in range(1, CORE_TEACHERS + 1)
    )
    part = tuple(
        Teacher(tid, boundary_limit if tid == BOUNDARY else limit, PART_TIME_OFF)
        for tid, (limit, _) in PART_TIMERS.items()
    )
    return Problem(
        kind=TimetableKind.MAIN,
        school_days=DAYS,
        lessons_per_day=LESSONS_PER_DAY,
        staff=STAFF,
        capacity=Passport(PASSPORT),
        classes=tuple(
            SchoolClass(c, n, priority=False) for c, n in zip(CLASS_IDS, STUDENTS, strict=True)
        ),
        subjects=_subjects(),
        teachers=(*core, *part),
        rows=tuple(rows),
    )


@pytest.fixture(scope="module")
def solved() -> Solved:
    problem = build_school()
    return Solved(problem, solve(problem, PARAMS))


def test_scenario_matches_archetype_and_passes_prechecks() -> None:
    p = build_school()
    pupils = sum(c.students for c in p.classes)
    people = pupils + p.staff + len(p.teachers)
    load = Counter[str]()
    hours = Counter[str]()
    for r in p.rows:
        load[r.teacher_id] += r.hours
        hours[r.class_id] += r.hours
    assert (len(p.classes), pupils, len(p.teachers), people) == (14, 365, 30, 406)
    assert all(24 <= c.students <= 28 for c in p.classes)
    assert 0.58 <= PASSPORT / people <= 0.65
    assert all(hours[c] == sum(CURRICULUM[_grade(c)].values()) for c in CLASS_IDS)
    assert all(load[t.id] <= t.weekly_limit for t in p.teachers)
    part = [t for t in p.teachers if t.unavailable_days == PART_TIME_OFF]
    assert {t.id for t in part} == set(PART_TIMERS)
    assert all(4 <= load[t.id] <= 12 for t in part if t.id != BOUNDARY)
    assert load[BOUNDARY] == LESSONS_PER_DAY * len(PART_TIME_DAYS) == 14
    core = [t for t in p.teachers if t.id not in PART_TIMERS]
    assert all(18 <= t.weekly_limit <= 27 for t in core)
    assert sum(1 for t in core if t.unavailable_days) == len(CORE_UNAVAILABLE) == 6
    assert precheck(p) == ()


@pytest.mark.slow
def test_builds_admissible_timetable_within_time_limit(solved: Solved) -> None:
    p, result = solved.problem, solved.result
    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(p, result) == []
    assert result.stats.objective == oracle.objective(p, result).value
    assert result.stats.wall_time_s <= TIME_LIMIT_S + WALL_SLACK_S
    assert result.stats.first_solution_s is not None
    assert result.stats.first_solution_s < 10.0  # harsher prototype: 0.15 s


@pytest.mark.slow
def test_passport_holds_every_day_and_shelter_is_used(solved: Solved) -> None:
    p, result = solved.problem, solved.result
    loads = {d: oracle.day_load(p, result, d) for d in DAYS}
    assert all(x.people <= PASSPORT for x in loads.values()), loads
    students = {c.id: c.students for c in p.classes}
    in_person = [f for f in result.day_formats if f.in_person]
    share = sum(students[f.class_id] for f in in_person) / (sum(students.values()) * len(DAYS))
    # Per day at most 250 - 11 staff - teachers on site pupils, i.e. below 239 / 365 = 0.65.
    assert 0.4 <= share < (PASSPORT - STAFF) / sum(students.values()), share  # prototype: 0.57


@pytest.mark.slow
def test_part_timers_teach_only_on_their_days_and_boundary_fills_them(solved: Solved) -> None:
    p, result = solved.problem, solved.result
    for teacher in p.teachers:
        days = {x.day for x in result.lessons if x.teacher_id == teacher.id}
        assert not days & teacher.unavailable_days, (teacher.id, days)
    for teacher_id in PART_TIMERS:
        days = {x.day for x in result.lessons if x.teacher_id == teacher_id}
        assert days <= PART_TIME_DAYS, (teacher_id, days)
    boundary = Counter((x.day, x.lesson_no) for x in result.lessons if x.teacher_id == BOUNDARY)
    expected = {(d, n) for d in PART_TIME_DAYS for n in range(1, LESSONS_PER_DAY + 1)}
    assert set(boundary) == expected
    assert set(boundary.values()) == {1}


@pytest.mark.parametrize(
    ("limit", "rules"),
    [
        (15, {ConflictRule.TEACHER_HOURS_EXCEED_AVAILABLE_SLOTS}),
        (
            14,
            {ConflictRule.TEACHER_HOURS_EXCEED_AVAILABLE_SLOTS, ConflictRule.TEACHER_WEEKLY_LIMIT},
        ),
    ],
)
def test_fifteen_hours_on_two_days_is_rejected_by_precheck(
    limit: int, rules: set[ConflictRule]
) -> None:
    problem = build_school(boundary_hours=15, boundary_limit=limit)
    result = solve(problem, PARAMS)
    assert result.status is SolveStatus.INFEASIBLE
    assert {c.rule for c in result.conflicts} == rules
    assert {c.teacher_id for c in result.conflicts} == {BOUNDARY}
    assert result.lessons == ()
