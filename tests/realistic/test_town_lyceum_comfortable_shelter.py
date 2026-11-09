"""Realistic school: a typical town lyceum 1-11, two parallels, shelter for ~65% of people.

Archetype ``town_lyceum_comfortable_shelter``: 22 classes, 546 pupils, 35 teachers, 14 staff,
main timetable (model 1.1) with ``Passport(386)`` ~ 65% of 595 people, Mon-Fri, 7 lessons.
Capacity is comfortable, so the build is feasible; what matters is the quality of the split
between in-person and remote days. Every result is checked by the independent oracle.

The problem is built deterministically (no randomness); ids are opaque and PD-free.
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
from lad_solver.domain.result import Result, SolveStatus

TIME_LIMIT_S = 60
PARAMS = Params(time_limit_s=TIME_LIMIT_S, num_workers=2, random_seed=0)
DAYS = (1, 2, 3, 4, 5)
LESSONS_PER_DAY = 7
STAFF = 14
PASSPORT = 386
WALL_SLACK_S = 5.0  # phase B feasibility retry and extraction may run slightly past the limit

# MON priority areas (spec §04): maths, natural sciences, language-literature, civic-history.
PRIORITY = frozenset(
    {"s-ukr", "s-lit", "s-eng", "s-math", "s-alg", "s-geo", "s-nature", "s-bio", "s-geog"}
    | {"s-phys", "s-chem", "s-hist", "s-civ"}
)
NO_SHELTER = frozenset({"s-pe"})  # irrelevant for model 1.1, set as a school would

# Weekly hours per grade (one class). Grades 1-4 are mostly taught by the homeroom teacher.
CURRICULUM: dict[int, dict[str, int]] = {
    1: {"s-ukr": 7, "s-math": 4, "s-nature": 2, "s-eng": 2, "s-arts": 2, "s-pe": 3},
    2: {"s-ukr": 7, "s-math": 4, "s-nature": 3, "s-eng": 3, "s-arts": 2, "s-pe": 3},
    3: {"s-ukr": 7, "s-math": 5, "s-nature": 3, "s-eng": 3, "s-arts": 2, "s-pe": 3},
    4: {"s-ukr": 7, "s-math": 5, "s-nature": 3, "s-eng": 3, "s-arts": 2, "s-pe": 3},
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
HOMEROOM_SUBJECTS = frozenset({"s-ukr", "s-math", "s-nature", "s-arts"})  # grades 1-4

# Subject-teacher specialties: (subjects, number of teachers). A class's whole load in a
# specialty goes to one teacher (the least loaded so far), as a deputy head would assign it.
SPECIALTIES: tuple[tuple[frozenset[str], int], ...] = (
    (frozenset({"s-ukr", "s-lit"}), 4),
    (frozenset({"s-eng"}), 3),
    (frozenset({"s-math", "s-alg", "s-geo"}), 4),
    (frozenset({"s-nature", "s-bio"}), 2),
    (frozenset({"s-chem"}), 1),
    (frozenset({"s-geog"}), 1),
    (frozenset({"s-phys"}), 2),
    (frozenset({"s-hist", "s-civ"}), 2),
    (frozenset({"s-inf"}), 2),
    (frozenset({"s-arts"}), 1),
    (frozenset({"s-pe"}), 3),
    (frozenset({"s-elective"}), 2),
)
HOMEROOM_TEACHERS = 8

# ~10% of teachers have one unavailable day; all of them have a light load (<= 7 x 4).
UNAVAILABLE = {"t-26": 4, "t-27": 5, "t-30": 1, "t-35": 3}

# Pupils per class, grade-major (a, b); grades 1-4 <= 24, total 546.
STUDENTS = (
    *(24, 23, 24, 22, 23, 24, 22, 24),
    *(28, 27, 26, 25, 27, 26, 25, 28, 24, 26, 25, 27, 22, 24),
)


@dataclass(frozen=True, slots=True)
class Solved:
    problem: Problem
    result: Result


def _class_ids() -> list[tuple[int, str]]:
    return [(g, f"c-{g}{p}") for g in range(1, 12) for p in "ab"]


def _subjects() -> tuple[Subject, ...]:
    ids = sorted({s for plan in CURRICULUM.values() for s in plan})
    return tuple(Subject(s, priority_area=s in PRIORITY, no_shelter=s in NO_SHELTER) for s in ids)


def _limit(load: int) -> int:
    """Weekly limit 18..27, a little above the assigned load (deputy heads keep a margin)."""
    return min(27, max(18, load + 2))


def build_lyceum() -> Problem:
    classes = _class_ids()
    rows: list[CurriculumRow] = []
    load = Counter[str]()

    def add(class_id: str, teacher_id: str, plan: dict[str, int], subjects: frozenset[str]) -> None:
        for subject_id, hours in plan.items():
            if subject_id in subjects:
                rows.append(CurriculumRow(class_id, subject_id, teacher_id, hours))
                load[teacher_id] += hours

    primary = [(g, c) for g, c in classes if g <= 4]
    for n, (grade, class_id) in enumerate(primary, start=1):
        add(class_id, f"t-{n:02d}", CURRICULUM[grade], HOMEROOM_SUBJECTS)

    next_no = HOMEROOM_TEACHERS + 1
    for subjects, count in SPECIALTIES:
        pool = [f"t-{next_no + i:02d}" for i in range(count)]
        next_no += count
        for grade, class_id in classes:
            plan = CURRICULUM[grade]
            if grade <= 4:
                plan = {s: h for s, h in plan.items() if s not in HOMEROOM_SUBJECTS}
            if not any(s in subjects for s in plan):
                continue
            teacher_id = min(pool, key=lambda t: (load[t], t))
            add(class_id, teacher_id, plan, subjects)

    teachers = tuple(
        Teacher(
            f"t-{n:02d}",
            _limit(load[f"t-{n:02d}"]),
            frozenset({UNAVAILABLE[f"t-{n:02d}"]}) if f"t-{n:02d}" in UNAVAILABLE else frozenset(),
        )
        for n in range(1, next_no)
    )
    return Problem(
        kind=TimetableKind.MAIN,
        school_days=DAYS,
        lessons_per_day=LESSONS_PER_DAY,
        staff=STAFF,
        capacity=Passport(PASSPORT),
        classes=tuple(
            SchoolClass(c, students, priority=False)
            for (_, c), students in zip(classes, STUDENTS, strict=True)
        ),
        subjects=_subjects(),
        teachers=teachers,
        rows=tuple(rows),
    )


@pytest.fixture(scope="module")
def solved() -> Solved:
    problem = build_lyceum()
    return Solved(problem, solve(problem, PARAMS))


def test_scenario_matches_archetype_and_passes_prechecks() -> None:
    p = build_lyceum()
    students = sum(c.students for c in p.classes)
    people = students + p.staff + len(p.teachers)
    load = Counter[str]()
    for r in p.rows:
        load[r.teacher_id] += r.hours
    assert (len(p.classes), students, len(p.teachers), people) == (22, 546, 35, 595)
    assert all(c.students <= 24 for c in p.classes[:8])
    assert all(22 <= c.students <= 28 for c in p.classes)
    assert sum(r.hours for r in p.rows) == 616
    assert 0.6 <= PASSPORT / people <= 0.7
    assert all(18 <= t.weekly_limit <= 27 for t in p.teachers)
    assert all(load[t.id] <= t.weekly_limit for t in p.teachers)
    with_day_off = [t for t in p.teachers if t.unavailable_days]
    assert len(with_day_off) == 4
    assert all(load[t.id] <= LESSONS_PER_DAY * 4 for t in with_day_off)
    assert precheck(p) == ()


@pytest.mark.slow
def test_builds_admissible_timetable_within_time_limit(solved: Solved) -> None:
    p, result = solved.problem, solved.result
    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(p, result) == []
    recomputed = oracle.objective(p, result)
    assert result.stats.objective == recomputed.value
    assert result.stats.primary == recomputed.primary
    assert result.stats.secondary == 0  # no priority classes
    assert not result.stats.quality_relaxed  # the row-day cap is satisfiable here
    assert result.stats.wall_time_s <= TIME_LIMIT_S + WALL_SLACK_S
    assert result.stats.first_solution_s is not None
    assert result.stats.first_solution_s < 5.0  # prototype: 0.23 s


@pytest.mark.slow
def test_passport_holds_every_day_and_shelter_is_used(solved: Solved) -> None:
    p, result = solved.problem, solved.result
    loads = [oracle.day_load(p, result, d) for d in DAYS]
    assert all(x.people <= PASSPORT for x in loads), loads
    # Comfortable shelter: an optimiser fills it; leaving most of it empty is a quality bug.
    assert all(x.people >= 0.9 * PASSPORT for x in loads), loads


@pytest.mark.slow
def test_in_person_share_close_to_shelter_share(solved: Solved) -> None:
    p, result = solved.problem, solved.result
    students = {c.id: c.students for c in p.classes}
    in_person = oracle.in_person_days(result)
    pupil_days = sum(students[c] for c, _ in in_person)
    share = pupil_days / (sum(students.values()) * len(DAYS))
    # Upper bound: per day pupils <= 386 - 14 staff - in-person teachers, i.e. < 372 / 546.
    assert 0.55 <= share < (PASSPORT - STAFF) / sum(students.values()), share
    # Regression baseline (seed 0, 2 workers, 60 s): primary 10847 = 94.6% of the
    # all-in-person value, gap to the bound 5.4%. Floors keep headroom for slower machines.
    ceiling = sum(students[r.class_id] * r.hours for r in p.rows if r.subject_id in PRIORITY)
    primary, bound = result.stats.primary, result.stats.best_bound
    assert primary is not None
    assert bound is not None
    assert primary >= 0.9 * ceiling, (primary, ceiling)
    assert primary >= 0.9 * bound, (primary, bound)


@pytest.mark.slow
def test_class_days_single_format_lessons_in_grid_unavailable_days_kept(solved: Solved) -> None:
    p, result = solved.problem, solved.result
    formats = Counter((f.class_id, f.day) for f in result.day_formats)
    assert set(formats) == {(c.id, d) for c in p.classes for d in DAYS}
    assert set(formats.values()) == {1}
    assert all(1 <= x.lesson_no <= LESSONS_PER_DAY for x in result.lessons)
    weekly = Counter(x.class_id for x in result.lessons)
    planned = Counter[str]()
    for r in p.rows:
        planned[r.class_id] += r.hours
    assert weekly == planned
    for teacher_id, day in UNAVAILABLE.items():
        assert not [x for x in result.lessons if x.teacher_id == teacher_id and x.day == day]
