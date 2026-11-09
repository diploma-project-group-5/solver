"""Realistic school: frontline gymnasium 1-9 with a tight shelter (about 35 % of the people).

Based on Kharkiv (a shelter for 83 people holds three classes, two shifts) and Zaporizhzhia
(in person only in shelters); MON shelter priority: grades 1-4 first, then 9. 18 classes of
23-27 pupils (447 in total), 29 teachers, 12 non-teaching staff, main timetable (model 1.1).
Passport(171) leaves room for five or six classes plus their teachers per day, so the day
assignment is a knapsack over classes with every in-person teacher counted once.

Assertions are on behaviour (status, oracle, capacity, objective properties), never on grids.
"""

from collections import Counter
from dataclasses import dataclass, replace
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
from lad_solver.domain.result import Result, SolveStatus

TIME_LIMIT_S = 30.0
PARAMS = Params(time_limit_s=TIME_LIMIT_S, num_workers=2, random_seed=0)
WALL_SLACK_S = 3.0  # process scheduling on a busy CI runner
FIRST_SOLUTION_MAX_S = 5.0  # prototype: 0.17 s
PASSPORT = 171  # about 35 % of 447 pupils + 29 teachers + 12 staff
KHARKIV_PASSPORT = 83  # the Kharkiv shelter that fits three classes (without staff)
MAX_KHARKIV_CLASSES = 3
STAFF = 12
DAYS = (1, 2, 3, 4, 5)
LESSONS_PER_DAY = 7
PARALLELS = ("a", "b")
FOUND = {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}

# Pupils per class, grade 1 -> 9, parallels a and b (sum 447).
STUDENTS: dict[int, tuple[int, int]] = {
    1: (27, 26),
    2: (26, 25),
    3: (25, 25),
    4: (24, 25),
    5: (27, 26),
    6: (25, 24),
    7: (24, 23),
    8: (23, 24),
    9: (25, 23),
}
PRIORITY_GRADES = frozenset({1, 2, 3, 4, 9})  # MON shelter priority

# Subject id -> (priority educational area, not held in the shelter).
SUBJECTS: dict[str, tuple[bool, bool]] = {
    "s-ukr": (True, False),
    "s-lit": (False, False),
    "s-wlit": (False, False),
    "s-math": (True, False),
    "s-eng": (False, False),
    "s-nature": (False, False),
    "s-sci": (False, False),
    "s-hist": (False, False),
    "s-bio": (False, False),
    "s-geo": (False, False),
    "s-phys": (False, False),
    "s-chem": (False, False),
    "s-inf": (False, False),
    "s-art": (False, False),
    "s-music": (False, False),
    "s-pe": (False, True),
    "s-tech": (False, False),
    "s-health": (False, False),
}

# Weekly hours per grade (typical NUS plan for a five-day week).
PRIMARY_12 = {"s-ukr": 7, "s-math": 4, "s-nature": 3, "s-art": 1, "s-music": 1, "s-pe": 3}
PRIMARY_34 = {"s-ukr": 7, "s-math": 5, "s-nature": 3, "s-art": 1, "s-inf": 1, "s-pe": 3}
CURRICULUM: dict[int, dict[str, int]] = {
    1: PRIMARY_12 | {"s-eng": 2},
    2: PRIMARY_12 | {"s-eng": 3},
    3: PRIMARY_34 | {"s-eng": 3},
    4: PRIMARY_34 | {"s-eng": 3},
    5: {
        "s-ukr": 4, "s-lit": 2, "s-wlit": 2, "s-math": 4, "s-eng": 3, "s-hist": 1,
        "s-sci": 3, "s-inf": 1, "s-art": 2, "s-pe": 3, "s-tech": 2, "s-health": 1,
    },
    6: {
        "s-ukr": 4, "s-lit": 2, "s-wlit": 2, "s-math": 4, "s-eng": 3, "s-hist": 2,
        "s-sci": 3, "s-geo": 1, "s-inf": 1, "s-art": 2, "s-pe": 3, "s-tech": 2,
        "s-health": 1,
    },
    7: {
        "s-ukr": 3, "s-lit": 2, "s-wlit": 2, "s-math": 5, "s-eng": 3, "s-hist": 2,
        "s-bio": 2, "s-geo": 2, "s-phys": 2, "s-chem": 2, "s-inf": 1, "s-art": 1,
        "s-pe": 3, "s-tech": 1, "s-health": 1,
    },
    8: {
        "s-ukr": 3, "s-lit": 2, "s-wlit": 2, "s-math": 5, "s-eng": 3, "s-hist": 2,
        "s-bio": 2, "s-geo": 2, "s-phys": 3, "s-chem": 2, "s-inf": 1, "s-pe": 3,
        "s-tech": 1, "s-health": 1,
    },
    9: {
        "s-ukr": 3, "s-lit": 2, "s-wlit": 2, "s-math": 5, "s-eng": 3, "s-hist": 2,
        "s-bio": 2, "s-geo": 2, "s-phys": 3, "s-chem": 2, "s-inf": 2, "s-pe": 3,
        "s-tech": 1, "s-health": 1,
    },
}  # fmt: skip

HOMEROOM_SUBJECTS = frozenset({"s-ukr", "s-math", "s-nature", "s-art"})  # grades 1-4


@dataclass(frozen=True, slots=True)
class Department:
    subjects: frozenset[str]
    limits: tuple[int, ...]  # one subject teacher per limit, filled first-fit by class


DEPARTMENTS: dict[str, Department] = {
    "ukr": Department(frozenset({"s-ukr", "s-lit"}), (20, 18, 18, 18)),
    "wlit": Department(frozenset({"s-wlit"}), (20,)),
    "math": Department(frozenset({"s-math"}), (24, 25)),
    "eng": Department(frozenset({"s-eng"}), (18, 18, 18)),
    "hist": Department(frozenset({"s-hist"}), (20,)),
    "bio": Department(frozenset({"s-sci", "s-bio"}), (25,)),
    "geo": Department(frozenset({"s-geo", "s-health"}), (25,)),
    "phys": Department(frozenset({"s-phys"}), (18,)),
    "chem": Department(frozenset({"s-chem"}), (18,)),
    "inf": Department(frozenset({"s-inf"}), (18,)),
    "art": Department(frozenset({"s-art", "s-music"}), (18,)),
    "pe": Department(frozenset({"s-pe"}), (20, 20, 18)),
    "tech": Department(frozenset({"s-tech"}), (18,)),
}
HOMEROOM_LIMIT = 18


def _class_id(grade: int, parallel: str) -> str:
    return f"c-{grade}{parallel}"


@cache
def gymnasium(passport: int = PASSPORT) -> Problem:
    classes: list[SchoolClass] = []
    teachers: list[Teacher] = []
    rows: list[CurriculumRow] = []
    next_no = iter(range(1, 100))
    pools: dict[str, list[Teacher]] = {}
    for name, dep in DEPARTMENTS.items():
        pools[name] = [Teacher(f"t-{next(next_no)}", limit) for limit in dep.limits]
        teachers.extend(pools[name])
    load = Counter[str]()
    for grade, students in STUDENTS.items():
        for parallel, n in zip(PARALLELS, students, strict=True):
            cid = _class_id(grade, parallel)
            classes.append(SchoolClass(cid, n, priority=grade in PRIORITY_GRADES))
            hours = dict(CURRICULUM[grade])
            if grade <= 4:
                homeroom = Teacher(f"t-{next(next_no)}", HOMEROOM_LIMIT)
                teachers.append(homeroom)
                own = {s: h for s, h in hours.items() if s in HOMEROOM_SUBJECTS}
                rows.extend(CurriculumRow(cid, s, homeroom.id, h) for s, h in sorted(own.items()))
                hours = {s: h for s, h in hours.items() if s not in HOMEROOM_SUBJECTS}
            for name, dep in DEPARTMENTS.items():
                chunk = {s: h for s, h in hours.items() if s in dep.subjects}
                if not chunk:
                    continue
                need = sum(chunk.values())
                fit = next((t for t in pools[name] if load[t.id] + need <= t.weekly_limit), None)
                assert fit is not None, f"department {name} has no room for {cid}"
                load[fit.id] += need
                rows.extend(CurriculumRow(cid, s, fit.id, h) for s, h in sorted(chunk.items()))
    return Problem(
        kind=TimetableKind.MAIN,
        school_days=DAYS,
        lessons_per_day=LESSONS_PER_DAY,
        staff=STAFF,
        capacity=Passport(passport),
        classes=tuple(classes),
        subjects=tuple(Subject(s, pa, ns) for s, (pa, ns) in SUBJECTS.items()),
        teachers=tuple(teachers),
        rows=tuple(rows),
    )


@cache
def solved(passport: int = PASSPORT) -> Result:
    return solve(gymnasium(passport), PARAMS)


def _assert_admissible(problem: Problem, result: Result) -> None:
    assert result.status in FOUND, result.status
    assert oracle.violations(problem, result) == []
    recomputed = oracle.objective(problem, result)
    assert result.stats.objective == recomputed.value


def test_scenario_is_realistic() -> None:
    p = gymnasium()
    assert len(p.classes) == 18
    assert sum(c.students for c in p.classes) == 447
    assert all(23 <= c.students <= 27 for c in p.classes)
    load = Counter[str]()
    for r in p.rows:
        load[r.teacher_id] += r.hours
    assert 28 <= len(p.teachers) <= 32
    assert all(load[t.id] > 0 for t in p.teachers)
    assert all(18 <= t.weekly_limit <= 27 for t in p.teachers)
    people = sum(c.students for c in p.classes) + len(p.teachers) + STAFF
    assert 0.30 <= PASSPORT / people <= 0.40


@pytest.mark.slow
def test_week_is_feasible_within_time_limit_and_passes_oracle() -> None:
    p, result = gymnasium(), solved()
    _assert_admissible(p, result)
    assert result.stats.wall_time_s <= TIME_LIMIT_S + WALL_SLACK_S
    assert result.stats.first_solution_s is not None
    assert result.stats.first_solution_s < FIRST_SOLUTION_MAX_S
    assert not result.stats.quality_relaxed


@pytest.mark.slow
def test_passport_is_binding_on_every_day() -> None:
    p, result = gymnasium(), solved()
    _assert_admissible(p, result)
    biggest = max(c.students for c in p.classes)
    for d in DAYS:
        load = oracle.day_load(p, result, d)
        assert load.people <= PASSPORT, (d, load)
        # Within about one class of the limit: the day is filled, not left half empty.
        assert load.people >= PASSPORT - biggest - 5, (d, load)
    share = len(oracle.in_person_days(result)) / (len(p.classes) * len(DAYS))
    assert 0.25 <= share <= 0.45, share


@pytest.mark.slow
def test_priority_classes_in_person_days_are_counted_in_the_objective() -> None:
    # The priority flag is only a tie-break below W * primary, so "flagged classes get more
    # in-person days on average" is not a guaranteed property; check that it is counted instead.
    p, result = gymnasium(), solved()
    _assert_admissible(p, result)
    days_of = Counter(c for c, _ in oracle.in_person_days(result))
    flagged_days = sum(days_of[c.id] for c in p.classes if c.priority)
    assert result.stats.secondary == flagged_days
    assert flagged_days > 0


@pytest.mark.slow
def test_kharkiv_shelter_for_83_still_feasible_with_two_or_three_classes() -> None:
    # 83 - 12 staff leaves 71 places: two classes and their teachers; a third does not fit.
    p, result = gymnasium(KHARKIV_PASSPORT), solved(KHARKIV_PASSPORT)
    _assert_admissible(p, result)
    # Hard rules only: the run stops at the time limit (FEASIBLE), so "no room for one more
    # class" or "strictly worse than the wider shelter" are optimality claims, not guarantees.
    per_day = Counter(d for _, d in oracle.in_person_days(result))
    for d in DAYS:
        load = oracle.day_load(p, result, d)
        assert load.people <= KHARKIV_PASSPORT, (d, load)
        assert per_day[d] <= MAX_KHARKIV_CLASSES, (d, per_day[d])
    assert sum(per_day.values()) >= len(DAYS), per_day  # the shelter is used, not "all remote"
    if solved().status is SolveStatus.OPTIMAL:  # only then is the wider shelter's run a bound
        wider = replace(p, capacity=Passport(PASSPORT))
        assert oracle.objective(wider, result).primary <= oracle.objective(wider, solved()).primary
