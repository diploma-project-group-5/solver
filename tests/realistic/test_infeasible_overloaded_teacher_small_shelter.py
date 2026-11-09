"""Realistic school: a small 5-9 gymnasium with an overloaded science teacher and a tiny shelter.

47 pupils in five single-class grades (just above the 45-pupil subvention threshold), 11
teachers and 7 non-teaching staff. The science teacher (t-03) carries nature (g5-6) plus
physics, chemistry and biology (g7-9): 22 h against a rate of 18 h, as a part-timer on Mon-Tue
only. The shelter is the "simplest" one for 6 people.

Variant a must be rejected by the pre-checks with all three conflicts at once. Variant b (the
control: the teacher fixed, passport 12) proves that a shelter too small for even one class is
not infeasible, because "everything remote" is always admissible (ADR 2026-11-04).
"""

from collections import Counter

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
from lad_solver.domain.result import (
    CapacityCondition,
    Conflict,
    ConflictRule,
    Result,
    SolveStatus,
    StopReason,
)

pytestmark = pytest.mark.slow  # real CP-SAT solves: run with -m "" (CI scenarios job)
TIME_LIMIT_S = 30
PARAMS = Params(time_limit_s=TIME_LIMIT_S, num_workers=2, random_seed=0)
STAFF = 7
TINY_SHELTER = Passport(6)  # variant a: below the non-teaching staff alone
CONTROL_SHELTER = Passport(12)  # variant b: >= staff, < staff + smallest class + one teacher
T_NAT = "t-03"
T_NAT_HOURS = 22
OTHER_LIMIT = 27

# grade -> (class id, pupils)
CLASSES: dict[int, tuple[str, int]] = {
    5: ("c-5a", 9),
    6: ("c-6a", 8),
    7: ("c-7a", 11),
    8: ("c-8a", 10),
    9: ("c-9a", 9),
}

# subject id -> (priority educational area, not held in the shelter)
SUBJECTS: dict[str, tuple[bool, bool]] = {
    "s-ukr": (True, False),
    "s-ukr-lit": (False, False),
    "s-for-lit": (False, False),
    "s-eng": (True, False),
    "s-math": (True, False),
    "s-alg": (True, False),
    "s-geom": (True, False),
    "s-hist": (False, False),
    "s-health": (False, False),
    "s-sci": (True, False),
    "s-geo": (False, False),
    "s-phys": (True, False),
    "s-chem": (True, False),
    "s-bio": (True, False),
    "s-inf": (False, False),
    "s-tech": (False, False),
    "s-art": (False, False),
    "s-pe": (False, True),
}

_SENIOR = {
    "s-ukr": 3,
    "s-ukr-lit": 2,
    "s-for-lit": 2,
    "s-eng": 3,
    "s-alg": 3,
    "s-geom": 2,
    "s-hist": 2,
    "s-geo": 2,
    "s-phys": 2,
    "s-chem": 2,
    "s-bio": 2,
    "s-inf": 2,
    "s-tech": 1,
    "s-art": 1,
    "s-pe": 3,
    "s-health": 1,
}

# grade -> subject -> weekly hours (typical educational programme of a 5-9 gymnasium)
CURRICULUM: dict[int, dict[str, int]] = {
    5: {
        "s-ukr": 5,
        "s-ukr-lit": 2,
        "s-for-lit": 2,
        "s-eng": 3,
        "s-math": 5,
        "s-hist": 1,
        "s-health": 1,
        "s-sci": 2,
        "s-inf": 1,
        "s-tech": 2,
        "s-art": 2,
        "s-pe": 2,
    },
    6: {
        "s-ukr": 4,
        "s-ukr-lit": 2,
        "s-for-lit": 2,
        "s-eng": 3,
        "s-math": 5,
        "s-hist": 2,
        "s-health": 1,
        "s-sci": 2,
        "s-geo": 2,
        "s-inf": 1,
        "s-tech": 2,
        "s-art": 2,
        "s-pe": 3,
    },
    7: {**_SENIOR, "s-inf": 1},
    8: _SENIOR,
    9: _SENIOR,
}


def _teacher_of(grade: int, subject: str) -> str:
    if subject == "s-ukr" or (subject == "s-ukr-lit" and grade <= 7):
        return "t-01"
    if subject in {"s-ukr-lit", "s-for-lit"}:
        return "t-02"
    if subject in {"s-sci", "s-phys", "s-chem", "s-bio"}:
        return T_NAT
    table = {
        "s-eng": "t-04",
        "s-math": "t-05",
        "s-alg": "t-05",
        "s-geom": "t-06",
        "s-inf": "t-06",
        "s-hist": "t-07",
        "s-health": "t-07",
        "s-geo": "t-08",
        "s-pe": "t-09",
        "s-tech": "t-10",
        "s-art": "t-11",
    }
    return table[subject]


def _school(*, overloaded: bool, capacity: Passport) -> Problem:
    rows = tuple(
        CurriculumRow(CLASSES[g][0], s, _teacher_of(g, s), h)
        for g, plan in CURRICULUM.items()
        for s, h in plan.items()
    )
    nat = (
        Teacher(T_NAT, 18, frozenset({3, 4, 5}))
        if overloaded
        else Teacher(T_NAT, OTHER_LIMIT, frozenset[int]())
    )
    others = tuple(
        Teacher(f"t-{n:02d}", OTHER_LIMIT, frozenset[int]()) for n in range(1, 12) if n != 3
    )
    return Problem(
        kind=TimetableKind.MAIN,
        school_days=(1, 2, 3, 4, 5),
        lessons_per_day=7,
        staff=STAFF,
        capacity=capacity,
        classes=tuple(SchoolClass(cid, n, priority=False) for cid, n in CLASSES.values()),
        subjects=tuple(Subject(s, pa, ns) for s, (pa, ns) in SUBJECTS.items()),
        teachers=tuple(sorted((nat, *others), key=lambda t: t.id)),
        rows=rows,
    )


EXPECTED_A = Counter(
    [
        Conflict(ConflictRule.STAFF_OVER_PASSPORT, condition=CapacityCondition.PASSPORT),
        Conflict(ConflictRule.TEACHER_WEEKLY_LIMIT, teacher_id=T_NAT),
        Conflict(ConflictRule.TEACHER_HOURS_EXCEED_AVAILABLE_SLOTS, teacher_id=T_NAT),
    ]
)


@pytest.fixture(scope="module")
def control() -> tuple[Problem, Result]:
    p = _school(overloaded=False, capacity=CONTROL_SHELTER)
    return p, solve(p, PARAMS)


def test_scenario_matches_the_archetype() -> None:
    p = _school(overloaded=True, capacity=TINY_SHELTER)
    assert sum(c.students for c in p.classes) == 47
    load = Counter[str]()
    for r in p.rows:
        load[r.teacher_id] += r.hours
    assert load[T_NAT] == T_NAT_HOURS
    assert p.lessons_per_day * 2 < T_NAT_HOURS  # only Mon-Tue: 22 > 7 x 2
    assert all(n <= OTHER_LIMIT for t, n in load.items() if t != T_NAT)
    assert len(p.teachers) == 11
    class_hours = Counter[str]()
    for r in p.rows:
        class_hours[r.class_id] += r.hours
    assert [class_hours[CLASSES[g][0]] for g in range(5, 10)] == [28, 31, 32, 33, 33]
    smallest = min(c.students for c in p.classes)
    assert STAFF <= CONTROL_SHELTER.passport < STAFF + smallest + 1


def test_precheck_reports_all_three_conflicts_at_once() -> None:
    p = _school(overloaded=True, capacity=TINY_SHELTER)
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.INFEASIBLE
    assert Counter(result.conflicts) == EXPECTED_A


def test_precheck_rejection_has_no_timetable_and_is_instant() -> None:
    p = _school(overloaded=True, capacity=TINY_SHELTER)
    result = solve(p, PARAMS)
    assert result.stats.stop_reason is StopReason.PRECHECK
    assert result.lessons == ()
    assert result.day_formats == ()
    assert result.stats.objective is None
    assert result.stats.wall_time_s < 1.0


@pytest.mark.parametrize(
    ("overloaded", "capacity", "rules"),
    [
        (False, TINY_SHELTER, {ConflictRule.STAFF_OVER_PASSPORT}),
        (
            True,
            CONTROL_SHELTER,
            {
                ConflictRule.TEACHER_WEEKLY_LIMIT,
                ConflictRule.TEACHER_HOURS_EXCEED_AVAILABLE_SLOTS,
            },
        ),
    ],
    ids=["shelter-only", "teacher-only"],
)
def test_each_cause_alone_yields_only_its_own_conflicts(
    *, overloaded: bool, capacity: Passport, rules: set[ConflictRule]
) -> None:
    result = solve(_school(overloaded=overloaded, capacity=capacity), PARAMS)
    assert result.status is SolveStatus.INFEASIBLE
    assert result.stats.stop_reason is StopReason.PRECHECK
    assert {c.rule for c in result.conflicts} == rules
    assert len(result.conflicts) == len(rules)


def test_shelter_too_small_for_one_class_is_optimal_all_remote(
    control: tuple[Problem, Result],
) -> None:
    p, result = control
    assert result.status is SolveStatus.OPTIMAL
    assert result.conflicts == ()
    assert oracle.violations(p, result) == []
    recomputed = oracle.objective(p, result)
    assert result.stats.objective == recomputed.value == 0
    assert len(result.day_formats) == len(p.classes) * len(p.school_days)
    assert not any(f.in_person for f in result.day_formats)  # in-person share 0
    assert result.stats.wall_time_s < TIME_LIMIT_S


def test_control_passport_holds_every_day_with_nobody_but_staff(
    control: tuple[Problem, Result],
) -> None:
    p, result = control
    for day in p.school_days:
        load = oracle.day_load(p, result, day)
        assert load.people <= CONTROL_SHELTER.passport
        assert (load.pupils, load.teachers, load.people) == (0, 0, STAFF)
    # With the part-time restriction lifted, t-03 must use at least four days: 22 > 7 x 3.
    assert sum(1 for x in result.lessons if x.teacher_id == T_NAT) == T_NAT_HOURS
    assert len({x.day for x in result.lessons if x.teacher_id == T_NAT}) >= 4
