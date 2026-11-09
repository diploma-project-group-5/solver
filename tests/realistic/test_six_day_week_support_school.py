"""Realistic scenario: a small "support" school (опорна школа) on a six-day week.

Archetype ``six_day_week_support_school``: a six-day week is an allowed exception (the sanitary
rules treat five days as the baseline; MoE 2026 permits six days during power cuts). Grades 1-11
with 1-2 parallels (16 classes, ~348 pupils), 6 lessons a day on days 1..6, 25 teachers with a
weekly limit of 24, every third teacher (part-timers) never works on Saturday. Model 1.1 with a
passport of 270 (about 70 % of everyone in the building).

What it stresses: ``school_days`` including day 6, uneven availability (5 vs 6 days), a shorter
day with more days, and the row-day cap computed over 6 or 5 available days.

The engine is checked by the independent oracle (``tests/oracle.py``), never by its own code.
"""

from collections import Counter
from math import ceil

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
from lad_solver.domain.result import ConflictRule, Result, SolveStatus

DAYS = (1, 2, 3, 4, 5, 6)
SATURDAY = 6
LESSONS_PER_DAY = 6
STAFF = 12
PASSPORT = 270
WEEKLY_LIMIT = 24
TIME_LIMIT_S = 30.0
WALL_TIME_MARGIN_S = 5.0  # phase B feasibility retry may run past the budget
ROW_DAY_CAP_MIN = 2
PARAMS = Params(time_limit_s=TIME_LIMIT_S, num_workers=2, random_seed=0)

# (class id, pupils): two 1st, 5th, 7th, 9th and 11th grades; 348 pupils in total.
CLASSES: tuple[tuple[str, int], ...] = (
    ("c-1a", 22),
    ("c-1b", 21),
    ("c-2a", 24),
    ("c-3a", 23),
    ("c-4a", 25),
    ("c-5a", 26),
    ("c-5b", 24),
    ("c-6a", 22),
    ("c-7a", 21),
    ("c-7b", 23),
    ("c-8a", 20),
    ("c-9a", 19),
    ("c-9b", 18),
    ("c-10a", 22),
    ("c-11a", 20),
    ("c-11b", 18),
)

# (subject id, priority educational area, not held in the shelter)
SUBJECTS: tuple[tuple[str, bool, bool], ...] = (
    ("s-ukr", True, False),
    ("s-math", True, False),
    ("s-eng", True, False),
    ("s-lit", False, False),
    ("s-world", False, False),  # "I explore the world", primary school
    ("s-hist", False, False),
    ("s-geo", False, False),
    ("s-sci", False, False),  # integrated science, grades 5-6
    ("s-bio", False, False),
    ("s-chem", False, False),
    ("s-phys", False, False),
    ("s-it", False, False),
    ("s-art", False, False),
    ("s-labor", False, False),
    ("s-health", False, False),
    ("s-pe", False, True),
    ("s-defense", False, True),
)

# Weekly hours per grade (5-day-like plans squeezed into 6 x 6 = 36 slots; 8-11 have 33 h).
# Primary grades: homeroom subjects (s-ukr, s-math, s-world, s-art) + English + PE.
_PRIMARY: dict[int, dict[str, int]] = {
    1: {"s-ukr": 7, "s-math": 4, "s-world": 3, "s-art": 1, "s-eng": 2, "s-pe": 3},  # 20
    2: {"s-ukr": 7, "s-math": 4, "s-world": 3, "s-art": 2, "s-eng": 3, "s-pe": 3},  # 22
    3: {"s-ukr": 7, "s-math": 5, "s-world": 3, "s-art": 2, "s-eng": 3, "s-pe": 3},  # 23
    4: {"s-ukr": 7, "s-math": 5, "s-world": 3, "s-art": 2, "s-eng": 3, "s-pe": 3},  # 23
}
_G5 = {
    "s-ukr": 5,
    "s-math": 4,
    "s-eng": 3,
    "s-lit": 2,
    "s-hist": 1,
    "s-sci": 3,
    "s-it": 1,
    "s-art": 2,
    "s-labor": 2,
    "s-health": 1,
    "s-pe": 3,
}  # 27
_G6 = _G5 | {"s-math": 5, "s-hist": 2}  # 29
_G7 = {
    "s-ukr": 4,
    "s-math": 6,
    "s-eng": 3,
    "s-lit": 2,
    "s-hist": 2,
    "s-geo": 2,
    "s-bio": 2,
    "s-chem": 1,
    "s-phys": 2,
    "s-it": 1,
    "s-art": 1,
    "s-labor": 1,
    "s-health": 1,
    "s-pe": 3,
}  # 31
_G8 = _G7 | {"s-chem": 2, "s-it": 2}  # 33
_G10 = {
    "s-ukr": 4,
    "s-math": 5,
    "s-eng": 3,
    "s-lit": 2,
    "s-hist": 3,
    "s-geo": 1,
    "s-bio": 2,
    "s-chem": 2,
    "s-phys": 3,
    "s-it": 2,
    "s-art": 1,
    "s-defense": 2,
    "s-pe": 3,
}  # 33
_SECONDARY: dict[int, dict[str, int]] = {5: _G5, 6: _G6, 7: _G7, 8: _G8, 9: _G8, 10: _G10, 11: _G10}

# Who teaches what: homeroom teachers t-1..t-5, then subject teachers t-6..t-25.
_HOMEROOM = {"c-1a": "t-1", "c-1b": "t-2", "c-2a": "t-3", "c-3a": "t-4", "c-4a": "t-5"}
_HOMEROOM_SUBJECTS = frozenset({"s-ukr", "s-math", "s-world", "s-art"})
_PRIMARY_IDS = tuple(_HOMEROOM)
_ASSIGN: dict[str, dict[str, tuple[str, ...]]] = {
    "s-ukr": {
        "t-6": ("c-5a", "c-5b", "c-6a", "c-7a"),
        "t-7": ("c-7b", "c-8a", "c-9a", "c-9b"),
        "t-8": ("c-10a", "c-11a", "c-11b"),
    },
    "s-lit": {"t-9": ("*",)},
    "s-math": {
        "t-10": ("c-5a", "c-5b", "c-6a", "c-7a"),
        "t-11": ("c-7b", "c-8a", "c-9a"),
        "t-12": ("c-9b", "c-10a", "c-11a", "c-11b"),
    },
    "s-eng": {
        "t-13": (*_PRIMARY_IDS, "c-5a", "c-5b", "c-6a"),
        "t-14": ("c-7a", "c-7b", "c-8a", "c-9a", "c-9b", "c-10a", "c-11a", "c-11b"),
    },
    "s-hist": {"t-15": ("*",)},
    "s-geo": {"t-16": ("*",)},
    "s-health": {"t-16": ("*",)},
    "s-bio": {"t-17": ("*",)},
    "s-chem": {"t-18": ("*",)},
    "s-sci": {"t-18": ("*",)},
    "s-phys": {"t-19": ("*",)},
    "s-it": {"t-20": ("*",)},
    "s-art": {"t-21": ("*",)},
    "s-labor": {"t-22": ("*",)},
    "s-defense": {"t-22": ("*",)},
    "s-pe": {
        "t-23": _PRIMARY_IDS,
        "t-24": ("c-5a", "c-5b", "c-6a", "c-7a", "c-7b"),
        "t-25": ("c-8a", "c-9a", "c-9b", "c-10a", "c-11a", "c-11b"),
    },
}
TEACHER_IDS = tuple(f"t-{n}" for n in range(1, 26))
# Every third teacher is a part-timer who never works on Saturday.
SATURDAY_OFF = frozenset(tid for n, tid in enumerate(TEACHER_IDS, start=1) if n % 3 == 0)
# Grades 7-11 need more than 5 x 6 = 30 slots: the sixth day is what makes the plan fit.
OVER_FIVE_DAYS = frozenset({"c-7a", "c-7b", "c-8a", "c-9a", "c-9b", "c-10a", "c-11a", "c-11b"})


def _grade(class_id: str) -> int:
    return int(class_id.removeprefix("c-")[:-1])


def _teacher_for(class_id: str, subject_id: str) -> str:
    if class_id in _HOMEROOM and subject_id in _HOMEROOM_SUBJECTS:
        return _HOMEROOM[class_id]
    for teacher_id, class_ids in _ASSIGN[subject_id].items():
        if "*" in class_ids or class_id in class_ids:
            return teacher_id
    msg = f"no teacher for {class_id} {subject_id}"
    raise AssertionError(msg)


def _rows() -> tuple[CurriculumRow, ...]:
    rows: list[CurriculumRow] = []
    for class_id, _ in CLASSES:
        grade = _grade(class_id)
        plan = _PRIMARY[grade] if grade <= 4 else _SECONDARY[grade]
        rows.extend(
            CurriculumRow(class_id, subject_id, _teacher_for(class_id, subject_id), hours)
            for subject_id, hours in plan.items()
        )
    return tuple(rows)


def build_problem(days: tuple[int, ...] = DAYS) -> Problem:
    return Problem(
        kind=TimetableKind.MAIN,
        school_days=days,
        lessons_per_day=LESSONS_PER_DAY,
        staff=STAFF,
        capacity=Passport(PASSPORT),
        classes=tuple(SchoolClass(cid, n, priority=False) for cid, n in CLASSES),
        subjects=tuple(Subject(sid, prio, outside) for sid, prio, outside in SUBJECTS),
        teachers=tuple(
            Teacher(
                tid,
                WEEKLY_LIMIT,
                frozenset({SATURDAY}) if tid in SATURDAY_OFF else frozenset[int](),
            )
            for tid in TEACHER_IDS
        ),
        rows=_rows(),
    )


@pytest.fixture(scope="module")
def solved() -> tuple[Problem, Result]:
    problem = build_problem()
    return problem, solve(problem, PARAMS)


def test_scenario_is_realistic() -> None:
    problem = build_problem()
    pupils = sum(c.students for c in problem.classes)
    people = pupils + STAFF + len(problem.teachers)
    assert len(problem.classes) == 16
    assert pupils == 348
    assert 0.65 <= PASSPORT / people <= 0.75
    load = Counter[str]()
    hours = Counter[str]()
    for r in problem.rows:
        load[r.teacher_id] += r.hours
        hours[r.class_id] += r.hours
    assert set(load) == set(TEACHER_IDS)
    assert max(load.values()) <= WEEKLY_LIMIT
    assert len(SATURDAY_OFF) == 8
    assert {hours[c] for c in ("c-8a", "c-9a", "c-9b", "c-10a", "c-11a", "c-11b")} == {33}
    assert all(h <= LESSONS_PER_DAY * len(DAYS) for h in hours.values())


@pytest.mark.slow
def test_solves_within_time_limit_and_passes_oracle(solved: tuple[Problem, Result]) -> None:
    problem, result = solved
    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(problem, result) == []
    recomputed = oracle.objective(problem, result)
    assert result.stats.objective == recomputed.value
    assert result.stats.wall_time_s < TIME_LIMIT_S + WALL_TIME_MARGIN_S


@pytest.mark.slow
def test_passport_holds_every_day_including_saturday(solved: tuple[Problem, Result]) -> None:
    problem, result = solved
    for day in DAYS:
        load = oracle.day_load(problem, result, day)
        assert load.people <= PASSPORT, f"day {day}: {load}"
    in_person = [f for f in result.day_formats if f.in_person]
    share = len(in_person) / len(result.day_formats)
    assert share >= 0.5, f"in-person share {share:.2f}"  # prototype: 0.69


@pytest.mark.slow
def test_saturday_is_used_but_never_by_part_timers(solved: tuple[Problem, Result]) -> None:
    _, result = solved
    saturday = [x for x in result.lessons if x.day == SATURDAY]
    assert saturday, "the sixth day must carry lessons: grades 7-11 do not fit in five"
    assert not [x for x in saturday if x.teacher_id in SATURDAY_OFF]
    on_saturday = {x.class_id for x in saturday}
    assert on_saturday >= OVER_FIVE_DAYS


@pytest.mark.slow
def test_row_day_cap_counts_only_available_days(solved: tuple[Problem, Result]) -> None:
    problem, result = solved
    assert not result.stats.quality_relaxed  # the cap is satisfiable here and must be kept
    per_day = Counter((x.class_id, x.subject_id, x.teacher_id, x.day) for x in result.lessons)
    for r in problem.rows:
        n_days = len(DAYS) - (1 if r.teacher_id in SATURDAY_OFF else 0)
        cap = max(ROW_DAY_CAP_MIN, ceil(r.hours / n_days))
        for day in DAYS:
            n = per_day[r.class_id, r.subject_id, r.teacher_id, day]
            assert n <= cap, f"{r} has {n} lessons on day {day}, cap {cap}"


def test_five_day_week_cannot_hold_the_senior_plan() -> None:
    problem = build_problem(days=(1, 2, 3, 4, 5))
    result = solve(problem, PARAMS)
    assert result.status is SolveStatus.INFEASIBLE
    assert {c.rule for c in result.conflicts} == {ConflictRule.CLASS_HOURS_EXCEED_SLOTS}
    assert {c.class_id for c in result.conflicts} == OVER_FIVE_DAYS
