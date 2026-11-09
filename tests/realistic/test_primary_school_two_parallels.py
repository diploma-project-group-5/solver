"""Realistic archetype: primary school, grades 1-4, two parallels, short 5-lesson grid.

Based on Art. 12 of the Law "On Complete General Secondary Education" (at most 24 pupils in
grades 1-4), weekly limits 20/22/23/23 h, 1-4 grades in the first shift, and the MON guidance
that grades 1-4 get priority in the shelter. Shelter capacity is ~60% of the people on site.

Stresses: grades 3-4 have only 2 spare slots per week (23 h in 25 slots); two specialists
(foreign language, PE) teach every class, so their daily format couples every class they meet
that day; a part-time informatics teacher squeezes 6 h into 3 days.
"""

from collections import Counter, defaultdict
from dataclasses import replace

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

TIME_LIMIT_S = 30.0
PARAMS = Params(time_limit_s=TIME_LIMIT_S, num_workers=2, random_seed=0)
DAYS = (1, 2, 3, 4, 5)
LESSONS_PER_DAY = 5
STAFF = 8
PASSPORT = 116  # ~60% of 175 pupils + 11 teachers + 8 staff = 194 people

# (class id, grade, pupils); every class is at most 24 pupils (Art. 12).
CLASSES: tuple[tuple[str, int, int], ...] = (
    ("c-1a", 1, 24),
    ("c-1b", 1, 22),
    ("c-2a", 2, 23),
    ("c-2b", 2, 21),
    ("c-3a", 3, 24),
    ("c-3b", 3, 20),
    ("c-4a", 4, 22),
    ("c-4b", 4, 19),
)
GRADE_ONE = frozenset({"c-1a", "c-1b"})

HOMEROOM = {cid: f"t-{i:02d}" for i, (cid, _, _) in enumerate(CLASSES, start=1)}
T_FLANG = "t-09"
T_PE = "t-10"
T_INF = "t-11"
INF_UNAVAILABLE = frozenset({4, 5})

# Homeroom subjects per grade: ukr, math, explore, arts, tech (15 / 15 / 16 / 16 h).
HOMEROOM_PLAN: dict[int, tuple[tuple[str, int], ...]] = {
    1: (("s-ukr", 7), ("s-math", 4), ("s-explore", 2), ("s-arts", 1), ("s-tech", 1)),
    2: (("s-ukr", 7), ("s-math", 4), ("s-explore", 2), ("s-arts", 1), ("s-tech", 1)),
    3: (("s-ukr", 7), ("s-math", 5), ("s-explore", 2), ("s-arts", 1), ("s-tech", 1)),
    4: (("s-ukr", 7), ("s-math", 5), ("s-explore", 2), ("s-arts", 1), ("s-tech", 1)),
}
FLANG_HOURS = {1: 2, 2: 3, 3: 3, 4: 3}  # 2*2 + 6*3 = 22 h
PE_HOURS = 3  # 3 h x 8 classes = 24 h
INF_HOURS = {2: 1, 3: 1, 4: 1}  # grades 2-4 only: 6 h
GRADE_HOURS = {1: 20, 2: 22, 3: 23, 4: 23}


def _subjects() -> tuple[Subject, ...]:
    def s(sid: str, *, prio: bool, outside: bool = False) -> Subject:
        return Subject(sid, priority_area=prio, no_shelter=outside)

    return (
        s("s-ukr", prio=True),
        s("s-math", prio=True),
        s("s-explore", prio=True),
        s("s-flang", prio=True),
        s("s-arts", prio=False),
        s("s-tech", prio=False),
        s("s-inf", prio=False),
        s("s-pe", prio=False, outside=True),  # only matters for 1.2; this is a main timetable
    )


def _rows() -> tuple[CurriculumRow, ...]:
    rows: list[CurriculumRow] = []
    for cid, grade, _ in CLASSES:
        rows.extend(CurriculumRow(cid, sid, HOMEROOM[cid], h) for sid, h in HOMEROOM_PLAN[grade])
        rows.append(CurriculumRow(cid, "s-flang", T_FLANG, FLANG_HOURS[grade]))
        rows.append(CurriculumRow(cid, "s-pe", T_PE, PE_HOURS))
        if grade in INF_HOURS:
            rows.append(CurriculumRow(cid, "s-inf", T_INF, INF_HOURS[grade]))
    return tuple(rows)


def build_problem() -> Problem:
    teachers = (
        *(Teacher(tid, 18) for tid in HOMEROOM.values()),
        Teacher(T_FLANG, 24),
        Teacher(T_PE, 24),
        Teacher(T_INF, 8, INF_UNAVAILABLE),
    )
    return Problem(
        kind=TimetableKind.MAIN,
        school_days=DAYS,
        lessons_per_day=LESSONS_PER_DAY,
        staff=STAFF,
        capacity=Passport(PASSPORT),
        classes=tuple(SchoolClass(cid, n, cid in GRADE_ONE) for cid, _, n in CLASSES),
        subjects=_subjects(),
        teachers=teachers,
        rows=_rows(),
    )


@pytest.fixture(scope="module")
def school() -> Problem:
    problem = build_problem()
    _check_archetype(problem)
    return problem


def _check_archetype(problem: Problem) -> None:
    """Guard the scenario itself, so a typo in the curriculum cannot pass as a solver result."""
    hours = Counter[str]()
    load = Counter[str]()
    for r in problem.rows:
        hours[r.class_id] += r.hours
        load[r.teacher_id] += r.hours
    assert {cid: hours[cid] for cid, _, _ in CLASSES} == {
        cid: GRADE_HOURS[g] for cid, g, _ in CLASSES
    }
    assert sum(c.students for c in problem.classes) == 175
    assert max(c.students for c in problem.classes) <= 24
    assert (load[T_FLANG], load[T_PE], load[T_INF]) == (22, 24, 6)
    assert all(load[t] in {15, 16} for t in HOMEROOM.values())


@pytest.fixture(scope="module")
def solved(school: Problem) -> Result:
    return solve(school, PARAMS)  # one 30 s build shared by the property tests below


@pytest.mark.slow
def test_main_timetable_is_feasible_and_admissible(school: Problem, solved: Result) -> None:
    assert solved.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(school, solved) == []
    recomputed = oracle.objective(school, solved)
    assert solved.stats.objective == recomputed.value
    assert not solved.stats.quality_relaxed
    assert solved.stats.first_solution_s is not None
    assert solved.stats.first_solution_s < 5.0  # prototype: 0.06 s
    assert solved.stats.wall_time_s <= TIME_LIMIT_S + 2.0
    share = len(oracle.in_person_days(solved)) / (len(school.classes) * len(DAYS))
    assert share >= 0.4  # ~0.5 expected under a 60% shelter


@pytest.mark.slow
def test_passport_holds_every_day(school: Problem, solved: Result) -> None:
    for d in DAYS:
        load = oracle.day_load(school, solved, d)
        assert load.people <= PASSPORT, f"day {d}: {load}"


@pytest.mark.slow
def test_specialists_teach_in_one_format_per_day(solved: Result) -> None:
    in_person = oracle.in_person_days(solved)
    formats: defaultdict[tuple[str, int], set[bool]] = defaultdict(set)
    for x in solved.lessons:
        if x.teacher_id in {T_FLANG, T_PE}:
            formats[x.teacher_id, x.day].add((x.class_id, x.day) in in_person)
    assert formats  # both specialists actually teach
    assert all(len(v) == 1 for v in formats.values()), dict(formats)


@pytest.mark.slow
def test_part_time_informatics_fits_its_three_days(solved: Result) -> None:
    days = Counter(x.day for x in solved.lessons if x.teacher_id == T_INF)
    assert sum(days.values()) == 6
    assert not set(days) & INF_UNAVAILABLE


@pytest.mark.slow
def test_no_remote_grade_one_day_could_be_made_in_person_alone(
    school: Problem, solved: Result
) -> None:
    # Flipping a remote grade-1 day with lessons raises at least the secondary objective, so
    # a single admissible flip means the engine left an obvious gain on the table.
    in_person = oracle.in_person_days(solved)
    students = {c.id: c.students for c in school.classes}
    for cid in sorted(GRADE_ONE):
        for d in DAYS:
            mine = {x.teacher_id for x in solved.lessons if x.class_id == cid and x.day == d}
            if (cid, d) in in_person or not mine:
                continue
            consistent = all(
                (x.class_id, d) in in_person
                for x in solved.lessons
                if x.teacher_id in mine and x.day == d and x.class_id != cid
            )
            on_site = {
                x.teacher_id for x in solved.lessons if x.day == d and (x.class_id, d) in in_person
            }
            people = oracle.day_load(school, solved, d).people + students[cid]
            people += len(mine - on_site)
            assert not (consistent and people <= PASSPORT), f"{cid} day {d}: {people} people"


def test_part_timer_left_with_one_day_is_rejected_before_cp_sat(school: Problem) -> None:
    teachers = tuple(
        replace(t, unavailable_days=frozenset({2, 3, 4, 5})) if t.id == T_INF else t
        for t in school.teachers
    )
    result = solve(replace(school, teachers=teachers), PARAMS)  # 6 h > 5 slots of day 1
    assert result.status is SolveStatus.INFEASIBLE
    rules = {(c.rule, c.teacher_id) for c in result.conflicts}
    assert (ConflictRule.TEACHER_HOURS_EXCEED_AVAILABLE_SLOTS, T_INF) in rules
    assert result.lessons == ()
