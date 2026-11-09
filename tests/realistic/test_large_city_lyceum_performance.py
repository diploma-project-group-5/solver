"""Realistic archetype: a large city lyceum with 50 classes (performance test, main timetable 1.1).

Kyiv averages 25.6 pupils per class; schools of 1500+ pupils are rare, so this is the upper end
of what the demo must handle: grades 1-6 x 5 parallels + grades 7-11 x 4 = 50 classes, about
1300 pupils, 75 teachers, about 1400 lesson-hours a week, 5 days x 7 lessons. The shelter
passport (700) holds about half of the ~1400 people, so capacity is the binding constraint.

The problem is built deterministically (fixed seed); one 90 s solve on 2 workers is shared by
all checks, and every result is verified by the independent oracle.
"""

import random
from collections import Counter, defaultdict

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
from lad_solver.domain.result import Result, SolveStatus, StopReason

TIME_LIMIT_S = 90.0
WALL_SLACK_S = 5.0  # phase B retries and Python around CP-SAT, as in the other scenarios
PARAMS = Params(time_limit_s=TIME_LIMIT_S, num_workers=2, random_seed=0)
DAYS = (1, 2, 3, 4, 5)
LESSONS_PER_DAY = 7
STAFF = 30
PASSPORT = 700
SEED = 20261113
MAX_LIMIT, MIN_LIMIT = 27, 18
FIRST_SOLUTION_MAX_S = 2.0  # prototype: 0.62 s
MIN_IN_PERSON_SHARE = 0.40  # prototype: 0.48; the passport allows at most ~0.52

LETTERS = "abcde"
LAST_PRIMARY = 4  # grades 1-4: homeroom teacher
PRIORITY_CLASSES = 15  # ~30% of 50 classes

# (subject id, priority educational area, not held in the shelter)
SUBJECTS: tuple[tuple[str, bool, bool], ...] = (
    ("s-ukr", True, False),
    ("s-ulit", False, False),
    ("s-wlit", False, False),
    ("s-eng", False, False),
    ("s-ger", False, False),
    ("s-math", True, False),
    ("s-alg", True, False),
    ("s-geom", True, False),
    ("s-explore", True, False),
    ("s-sci", True, False),
    ("s-bio", True, False),
    ("s-phys", True, False),
    ("s-chem", True, False),
    ("s-geo", False, False),
    ("s-hist", False, False),
    ("s-civics", False, False),
    ("s-info", False, False),
    ("s-tech", False, False),
    ("s-art", False, False),
    ("s-music", False, False),
    ("s-health", False, False),
    ("s-defense", False, False),
    ("s-pe", False, True),
)

# Weekly hours per grade band (Typical Educational Programme, simplified, no group splits).
PRIMARY_HOMEROOM = (
    ("s-ukr", 7),
    ("s-math", 4),
    ("s-explore", 3),
    ("s-art", 1),
    ("s-tech", 1),
    ("s-music", 1),
)
PRIMARY_SPECIALISTS = (("s-eng", 3), ("s-pe", 3))
GRADES_5_6 = (
    ("s-ukr", 4),
    ("s-ulit", 2),
    ("s-wlit", 2),
    ("s-eng", 3),
    ("s-ger", 2),
    ("s-math", 5),
    ("s-hist", 1),
    ("s-sci", 2),
    ("s-info", 1),
    ("s-tech", 1),
    ("s-art", 1),
    ("s-music", 1),
    ("s-pe", 3),
    ("s-health", 1),
)
GRADES_7_9 = (
    ("s-ukr", 3),
    ("s-ulit", 2),
    ("s-wlit", 1),
    ("s-eng", 3),
    ("s-ger", 2),
    ("s-alg", 3),
    ("s-geom", 2),
    ("s-hist", 2),
    ("s-geo", 2),
    ("s-bio", 2),
    ("s-phys", 2),
    ("s-chem", 2),
    ("s-info", 1),
    ("s-tech", 1),
    ("s-art", 1),
    ("s-pe", 3),
)
GRADES_10_11 = (
    ("s-ukr", 2),
    ("s-ulit", 2),
    ("s-wlit", 1),
    ("s-eng", 3),
    ("s-alg", 3),
    ("s-geom", 2),
    ("s-hist", 3),
    ("s-civics", 1),
    ("s-geo", 1),
    ("s-bio", 2),
    ("s-phys", 3),
    ("s-chem", 2),
    ("s-info", 2),
    ("s-defense", 2),
    ("s-art", 1),
    ("s-pe", 3),
)

# Specialty -> (its subjects, number of teachers). One specialty per specialist teacher.
DEPARTMENTS: dict[str, tuple[frozenset[str], int]] = {
    "ukr": (frozenset({"s-ukr", "s-ulit"}), 8),
    "wlit": (frozenset({"s-wlit"}), 2),
    "eng": (frozenset({"s-eng"}), 8),
    "ger": (frozenset({"s-ger"}), 2),
    "math": (frozenset({"s-math", "s-alg", "s-geom"}), 8),
    "hist": (frozenset({"s-hist", "s-civics"}), 3),
    "bio": (frozenset({"s-sci", "s-bio", "s-health"}), 3),
    "geo": (frozenset({"s-geo"}), 2),
    "phys": (frozenset({"s-phys"}), 2),
    "chem": (frozenset({"s-chem"}), 2),
    "info": (frozenset({"s-info"}), 3),
    "tech": (frozenset({"s-tech"}), 1),
    "arts": (frozenset({"s-art", "s-music"}), 2),
    "pe": (frozenset({"s-pe"}), 8),
    "defense": (frozenset({"s-defense"}), 1),
}
HOMEROOM_TEACHERS = 20
UNAVAILABLE_EVERY = 7  # roughly every 7th lightly loaded specialist misses one day
UNAVAILABLE_MAX_LOAD = 21


def _class_ids() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for grade in range(1, 12):
        parallels = 5 if grade <= 6 else 4
        out.extend((f"c-{grade}{LETTERS[k]}", grade) for k in range(parallels))
    return out


def _curriculum(grade: int) -> tuple[tuple[str, int], ...]:
    if grade <= LAST_PRIMARY:
        info = (("s-info", 1),) if grade >= 2 else ()
        return (*PRIMARY_HOMEROOM, *PRIMARY_SPECIALISTS, *info)
    if grade <= 6:
        return GRADES_5_6
    if grade <= 9:
        return GRADES_7_9
    return GRADES_10_11


def _department(subject_id: str) -> str:
    return next(d for d, (subjects, _) in DEPARTMENTS.items() if subject_id in subjects)


def _classes(rng: random.Random, ids: list[tuple[str, int]]) -> tuple[SchoolClass, ...]:
    students = [rng.randint(22, 24) if g <= LAST_PRIMARY else rng.randint(24, 30) for _, g in ids]
    flagged = set(rng.sample([cid for cid, _ in ids], k=PRIORITY_CLASSES))
    return tuple(
        SchoolClass(cid, n, cid in flagged) for (cid, _), n in zip(ids, students, strict=True)
    )


def _specialist_bundles(ids: list[tuple[str, int]]) -> list[tuple[str, str, list[tuple[str, int]]]]:
    """(department, class, its subject-hours) bundles, heaviest first."""
    homeroom = {sid for sid, _ in PRIMARY_HOMEROOM}
    bundles: defaultdict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for cid, grade in ids:
        for sid, hours in _curriculum(grade):
            if grade > LAST_PRIMARY or sid not in homeroom:
                bundles[_department(sid), cid].append((sid, hours))
    ordered = sorted(bundles.items(), key=lambda kv: (-sum(h for _, h in kv[1]), kv[0]))
    return [(dept, cid, items) for (dept, cid), items in ordered]


def _rows(ids: list[tuple[str, int]]) -> tuple[list[CurriculumRow], Counter[str], int]:
    """Curriculum rows, teacher loads and the number of teachers."""
    rows: list[CurriculumRow] = []
    load: Counter[str] = Counter()
    primary = [cid for cid, g in ids if g <= LAST_PRIMARY]
    for n, cid in enumerate(primary, start=1):  # homeroom teachers t-1..t-20
        rows.extend(CurriculumRow(cid, sid, f"t-{n}", h) for sid, h in PRIMARY_HOMEROOM)
        load[f"t-{n}"] += sum(h for _, h in PRIMARY_HOMEROOM)
    staff_by_dept: dict[str, list[str]] = {}
    next_no = HOMEROOM_TEACHERS + 1
    for dept, (_, count) in DEPARTMENTS.items():
        staff_by_dept[dept] = [f"t-{next_no + k}" for k in range(count)]
        next_no += count
    for dept, cid, items in _specialist_bundles(ids):  # least loaded teacher of the specialty
        tid = min(staff_by_dept[dept], key=lambda t: (load[t], int(t[2:])))
        rows.extend(CurriculumRow(cid, sid, tid, h) for sid, h in items)
        load[tid] += sum(h for _, h in items)
    return rows, load, next_no - 1


def _teachers(load: Counter[str], count: int) -> tuple[Teacher, ...]:
    teachers: list[Teacher] = []
    light = 0
    for n in range(1, count + 1):
        tid = f"t-{n}"
        assert load[tid] <= MAX_LIMIT, f"scenario: {tid} has {load[tid]} h"
        limit = min(MAX_LIMIT, max(MIN_LIMIT, load[tid] + n % 3))
        unavailable: frozenset[int] = frozenset()
        if n > HOMEROOM_TEACHERS and load[tid] <= UNAVAILABLE_MAX_LOAD:
            light += 1
            if light % UNAVAILABLE_EVERY == 0:
                unavailable = frozenset({DAYS[(light // UNAVAILABLE_EVERY) % len(DAYS)]})
        teachers.append(Teacher(tid, limit, unavailable))
    return tuple(teachers)


def _build() -> Problem:
    rng = random.Random(SEED)  # noqa: S311 - deterministic fixture data, not cryptography
    ids = _class_ids()
    rows, load, count = _rows(ids)
    return Problem(
        kind=TimetableKind.MAIN,
        school_days=DAYS,
        lessons_per_day=LESSONS_PER_DAY,
        staff=STAFF,
        capacity=Passport(PASSPORT),
        classes=_classes(rng, ids),
        subjects=tuple(Subject(s, priority_area=p, no_shelter=o) for s, p, o in SUBJECTS),
        teachers=_teachers(load, count),
        rows=tuple(rows),
    )


@pytest.fixture(scope="module")
def lyceum() -> Problem:
    return _build()


@pytest.fixture(scope="module")
def solved(lyceum: Problem) -> Result:
    return solve(lyceum, PARAMS)


def _in_person_days(result: Result) -> Counter[str]:
    return Counter(f.class_id for f in result.day_formats if f.in_person)


def test_scenario_matches_archetype_and_passes_prechecks(lyceum: Problem) -> None:
    pupils = sum(c.students for c in lyceum.classes)
    lesson_hours = sum(r.hours for r in lyceum.rows)
    people = pupils + STAFF + len(lyceum.teachers)
    assert len(lyceum.classes) == 50
    assert 1250 <= pupils <= 1350
    assert 74 <= len(lyceum.teachers) <= 80
    assert 1350 <= lesson_hours <= 1450
    assert 0.45 <= PASSPORT / people <= 0.55  # shelter for about half of the building
    assert sum(c.priority for c in lyceum.classes) == PRIORITY_CLASSES
    assert 5 <= sum(bool(t.unavailable_days) for t in lyceum.teachers) <= 10
    assert precheck(lyceum) == ()


@pytest.mark.slow
def test_large_build_is_feasible_valid_and_not_relaxed(lyceum: Problem, solved: Result) -> None:
    assert solved.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(lyceum, solved) == []
    assert not solved.stats.quality_relaxed
    expected_stop = (
        StopReason.COMPLETED if solved.status is SolveStatus.OPTIMAL else StopReason.TIME_LIMIT
    )
    assert solved.stats.stop_reason is expected_stop
    recomputed = oracle.objective(lyceum, solved)
    assert (solved.stats.primary, solved.stats.secondary, solved.stats.objective) == (
        recomputed.primary,
        recomputed.secondary,
        recomputed.value,
    )


@pytest.mark.slow
def test_passport_holds_every_day_and_is_actually_used(lyceum: Problem, solved: Result) -> None:
    pupils = sum(c.students for c in lyceum.classes)
    in_person_pupils = 0
    for day in DAYS:
        load = oracle.day_load(lyceum, solved, day)
        assert load.people <= PASSPORT, f"day {day}: {load}"
        in_person_pupils += load.pupils
    share = in_person_pupils / (pupils * len(DAYS))
    assert share >= MIN_IN_PERSON_SHARE, f"in-person share {share:.3f}"


@pytest.mark.slow
def test_wall_time_and_first_solution_within_budget(solved: Result) -> None:
    assert solved.stats.wall_time_s <= TIME_LIMIT_S + WALL_SLACK_S
    assert solved.stats.first_solution_s is not None
    assert solved.stats.first_solution_s < FIRST_SOLUTION_MAX_S
    assert solved.stats.num_workers == 2


@pytest.mark.slow
def test_unavailable_days_and_priority_flag_are_honoured(lyceum: Problem, solved: Result) -> None:
    away = {t.id: t.unavailable_days for t in lyceum.teachers if t.unavailable_days}
    assert away
    assert not [x for x in solved.lessons if x.day in away.get(x.teacher_id, frozenset())]
    # The priority flag is only a tie-break below W * primary, so "flagged classes get more
    # in-person days on average" is not a guaranteed property; check that it is counted instead.
    days = _in_person_days(solved)
    flagged_days = sum(days[c.id] for c in lyceum.classes if c.priority)
    assert solved.stats.secondary == flagged_days
    assert flagged_days > 0
