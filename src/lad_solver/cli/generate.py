"""Synthetic but realistic Ukrainian school for the CLI and bench (deterministic per seed).

Grades 1-11 with parallels, 20-32 pupils per class, weekly hours close to the typical
curriculum (23 h in grade 1 up to 33 h in upper grades), 9 educational areas of which the
priority ones (spec §04) are mathematics, natural sciences, language and literature, civic and
history. Primary classes have a homeroom teacher; specialists teach 1-2 subjects across classes.
No personal data: every identifier is synthetic.

Two profiles. ``legacy`` (default) is the original generator: CLI, tests and the bench run of
21.11.2026 depend on it, so it must stay byte-for-byte reproducible. ``top2026`` follows the
typical plans in force for 2026/27 (MON orders No 743 for grades 1-4, No 235 as amended by
No 1430 for grades 5-9, No 408 as amended by No 890 for grades 10-11): full weekly hours
including physical education (23-26 / 31-36 / 38 h), the hours "for redistribution" split
half to priority and half to other areas (an assumption: the plans leave it to the school),
8 lessons a day so that 36-38 h fit a 5-day week, the legal class sizes of Law No 463-IX
art. 12 (up to 24 pupils in grades 1-4, up to 30 in 5-11) and about 12 pupils per teacher.
"""

import math
import random
from dataclasses import dataclass

from lad_solver.domain.problem import (
    AlertLimits,
    CurriculumRow,
    Passport,
    Problem,
    SchoolClass,
    Subject,
    Teacher,
    TimetableKind,
)

PRIORITY_AREAS = frozenset({"mathematics", "natural", "language_literature", "civic_history"})
LESSONS_PER_DAY = 7
GRADES = 11
STAFF_SHARE = 0.08  # non-teaching staff as a share of everyone in the building
SEATS_SHARE = 0.9  # equipped seats vs. passport capacity (alert)
PRIORITY_CLASS_SHARE = 0.3
UNAVAILABLE_SHARE = 0.2
PUPILS = (20, 32)
TEACHER_LIMIT = (18, 27)

# subject id -> (educational area, not held in the shelter)
SUBJECTS: dict[str, tuple[str, bool]] = {
    "ukr_lang": ("language_literature", False),
    "ukr_lit": ("language_literature", False),
    "foreign_lang": ("language_literature", False),
    "foreign_lit": ("language_literature", False),
    "math": ("mathematics", False),
    "algebra": ("mathematics", False),
    "geometry": ("mathematics", False),
    "explore_world": ("natural", False),
    "science": ("natural", False),
    "biology": ("natural", False),
    "geography": ("natural", False),
    "physics": ("natural", False),
    "chemistry": ("natural", False),
    "history": ("civic_history", False),
    "civics": ("civic_history", False),
    "informatics": ("informatics", False),
    "arts": ("arts", False),
    "technology": ("technology", True),
    "health": ("social_health", False),
    "physical_education": ("physical_culture", True),
}

_GRADE_1 = {"ukr_lang": 7, "math": 4, "explore_world": 3, "foreign_lang": 2, "informatics": 1,
            "arts": 2, "technology": 1, "physical_education": 3}  # fmt: skip
_GRADE_2 = _GRADE_1 | {"foreign_lang": 3}
_GRADE_3 = _GRADE_2 | {"math": 5}
_GRADE_4 = _GRADE_3 | {"health": 1}
_GRADE_5 = {"ukr_lang": 4, "ukr_lit": 2, "foreign_lit": 2, "foreign_lang": 3, "math": 4,
            "science": 2, "history": 2, "informatics": 1, "arts": 2, "technology": 2, "health": 1,
            "physical_education": 3}  # fmt: skip
_GRADE_6 = _GRADE_5 | {"geography": 2}
_GRADE_7 = {"ukr_lang": 3, "ukr_lit": 2, "foreign_lit": 2, "foreign_lang": 3, "algebra": 3,
            "geometry": 2, "biology": 2, "geography": 2, "physics": 2, "chemistry": 2,
            "history": 2, "informatics": 1, "arts": 1, "technology": 1, "health": 1,
            "physical_education": 3}  # fmt: skip
_GRADE_8 = _GRADE_7 | {"history": 3}
_UPPER = {"ukr_lang": 2, "ukr_lit": 3, "foreign_lit": 2, "foreign_lang": 3, "algebra": 3,
          "geometry": 2, "biology": 2, "geography": 1, "physics": 3, "chemistry": 2, "history": 3,
          "civics": 1, "informatics": 2, "arts": 1, "physical_education": 3}  # fmt: skip
CURRICULUM: dict[int, dict[str, int]] = {
    1: _GRADE_1, 2: _GRADE_2, 3: _GRADE_3, 4: _GRADE_4, 5: _GRADE_5, 6: _GRADE_6,
    7: _GRADE_7, 8: _GRADE_8, 9: _GRADE_8, 10: _UPPER, 11: _UPPER,
}  # fmt: skip

# top2026: integer weekly hours per grade (half-hours of the plans rounded, totals exact).
_T1 = {"ukr_lang": 7, "math": 4, "explore_world": 3, "foreign_lang": 2, "informatics": 1,
       "arts": 2, "technology": 1, "physical_education": 3}  # fmt: skip
_T2 = _T1 | {"foreign_lang": 3, "health": 1}
_T3 = _T2 | {"math": 5}
_T5 = {"ukr_lang": 5, "ukr_lit": 2, "foreign_lit": 1, "foreign_lang": 4, "math": 5,
       "science": 2, "history": 1, "informatics": 2, "arts": 2, "technology": 2, "health": 2,
       "physical_education": 3}  # fmt: skip
_T6 = {"ukr_lang": 5, "ukr_lit": 2, "foreign_lit": 1, "foreign_lang": 4, "math": 5,
       "science": 2, "geography": 1, "history": 2, "informatics": 2, "arts": 3, "technology": 2,
       "health": 2, "physical_education": 3}  # fmt: skip
_T7 = {"ukr_lang": 3, "ukr_lit": 2, "foreign_lit": 1, "foreign_lang": 4, "algebra": 3,
       "geometry": 2, "biology": 2, "geography": 2, "physics": 2, "chemistry": 2, "history": 2,
       "informatics": 2, "arts": 2, "technology": 2, "health": 1,
       "physical_education": 3}  # fmt: skip
_T8 = _T7 | {"foreign_lang": 3, "history": 3, "health": 2}
_T9 = _T8 | {"civics": 1, "technology": 1}
_T10 = {"ukr_lang": 2, "ukr_lit": 2, "foreign_lit": 1, "foreign_lang": 3, "algebra": 4,
        "geometry": 2, "biology": 2, "geography": 1, "physics": 3, "chemistry": 2, "history": 3,
        "civics": 1, "informatics": 2, "arts": 1, "technology": 2, "defense": 3, "health": 1,
        "physical_education": 3}  # fmt: skip
CURRICULUM_TOP2026: dict[int, dict[str, int]] = {
    1: _T1, 2: _T2, 3: _T3, 4: _T3, 5: _T5, 6: _T6, 7: _T7, 8: _T8, 9: _T9, 10: _T10, 11: _T10,
}  # fmt: skip

HOMEROOM = frozenset({"ukr_lang", "math", "explore_world", "arts", "technology", "health"})
PRIMARY_GRADES = 4
SPECIALTIES: tuple[frozenset[str], ...] = (
    frozenset({"ukr_lang", "ukr_lit"}),
    frozenset({"foreign_lang"}),
    frozenset({"foreign_lit", "arts"}),
    frozenset({"math", "algebra", "geometry"}),
    frozenset({"science", "biology", "chemistry"}),
    frozenset({"geography", "health"}),
    frozenset({"physics", "informatics"}),
    frozenset({"history", "civics"}),
    frozenset({"technology"}),
    frozenset({"physical_education"}),
)


# "Defence of Ukraine" (grades 10-11): not a priority area in the government list.
SUBJECTS_TOP2026 = SUBJECTS | {"defense": ("defense", False)}
SPECIALTIES_TOP2026 = (*SPECIALTIES, frozenset({"defense"}))


@dataclass(frozen=True, slots=True)
class Profile:
    curriculum: dict[int, dict[str, int]]
    subjects: dict[str, tuple[str, bool]]
    specialties: tuple[frozenset[str], ...]
    lessons_per_day: int
    pupils_primary: tuple[int, int]
    pupils_upper: tuple[int, int]
    teacher_limit: tuple[int, int]


PROFILES: dict[str, Profile] = {
    "legacy": Profile(CURRICULUM, SUBJECTS, SPECIALTIES, LESSONS_PER_DAY, PUPILS, PUPILS,
                      TEACHER_LIMIT),
    "top2026": Profile(CURRICULUM_TOP2026, SUBJECTS_TOP2026, SPECIALTIES_TOP2026, 8, (18, 24),
                       (20, 30), (14, 24)),
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class _Bundle:
    """Lessons of one class that one teacher takes (one specialty)."""

    class_id: str
    hours: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.hours.values())


def generate_problem(  # noqa: PLR0913 - keyword-only generator knobs
    classes: int,
    shelter_ratio: float,
    *,
    seed: int = 0,
    saturday: bool = False,
    kind: TimetableKind = TimetableKind.MAIN,
    profile: str = "legacy",
) -> Problem:
    if classes < 1 or not math.isfinite(shelter_ratio) or shelter_ratio < 0:
        msg = "classes must be >= 1 and shelter_ratio a finite number >= 0"
        raise ValueError(msg)
    if profile not in PROFILES:
        msg = f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}"
        raise ValueError(msg)
    prof = PROFILES[profile]
    rng = random.Random(seed)  # noqa: S311 - synthetic data, not cryptography
    days = (1, 2, 3, 4, 5, 6) if saturday else (1, 2, 3, 4, 5)
    school_classes, grades = _classes(classes, rng, prof)
    bundles = _bundles(school_classes, grades, prof)
    teachers, rows = _staff_teachers(bundles, days, rng, prof)
    pupils = sum(c.students for c in school_classes)
    others = pupils + len(teachers)
    staff = round(STAFF_SHARE * others / (1 - STAFF_SHARE))
    passport = round(shelter_ratio * (others + staff))
    capacity = (
        AlertLimits(seats=round(SEATS_SHARE * passport), passport=passport)
        if kind is TimetableKind.ALERT
        else Passport(passport)
    )
    return Problem(
        kind=kind,
        school_days=days,
        lessons_per_day=prof.lessons_per_day,
        staff=staff,
        capacity=capacity,
        classes=school_classes,
        subjects=tuple(
            Subject(sid, area in PRIORITY_AREAS, no_shelter)
            for sid, (area, no_shelter) in prof.subjects.items()
        ),
        teachers=teachers,
        rows=rows,
    )


def _classes(
    n: int, rng: random.Random, prof: Profile
) -> tuple[tuple[SchoolClass, ...], dict[str, int]]:
    per_grade: dict[int, int] = {}
    for i in range(n):
        grade = 1 + i * GRADES // n
        per_grade[grade] = per_grade.get(grade, 0) + 1
    out: list[SchoolClass] = []
    grades: dict[str, int] = {}
    for grade, count in sorted(per_grade.items()):
        for k in range(count):
            class_id = f"c{grade:02d}-{chr(ord('a') + k)}"
            grades[class_id] = grade
            pupils = prof.pupils_primary if grade <= PRIMARY_GRADES else prof.pupils_upper
            out.append(
                SchoolClass(class_id, rng.randint(*pupils), rng.random() < PRIORITY_CLASS_SHARE)
            )
    return tuple(out), grades


def _bundles(
    classes: tuple[SchoolClass, ...], grades: dict[str, int], prof: Profile
) -> list[list[_Bundle]]:
    """Per specialty (homeroom first): what each class needs from it."""
    homeroom: list[_Bundle] = []
    by_specialty: list[list[_Bundle]] = [[] for _ in prof.specialties]
    for c in classes:
        plan = prof.curriculum[grades[c.id]]
        primary = grades[c.id] <= PRIMARY_GRADES
        if primary:
            homeroom.append(_Bundle(c.id, {s: h for s, h in plan.items() if s in HOMEROOM}))
        for i, specialty in enumerate(prof.specialties):
            hours = {
                s: h for s, h in plan.items() if s in specialty and not (primary and s in HOMEROOM)
            }
            if hours:
                by_specialty[i].append(_Bundle(c.id, hours))
    return [[b] for b in homeroom] + by_specialty


@dataclass(slots=True)
class _Staffing:
    days: tuple[int, ...]
    lessons_per_day: int
    rng: random.Random
    teachers: list[Teacher]
    rows: list[CurriculumRow]

    def hire(self, bundles: list[_Bundle], limit: int) -> None:
        teacher_id = f"t{len(self.teachers) + 1:03d}"
        load = sum(b.total for b in bundles)
        fits_without_a_day = load <= self.lessons_per_day * (len(self.days) - 1)
        unavailable = (
            frozenset({self.rng.choice(self.days)})
            if self.rng.random() < UNAVAILABLE_SHARE and fits_without_a_day
            else frozenset[int]()
        )
        self.teachers.append(Teacher(teacher_id, max(limit, load), unavailable))
        self.rows.extend(
            CurriculumRow(b.class_id, subject, teacher_id, hours)
            for b in bundles
            for subject, hours in b.hours.items()
        )


def _staff_teachers(
    groups: list[list[_Bundle]], days: tuple[int, ...], rng: random.Random, prof: Profile
) -> tuple[tuple[Teacher, ...], tuple[CurriculumRow, ...]]:
    staffing = _Staffing(days, prof.lessons_per_day, rng, [], [])
    for group in groups:
        current: list[_Bundle] = []
        limit = rng.randint(*prof.teacher_limit)
        for bundle in group:
            if current and sum(b.total for b in current) + bundle.total > limit:
                staffing.hire(current, limit)
                current, limit = [], rng.randint(*prof.teacher_limit)
            current.append(bundle)
        if current:
            staffing.hire(current, limit)
    return tuple(staffing.teachers), tuple(staffing.rows)
