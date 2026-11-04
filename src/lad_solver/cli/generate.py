"""Synthetic but realistic Ukrainian school for the CLI and bench (deterministic per seed).

Grades 1-11 with parallels, 20-32 pupils per class, weekly hours close to the typical
curriculum (23 h in grade 1 up to 33 h in upper grades), 9 educational areas of which the
priority ones (spec §04) are mathematics, natural sciences, language and literature, civic and
history. Primary classes have a homeroom teacher; specialists teach 1-2 subjects across classes.
No personal data: every identifier is synthetic.
"""

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


@dataclass(frozen=True, slots=True)
class _Bundle:
    """Lessons of one class that one teacher takes (one specialty)."""

    class_id: str
    hours: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.hours.values())


def generate_problem(
    classes: int,
    shelter_ratio: float,
    *,
    seed: int = 0,
    saturday: bool = False,
    kind: TimetableKind = TimetableKind.MAIN,
) -> Problem:
    if classes < 1 or shelter_ratio < 0:
        msg = "classes must be >= 1 and shelter_ratio >= 0"
        raise ValueError(msg)
    rng = random.Random(seed)  # noqa: S311 - synthetic data, not cryptography
    days = (1, 2, 3, 4, 5, 6) if saturday else (1, 2, 3, 4, 5)
    school_classes, grades = _classes(classes, rng)
    bundles = _bundles(school_classes, grades)
    teachers, rows = _staff_teachers(bundles, days, rng)
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
        lessons_per_day=LESSONS_PER_DAY,
        staff=staff,
        capacity=capacity,
        classes=school_classes,
        subjects=tuple(
            Subject(sid, area in PRIORITY_AREAS, no_shelter)
            for sid, (area, no_shelter) in SUBJECTS.items()
        ),
        teachers=teachers,
        rows=rows,
    )


def _classes(n: int, rng: random.Random) -> tuple[tuple[SchoolClass, ...], dict[str, int]]:
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
            out.append(
                SchoolClass(class_id, rng.randint(*PUPILS), rng.random() < PRIORITY_CLASS_SHARE)
            )
    return tuple(out), grades


def _bundles(classes: tuple[SchoolClass, ...], grades: dict[str, int]) -> list[list[_Bundle]]:
    """Per specialty (homeroom first): what each class needs from it."""
    homeroom: list[_Bundle] = []
    by_specialty: list[list[_Bundle]] = [[] for _ in SPECIALTIES]
    for c in classes:
        plan = CURRICULUM[grades[c.id]]
        primary = grades[c.id] <= PRIMARY_GRADES
        if primary:
            homeroom.append(_Bundle(c.id, {s: h for s, h in plan.items() if s in HOMEROOM}))
        for i, specialty in enumerate(SPECIALTIES):
            hours = {
                s: h for s, h in plan.items() if s in specialty and not (primary and s in HOMEROOM)
            }
            if hours:
                by_specialty[i].append(_Bundle(c.id, hours))
    return [[b] for b in homeroom] + by_specialty


@dataclass(slots=True)
class _Staffing:
    days: tuple[int, ...]
    rng: random.Random
    teachers: list[Teacher]
    rows: list[CurriculumRow]

    def hire(self, bundles: list[_Bundle], limit: int) -> None:
        teacher_id = f"t{len(self.teachers) + 1:03d}"
        load = sum(b.total for b in bundles)
        fits_without_a_day = load <= LESSONS_PER_DAY * (len(self.days) - 1)
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
    groups: list[list[_Bundle]], days: tuple[int, ...], rng: random.Random
) -> tuple[tuple[Teacher, ...], tuple[CurriculumRow, ...]]:
    staffing = _Staffing(days, rng, [], [])
    for group in groups:
        current: list[_Bundle] = []
        limit = rng.randint(*TEACHER_LIMIT)
        for bundle in group:
            if current and sum(b.total for b in current) + bundle.total > limit:
                staffing.hire(current, limit)
                current, limit = [], rng.randint(*TEACHER_LIMIT)
            current.append(bundle)
        if current:
            staffing.hire(current, limit)
    return tuple(staffing.teachers), tuple(staffing.rows)
