"""Engine input: a self-contained, PD-free description of one timetable build.

Identifiers are opaque strings (uuids in Lad). No names, signatures or other personal data
live here; the Kafka adapter strips them before building a ``Problem``
(ADR 2026-11-04, spec §04 "minimum PD").
"""

from dataclasses import dataclass, field
from enum import StrEnum


class TimetableKind(StrEnum):
    """Which timetable is built; the capacity model follows from it (spec §04, §15 p.15)."""

    MAIN = "main"  # capacity model 1.1
    ALERT = "alert"  # capacity model 1.2


@dataclass(frozen=True, slots=True)
class Passport:
    """Model 1.1: students + non-teaching staff + teachers on site <= passport capacity."""

    passport: int


@dataclass(frozen=True, slots=True)
class AlertLimits:
    """Model 1.2: students <= equipped seats AND students + staff + teachers on site <= passport."""

    seats: int
    passport: int


type Capacity = Passport | AlertLimits


@dataclass(frozen=True, slots=True)
class SchoolClass:
    id: str
    students: int
    priority: bool  # "priority for in-person learning" flag (secondary objective)


@dataclass(frozen=True, slots=True)
class Subject:
    id: str
    priority_area: bool  # subject belongs to a priority educational area (primary objective)
    no_shelter: bool  # "not held in the shelter": only on remote days, and only in model 1.2


@dataclass(frozen=True, slots=True)
class Teacher:
    id: str
    weekly_limit: int
    unavailable_days: frozenset[int] = field(default_factory=frozenset[int])


@dataclass(frozen=True, slots=True)
class CurriculumRow:
    """Curriculum line: ``hours`` weekly lessons of a subject for a class by one teacher."""

    class_id: str
    subject_id: str
    teacher_id: str
    hours: int


@dataclass(frozen=True, slots=True)
class Problem:
    kind: TimetableKind
    school_days: tuple[int, ...]  # ISO weekday numbers 1..6 (Mon..Sat), ascending
    lessons_per_day: int  # L; for ALERT it comes from the alert bells (ADR 2026-11-04)
    staff: int  # non-teaching staff in the building
    capacity: Capacity
    classes: tuple[SchoolClass, ...]
    subjects: tuple[Subject, ...]
    teachers: tuple[Teacher, ...]
    rows: tuple[CurriculumRow, ...]


@dataclass(frozen=True, slots=True)
class Params:
    """Solver knobs. Server-side bounds are enforced by the caller (config), not by clients."""

    time_limit_s: float
    num_workers: int = 4
    random_seed: int = 0
    row_day_cap_min: int = 2  # cap_r = max(row_day_cap_min, ceil(h_r / available days of teacher))
    phase_b_reserve_s: float | None = None  # default: min(15, 0.15 * time_limit_s)
    max_deterministic_time: float | None = None  # set for reproducible seed generation
    interleave_search: bool = False  # deterministic multi-worker search (seed generation)
    max_memory_mb: int | None = None  # CP-SAT max_memory_in_mb; None = no limit (CLI, tests)
