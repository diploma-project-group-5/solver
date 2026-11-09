"""Engine output. Mirrors ``solve.completed`` semantics (spec §12) without any transport details."""

from dataclasses import dataclass, field
from enum import StrEnum


class SolveStatus(StrEnum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN = "UNKNOWN"
    MODEL_INVALID = "MODEL_INVALID"
    EXPIRED = "EXPIRED"  # set by the worker, never by the engine


class ConflictRule(StrEnum):
    """Rule vocabulary of solver pre-checks (ADR 2026-11-04 §2); validator-lib adds its own."""

    STAFF_OVER_PASSPORT = "staff_over_passport"
    CLASS_HOURS_EXCEED_SLOTS = "class_hours_exceed_slots"
    TEACHER_WEEKLY_LIMIT = "teacher_weekly_limit"
    TEACHER_HOURS_EXCEED_AVAILABLE_SLOTS = "teacher_hours_exceed_available_slots"
    TEACHER_UNAVAILABLE_ALL_DAYS = "teacher_unavailable_all_days"
    ROW_DAY_CAP = "row_day_cap"
    ALERT_BELLS = "alert_bells"
    PACKING = "packing"
    INVALID_INPUT = "invalid_input"


class CapacityCondition(StrEnum):
    SEATS = "seats"
    PASSPORT = "passport"


class StopReason(StrEnum):
    COMPLETED = "completed"  # search finished (optimal or infeasible proven)
    TIME_LIMIT = "time_limit"
    STOPPED = "stopped"  # external stop (SIGTERM)
    NO_IMPROVEMENT = "no_improvement"  # phase A stopped: the objective stopped improving
    PRECHECK = "precheck"  # rejected before CP-SAT


@dataclass(frozen=True, slots=True)
class Conflict:
    rule: ConflictRule
    class_id: str | None = None
    teacher_id: str | None = None
    subject_id: str | None = None
    day: int | None = None
    condition: CapacityCondition | None = None
    hint: bool = False  # heuristic suggestion (e.g. busiest teachers for ``packing``), not a proof


@dataclass(frozen=True, slots=True)
class Lesson:
    class_id: str
    day: int
    lesson_no: int  # 1..lessons_per_day
    subject_id: str
    teacher_id: str


@dataclass(frozen=True, slots=True)
class DayFormat:
    class_id: str
    day: int
    in_person: bool


@dataclass(frozen=True, slots=True)
class Stats:
    objective: int | None = None  # W * primary + secondary of phase A
    best_bound: float | None = None
    primary: int | None = None
    secondary: int | None = None
    wall_time_s: float = 0.0
    first_solution_s: float | None = None  # wall seconds from start to the first phase A solution
    last_improvement_s: float | None = None  # wall seconds from start to the last improvement
    phase_a_status: str | None = None
    quality_relaxed: bool = False  # row-day cap was dropped to find a solution
    stop_reason: StopReason = StopReason.COMPLETED
    num_workers: int = 0
    random_seed: int = 0


@dataclass(frozen=True, slots=True)
class Result:
    status: SolveStatus
    lessons: tuple[Lesson, ...] = ()
    day_formats: tuple[DayFormat, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    stats: Stats = field(default_factory=Stats)
