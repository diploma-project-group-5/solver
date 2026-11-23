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


class CoverageVerdict(StrEnum):
    """What the coverage step (ADR 2026-11-14) did with the ladder's plan."""

    NOT_NEEDED = "not_needed"  # every class already has an in-person day
    NO_TIME = "no_time"  # the ladder left no time for the run
    STOPPED = "stopped"  # external stop before or during the run: the ladder's plan stays
    KEPT = "kept"  # the run found no better plan: the ladder's plan stays
    ACCEPTED = "accepted"  # fewer classes without an in-person day, objective kept


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
    # Engine telemetry for the worker logs, not part of solve.completed: counts and enums only.
    coverage_verdict: CoverageVerdict | None = None  # None: phase A found no plan
    coverage_status: SolveStatus | None = None  # status of the coverage run, if it ran
    zero_day_before: int | None = None  # classes without an in-person day before the step
    zero_day_after: int | None = None  # ... in the plan the run found (None: no plan)
    phase_b_retry_days: int = 0  # days on which phase B ran the feasibility-only retry


@dataclass(frozen=True, slots=True)
class Result:
    status: SolveStatus
    lessons: tuple[Lesson, ...] = ()
    day_formats: tuple[DayFormat, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    stats: Stats = field(default_factory=Stats)
