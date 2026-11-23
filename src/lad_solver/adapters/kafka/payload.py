"""Inbound event models (pydantic v2), hand-written after the ``contracts`` draft.

Mirrors ``contracts/schemas/events/{envelope,solve.build.requested,solve.substitution.requested}
.v1.json`` (contracts 0.5.0) until the ``contracts`` pip package exists (ADR 2026-11-04 §4).
Tolerant reader: unknown fields are ignored, so personal data that a producer might add by mistake
(class names, teacher signatures) never reaches the engine.
``hide_input_in_errors`` keeps input values out of ``ValidationError`` texts.
"""

import datetime as dt
import re
from datetime import datetime
from typing import Annotated, Final, Literal, Self
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

_CONFIG = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True, strict=False)

MAX_CLASSES = 200
MAX_LESSONS_PER_DAY = 10
MAX_STUDENTS = 60

type Id = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
]
type Day = Annotated[int, Field(ge=1, le=6)]
# Unbounded counts would pass validation and then overflow CP-SAT's int64 while the model is built.
type Count = Annotated[int, Field(ge=0, le=2**63 - 1)]


def _unique_days(days: tuple[int, ...]) -> tuple[int, ...]:
    if len(set(days)) != len(days):
        msg = "days must be unique"
        raise ValueError(msg)
    return tuple(sorted(days))


type Days = Annotated[tuple[Day, ...], Field(max_length=6), AfterValidator(_unique_days)]
type LessonNo = Annotated[int, Field(ge=1, le=MAX_LESSONS_PER_DAY)]

_FULL_DATE: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _full_date(value: object) -> object:
    """RFC 3339 full-date only (contracts ``format: date``), never a datetime or a timestamp."""
    if isinstance(value, str) and _FULL_DATE.fullmatch(value):
        return dt.date.fromisoformat(value)  # 2026-02-30 raises: a validation error
    msg = "a date must be YYYY-MM-DD"
    raise ValueError(msg)


type IsoDate = Annotated[dt.date, BeforeValidator(_full_date)]


def _unique_items[T](items: tuple[T, ...]) -> tuple[T, ...]:
    if len(set(items)) != len(items):
        msg = "items must be unique"
        raise ValueError(msg)
    return items


class _Model(BaseModel):
    model_config = _CONFIG


class JobRef(_Model):
    job_id: UUID


# MAJOR[.MINOR[.PATCH]]: the contract writes MAJOR.MINOR; a reader also takes "1" and "1.0.0".
SCHEMA_VERSION_PATTERN = r"^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*)){0,2}$"
MAX_CORRELATION_ID = 128


class AddressHead(_Model):
    """The minimum needed to answer a job with ``solve.completed``; nothing else is required."""

    school_id: UUID
    data: JobRef


class EnvelopeHead(_Model):
    """The rest of the envelope; a failure here is answered with MODEL_INVALID, not dropped."""

    event_id: UUID
    type: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    schema_version: Annotated[str, StringConstraints(pattern=SCHEMA_VERSION_PATTERN)]
    correlation_id: Annotated[str, StringConstraints(min_length=1, max_length=MAX_CORRELATION_ID)]
    school_id: UUID
    data: JobRef


class PassportCapacity(_Model):
    passport: Count


class AlertCapacity(_Model):
    seats: Count
    passport: Count


class ClassIn(_Model):
    id: Id
    students: Annotated[int, Field(ge=0, le=MAX_STUDENTS)]
    priority: bool


class SubjectIn(_Model):
    id: Id
    priority_area: bool
    no_shelter: bool


class TeacherIn(_Model):
    id: Id
    weekly_limit: Annotated[int, Field(ge=0, le=60)]
    unavailable_days: Days = ()


class RowIn(_Model):
    class_id: Id
    subject_id: Id
    teacher_id: Id
    hours: Annotated[int, Field(ge=1, le=60)]


class BuildRequestedData(_Model):
    job_id: UUID
    kind: Literal["main", "alert"]
    model: Literal["1.1", "1.2"]
    snapshot_version: Count
    time_limit_s: Annotated[int, Field(ge=1, le=3600)]
    queue_deadline_at: AwareDatetime
    school_days: Annotated[Days, Field(min_length=1)]
    lessons_per_day: Annotated[int, Field(ge=1, le=MAX_LESSONS_PER_DAY)]
    staff: Count
    # Alert first: with extra='ignore' a {seats, passport} object would also match PassportCapacity.
    capacity: AlertCapacity | PassportCapacity
    classes: Annotated[tuple[ClassIn, ...], Field(max_length=MAX_CLASSES)]
    subjects: Annotated[tuple[SubjectIn, ...], Field(max_length=500)]
    teachers: Annotated[tuple[TeacherIn, ...], Field(max_length=500)]
    rows: Annotated[tuple[RowIn, ...], Field(max_length=5000)]


class BuildRequested(_Model):
    event_id: UUID
    type: Literal["solve.build.requested"]
    schema_version: str
    source: str | None = None
    time: datetime | None = None
    correlation_id: str
    school_id: UUID
    data: BuildRequestedData


class SubstitutionHead(_Model):
    """Envelope + ``data.job_id`` of a substitution job: the DLQ projection of its topic.

    Who is absent and which lessons never reach the DLQ copy (ADR 2026-11-15-solver-substitution-
    unsupported §4, kept with the engine): a broken job is diagnosed by its job id.
    """

    event_id: UUID
    type: Literal["solve.substitution.requested"]
    schema_version: str
    source: str | None = None
    time: datetime | None = None
    correlation_id: str
    school_id: UUID
    data: JobRef


# ---------------------------------------------------------------- substitution (contracts 0.5.0)

ISO_WEEK_PATTERN: Final = r"^[0-9]{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$"


class LessonIn(_Model):
    class_id: Id
    day: Day
    lesson_no: LessonNo
    subject_id: Id
    teacher_id: Id


class DayFormatIn(_Model):
    class_id: Id
    day: Day
    in_person: bool


class TemplateIn(_Model):
    timetable_id: UUID
    kind: Literal["main", "alert"]
    lessons: Annotated[tuple[LessonIn, ...], Field(max_length=12_000)]
    day_formats: Annotated[tuple[DayFormatIn, ...], Field(max_length=1_200)]


class LessonRefIn(_Model):
    class_id: Id
    lesson_no: LessonNo


class MovedToIn(_Model):
    date: IsoDate
    lesson_no: LessonNo


class ApprovedIn(_Model):
    """An approved overlay of the window; ``kind`` defaults to teacher (the 0.4.x shape)."""

    date: IsoDate
    class_id: Id
    lesson_no: LessonNo
    kind: Literal["teacher", "moved", "self_study", "cancelled"] = "teacher"
    teacher_id: Id | None = None
    subject_id: Id | None = None  # follows from the template; not read
    moved_to: MovedToIn | None = None

    @model_validator(mode="after")
    def _shape_follows_kind(self) -> Self:
        if (self.teacher_id is not None) != (self.kind == "teacher"):
            msg = "teacher_id is required for kind=teacher and not allowed otherwise"
            raise ValueError(msg)
        if (self.moved_to is not None) != (self.kind == "moved"):
            msg = "moved_to is required for kind=moved and not allowed otherwise"
            raise ValueError(msg)
        return self


class AbsenceIn(_Model):
    """An absence of the week: ids only, no reason; ``lesson_nos`` null = the whole date."""

    teacher_id: Id
    date: IsoDate
    lesson_nos: (
        Annotated[tuple[LessonNo, ...], Field(min_length=1), AfterValidator(_unique_items)] | None
    )


class SubstitutionRequestedData(_Model):
    job_id: UUID
    date: IsoDate
    absence_dates: Annotated[
        tuple[IsoDate, ...], Field(min_length=1), AfterValidator(_unique_items)
    ]
    absent_teacher_id: Id
    lessons: Annotated[tuple[LessonRefIn, ...], Field(min_length=1)]
    not_before: IsoDate
    iso_week: Annotated[str, StringConstraints(pattern=ISO_WEEK_PATTERN)]
    snapshot_version: Count
    time_limit_s: Annotated[int, Field(ge=1, le=3600)]
    queue_deadline_at: AwareDatetime
    school_days: Annotated[Days, Field(min_length=1)]
    lessons_per_day: Annotated[int, Field(ge=1, le=MAX_LESSONS_PER_DAY)]
    staff: Count
    capacity: AlertCapacity | PassportCapacity  # alert first, as in BuildRequestedData
    classes: Annotated[tuple[ClassIn, ...], Field(max_length=MAX_CLASSES)]
    subjects: Annotated[tuple[SubjectIn, ...], Field(max_length=500)]
    teachers: Annotated[tuple[TeacherIn, ...], Field(max_length=500)]
    rows: Annotated[tuple[RowIn, ...], Field(max_length=5000)]
    template: TemplateIn
    approved_substitutions: Annotated[tuple[ApprovedIn, ...], Field(max_length=12_000)]
    # 0.5.0, optional: missing absences = only the absent teacher, all of absence_dates; missing
    # non_school_dates = step 3 off. An explicit null is not "missing": the schema refuses it.
    absences: Annotated[tuple[AbsenceIn, ...], Field(max_length=3_500)] | None = None
    non_school_dates: (
        Annotated[tuple[IsoDate, ...], Field(max_length=7), AfterValidator(_unique_items)] | None
    ) = None
    allow_support_moves: bool = False

    @field_validator("absences", "non_school_dates", mode="before")
    @classmethod
    def _not_null(cls, value: object) -> object:
        if value is None:
            msg = "null is not allowed: leave the field out"
            raise ValueError(msg)
        return value


class SubstitutionRequested(_Model):
    event_id: UUID
    type: Literal["solve.substitution.requested"]
    schema_version: str
    source: str | None = None
    time: datetime | None = None
    correlation_id: str
    school_id: UUID
    data: SubstitutionRequestedData
