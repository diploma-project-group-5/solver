"""Inbound event models (pydantic v2), hand-written after the ``contracts`` draft.

Mirrors ``contracts/schemas/events/{envelope,solve.build.requested}.v1.json`` until the
``contracts`` pip package exists (ADR 2026-11-04 §4). Tolerant reader: unknown fields are ignored,
so personal data that a producer might add by mistake (class names, teacher signatures) never
reaches the engine.
``hide_input_in_errors`` keeps input values out of ``ValidationError`` texts.
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

_CONFIG = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True, strict=False)

MAX_CLASSES = 200
MAX_LESSONS_PER_DAY = 10
MAX_STUDENTS = 60

type Id = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
]
type Day = Annotated[int, Field(ge=1, le=6)]


def _unique_days(days: tuple[int, ...]) -> tuple[int, ...]:
    if len(set(days)) != len(days):
        msg = "days must be unique"
        raise ValueError(msg)
    return tuple(sorted(days))


type Days = Annotated[tuple[Day, ...], Field(max_length=6), AfterValidator(_unique_days)]


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
    passport: Annotated[int, Field(ge=0)]


class AlertCapacity(_Model):
    seats: Annotated[int, Field(ge=0)]
    passport: Annotated[int, Field(ge=0)]


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
    snapshot_version: Annotated[int, Field(ge=0)]
    time_limit_s: Annotated[int, Field(ge=1, le=3600)]
    queue_deadline_at: AwareDatetime
    school_days: Annotated[Days, Field(min_length=1)]
    lessons_per_day: Annotated[int, Field(ge=1, le=MAX_LESSONS_PER_DAY)]
    staff: Annotated[int, Field(ge=0)]
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
