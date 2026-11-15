"""Payload <-> engine mapping: parse and classify inbound events, build outbound ones.

Pure functions (the clock is passed in), so every branch of the worker's decision table is
unit-tested here without Kafka.
"""

import hashlib
import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import cache
from types import UnionType
from typing import Annotated, Any, Final, Protocol, TypeAliasType, Union, get_args, get_origin
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from lad_solver import __version__
from lad_solver.adapters.kafka.payload import (
    MAX_CORRELATION_ID,
    SCHEMA_VERSION_PATTERN,
    AddressHead,
    AlertCapacity,
    BuildRequested,
    BuildRequestedData,
    ClassIn,
    EnvelopeHead,
    Id,
    PassportCapacity,
    RowIn,
    SubjectIn,
    SubstitutionHead,
    SubstitutionRequested,
    SubstitutionRequestedData,
    TeacherIn,
)
from lad_solver.adapters.kafka.ports import InboundMessage, OutboundMessage
from lad_solver.domain.problem import (
    AlertLimits,
    Capacity,
    CurriculumRow,
    Params,
    Passport,
    Problem,
    SchoolClass,
    Subject,
    Teacher,
    TimetableKind,
)
from lad_solver.domain.result import DayFormat, Lesson, Result, SolveStatus
from lad_solver.domain.substitution import (
    Absence,
    Hour,
    InvalidRequestError,
    Overlay,
    OverlayKind,
    Proposal,
    ProposalKind,
    Slot,
    SubstitutionParams,
    SubstitutionRequest,
    SubstitutionResult,
    baseline_week,
    iso_week_dates,
)

BUILD_REQUESTED: Final = "solve.build.requested"
SUBSTITUTION_REQUESTED: Final = "solve.substitution.requested"
STARTED: Final = "solve.started"
COMPLETED: Final = "solve.completed"
SCHEMA_VERSION: Final = "1.0"
SUPPORTED_MAJOR: Final = 1
SOURCE: Final = "solver"
# Formulation of the CP-SAT models: phase A by days (a), its coverage step (a2), phase B per day
# (b) (ADR 2026-11-04 §1, 2026-11-14) and the substitution steps 1-3 (s, ADR 2026-11-15-solver-
# substitution-engine); the date is the last change of any of them.
MODEL_VERSION: Final = "a+a2+b+s/2026-11-15"
# solve.status substitution.proposals.maxItems (a static bound: one teacher's day is <= 10 rows).
MAX_PROPOSALS: Final = 600

# Fixed namespace: a re-sent event keeps its event_id, so timetable's inbox drops the duplicate.
EVENT_NAMESPACE: Final = UUID("5a0c6f2e-9d1b-4c3a-8e7f-6b5d4c3a2e1f")

_MODEL_FOR_KIND: Final = {"main": "1.1", "alert": "1.2"}


class ErrorCode(StrEnum):
    """The codes this solver sends. The contract also keeps ``unsupported_job_type``: solvers
    before 0.4.0 refused substitutions with it, and future job types may (ADR 2026-11-15)."""

    INVALID_PAYLOAD = "invalid_payload"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
    RESULT_TOO_LARGE = "result_too_large"
    INTERNAL = "internal"


# Errors caused by the request itself: the answer names the invalid_input rule in conflicts.
_INPUT_ERRORS: Final = frozenset({ErrorCode.INVALID_PAYLOAD, ErrorCode.UNSUPPORTED_SCHEMA_VERSION})


class InvalidInputError(ValueError):
    """Semantically invalid request (dangling id, kind/model mismatch). Message has no values."""


@dataclass(frozen=True, slots=True)
class Address:
    """Where solve.status events for a job go; all values come from the inbound envelope."""

    job_id: UUID
    school_id: UUID
    correlation_id: str

    @property
    def key(self) -> bytes:
        return str(self.school_id).encode()


@dataclass(frozen=True, slots=True)
class Poison:
    """No job to answer: only the DLQ copy and a commit."""

    error_class: str


@dataclass(frozen=True, slots=True)
class Rejected:
    """A job that is answered with MODEL_INVALID without solving (one attempt, no retries)."""

    address: Address
    code: ErrorCode
    error_class: str


@dataclass(frozen=True, slots=True)
class Accepted:
    """A build job to solve."""

    address: Address
    data: BuildRequestedData
    problem: Problem
    input_sha256: str
    requested_at: datetime | None

    @property
    def queue_deadline_at(self) -> datetime:
        return self.data.queue_deadline_at


@dataclass(frozen=True, slots=True)
class SubstitutionAccepted:
    """A substitution job to solve: the request is consistent (its baseline week builds)."""

    address: Address
    data: SubstitutionRequestedData
    request: SubstitutionRequest
    input_sha256: str
    requested_at: datetime | None

    @property
    def queue_deadline_at(self) -> datetime:
        return self.data.queue_deadline_at


type ParseOutcome = Poison | Rejected | Accepted | SubstitutionAccepted


def event_id(job_id: UUID, event_type: str) -> UUID:
    return uuid5(EVENT_NAMESPACE, f"{job_id}:{event_type}")


# ---------------------------------------------------------------- inbound


_MAX_INT_CHARS: Final = 64  # far beyond any int64; json would refuse > 4300 digits outright


def _parse_int(literal: str) -> int | str:
    """A huge JSON integer stays a string, so validation rejects it as out of range instead of
    json.loads failing (which would make a valid, addressable job look like non-JSON)."""
    return int(literal) if len(literal) <= _MAX_INT_CHARS else literal


def _load(raw: bytes | None) -> dict[str, object] | Poison:
    try:
        document: object = json.loads(raw, parse_int=_parse_int) if raw is not None else None
    except ValueError, RecursionError:
        return Poison("not_json")
    if not isinstance(document, dict):
        return Poison("not_json_object")
    return document  # pyright: ignore[reportUnknownVariableType]


_SCHEMA_VERSION: Final = re.compile(SCHEMA_VERSION_PATTERN)


def _is_utf8(text: str) -> bool:
    """False for lone surrogates: json.loads accepts "\\ud800", but it cannot be encoded back."""
    try:
        text.encode()
    except UnicodeEncodeError:
        return False
    return True


def _address(document: dict[str, object]) -> Address | None:
    """``school_id`` + ``data.job_id``; a missing or invalid correlation_id falls back to the job id
    (the outbound envelope requires a non-empty one, and the job id still links the logs)."""
    try:
        head = AddressHead.model_validate(document)
    except ValidationError:
        return None
    correlation = document.get("correlation_id")
    if not (
        isinstance(correlation, str)
        and 1 <= len(correlation) <= MAX_CORRELATION_ID
        and _is_utf8(correlation)
    ):
        correlation = str(head.data.job_id)
    return Address(head.data.job_id, head.school_id, correlation)


def address_of(raw: bytes | None) -> Address | None:
    """Best-effort address of a raw message, to answer a job after a parser bug. Never raises."""
    try:
        document = _load(raw)
        return None if isinstance(document, Poison) else _address(document)
    except Exception:  # noqa: BLE001 -- last resort after a parser bug; the caller logs the bug
        return None


def _envelope_rejection(
    document: dict[str, object], expected_type: str
) -> tuple[ErrorCode, str] | None:
    version = document.get("schema_version")
    if not isinstance(version, str) or _SCHEMA_VERSION.fullmatch(version) is None:
        return ErrorCode.INVALID_PAYLOAD, "schema_version"
    # A string compare is exact (the pattern forbids leading zeros) and never parses a huge int.
    if version.split(".", 1)[0] != str(SUPPORTED_MAJOR):
        return ErrorCode.UNSUPPORTED_SCHEMA_VERSION, "schema_version"
    try:
        head = EnvelopeHead.model_validate(document)
    except ValidationError:
        return ErrorCode.INVALID_PAYLOAD, "invalid_envelope"
    if head.type != expected_type:
        return ErrorCode.INVALID_PAYLOAD, "unexpected_type"
    return None


def parse_build_request(raw: bytes | None) -> ParseOutcome:
    """Classify an inbound message: Poison (DLQ only), Rejected (MODEL_INVALID) or Accepted.

    Only a message without an address (``school_id`` + ``data.job_id``) is Poison; every other
    failure is answered, so timetable closes the job at once instead of waiting for its watchdog.
    """
    document = _load(raw)
    if isinstance(document, Poison):
        return document
    address = _address(document)
    if address is None:
        return Poison("invalid_envelope")
    rejection = _envelope_rejection(document, BUILD_REQUESTED)
    if rejection is None:
        try:
            event = BuildRequested.model_validate(document)
            problem = to_problem(event.data)
            digest = input_sha256(event.data)  # encodes every validated string
        except ValidationError:
            rejection = (ErrorCode.INVALID_PAYLOAD, "validation")
        except InvalidInputError:
            rejection = (ErrorCode.INVALID_PAYLOAD, "referential_integrity")
        except UnicodeEncodeError:
            rejection = (ErrorCode.INVALID_PAYLOAD, "invalid_unicode")
        else:
            return Accepted(address, event.data, problem, digest, event.time)
    return Rejected(address, *rejection)


def parse_substitution_request(raw: bytes | None) -> Poison | Rejected | SubstitutionAccepted:
    """Classify a substitution job as a build job is classified (contracts 0.5.0).

    The request must describe one consistent week (its baseline week builds) before
    ``solve.started`` is promised; otherwise the job is answered with invalid_payload.
    """
    document = _load(raw)
    if isinstance(document, Poison):
        return document
    address = _address(document)
    if address is None:
        return Poison("invalid_envelope")
    rejection = _envelope_rejection(document, SUBSTITUTION_REQUESTED)
    if rejection is None:
        try:
            event = SubstitutionRequested.model_validate(document)
            request = to_substitution(event.data)
            baseline_week(request)
            digest = input_sha256(event.data)
        except ValidationError:
            rejection = (ErrorCode.INVALID_PAYLOAD, "validation")
        except InvalidInputError, InvalidRequestError:
            rejection = (ErrorCode.INVALID_PAYLOAD, "referential_integrity")
        except UnicodeEncodeError:
            rejection = (ErrorCode.INVALID_PAYLOAD, "invalid_unicode")
        else:
            return SubstitutionAccepted(address, event.data, request, digest, event.time)
    return Rejected(address, *rejection)


def _unique(kind: str, ids: list[str]) -> set[str]:
    unique = set(ids)
    if len(unique) != len(ids):
        msg = f"duplicate {kind} id"
        raise InvalidInputError(msg)
    return unique


def to_problem(data: BuildRequestedData) -> Problem:
    if _MODEL_FOR_KIND[data.kind] != data.model:
        msg = "model does not match kind"
        raise InvalidInputError(msg)
    return _school(data, TimetableKind(data.kind))


class _Snapshot(Protocol):
    """The snapshot fields that build and substitution requests share."""

    @property
    def school_days(self) -> tuple[int, ...]: ...
    @property
    def lessons_per_day(self) -> int: ...
    @property
    def staff(self) -> int: ...
    @property
    def capacity(self) -> AlertCapacity | PassportCapacity: ...
    @property
    def classes(self) -> tuple[ClassIn, ...]: ...
    @property
    def subjects(self) -> tuple[SubjectIn, ...]: ...
    @property
    def teachers(self) -> tuple[TeacherIn, ...]: ...
    @property
    def rows(self) -> tuple[RowIn, ...]: ...


def _school(data: _Snapshot, kind: TimetableKind) -> Problem:
    """The snapshot as a ``Problem``; ids are unique and every curriculum row resolves."""
    is_alert_capacity = isinstance(data.capacity, AlertCapacity)
    if is_alert_capacity != (kind is TimetableKind.ALERT):
        msg = "capacity shape does not match kind"
        raise InvalidInputError(msg)

    class_ids = _unique("class", [c.id for c in data.classes])
    subject_ids = _unique("subject", [s.id for s in data.subjects])
    teacher_ids = _unique("teacher", [t.id for t in data.teachers])
    for row in data.rows:
        if (
            row.class_id not in class_ids
            or row.subject_id not in subject_ids
            or row.teacher_id not in teacher_ids
        ):
            msg = "curriculum row references an unknown id"
            raise InvalidInputError(msg)
    # One teacher per class x subject (spec), as validator-lib duplicate_row: no teacher in key.
    _unique("row", [f"{r.class_id}\x00{r.subject_id}" for r in data.rows])
    return Problem(
        kind=kind,
        school_days=data.school_days,
        lessons_per_day=data.lessons_per_day,
        staff=data.staff,
        capacity=_capacity(data.capacity),
        classes=tuple(SchoolClass(c.id, c.students, c.priority) for c in data.classes),
        subjects=tuple(Subject(s.id, s.priority_area, s.no_shelter) for s in data.subjects),
        teachers=tuple(
            Teacher(t.id, t.weekly_limit, frozenset(t.unavailable_days)) for t in data.teachers
        ),
        rows=tuple(
            CurriculumRow(r.class_id, r.subject_id, r.teacher_id, r.hours) for r in data.rows
        ),
    )


def _capacity(capacity: AlertCapacity | PassportCapacity) -> Capacity:
    if isinstance(capacity, AlertCapacity):
        return AlertLimits(seats=capacity.seats, passport=capacity.passport)
    return Passport(passport=capacity.passport)


def to_substitution(data: SubstitutionRequestedData) -> SubstitutionRequest:
    """The request as engine input; ids and dates resolve, the week itself is checked later.

    Every date of the template's week, overlays, absences and non-school dates lies in
    ``iso_week``; ``absence_dates`` may reach beyond it (all dates of one absence).
    """
    school = _school(data, TimetableKind(data.template.kind))
    week = iso_week_dates(data.iso_week)
    classes = {c.id for c in school.classes}
    subjects = {s.id for s in school.subjects}
    teachers = {t.id for t in school.teachers}
    non_school = None if data.non_school_dates is None else frozenset(data.non_school_dates)
    _require(data.absent_teacher_id in teachers, "the absent teacher is not in the snapshot")
    _require(data.date in week, "the date is not in iso_week")
    _require(data.date.isoweekday() in school.school_days, "the date is not a school day")
    _require(non_school is None or data.date not in non_school, "the date is a non-school date")
    _require(data.date in data.absence_dates, "the date is not one of absence_dates")
    _require(non_school is None or non_school <= set(week), "a non-school date outside iso_week")
    _require(all(x.class_id in classes for x in data.lessons), "a lesson of an unknown class")
    for x in data.template.lessons:
        known = x.class_id in classes and x.subject_id in subjects and x.teacher_id in teachers
        _require(known, "a template lesson references an unknown id")
    _require(
        all(f.class_id in classes for f in data.template.day_formats),
        "a day format of an unknown class",
    )
    for a in data.approved_substitutions:
        _require(a.class_id in classes and a.date in week, "an approved substitution outside")
        _require(a.teacher_id is None or a.teacher_id in teachers, "an unknown substitute")
        _require(a.moved_to is None or a.moved_to.date in week, "moved_to outside iso_week")
    for a in data.absences or ():
        _require(a.teacher_id in teachers and a.date in week, "an absence outside the snapshot")
    return SubstitutionRequest(
        school=school,
        date=data.date,
        absent_teacher_id=data.absent_teacher_id,
        lessons=tuple(Slot(data.date, x.class_id, x.lesson_no) for x in data.lessons),
        week=week,
        not_before=data.not_before,
        absence_dates=frozenset(data.absence_dates),
        template_lessons=tuple(
            Lesson(x.class_id, x.day, x.lesson_no, x.subject_id, x.teacher_id)
            for x in data.template.lessons
        ),
        template_formats=tuple(
            DayFormat(f.class_id, f.day, f.in_person) for f in data.template.day_formats
        ),
        approved=tuple(
            Overlay(
                Slot(a.date, a.class_id, a.lesson_no),
                OverlayKind(a.kind),
                a.teacher_id,
                None if a.moved_to is None else Hour(a.moved_to.date, a.moved_to.lesson_no),
            )
            for a in data.approved_substitutions
        ),
        absences=None
        if data.absences is None
        else tuple(
            Absence(
                a.teacher_id,
                a.date,
                None if a.lesson_nos is None else frozenset(a.lesson_nos),
            )
            for a in data.absences
        ),
        non_school_dates=non_school,
        allow_support_moves=data.allow_support_moves,
    )


def _require(condition: bool, what: str) -> None:  # noqa: FBT001 - a guard
    if not condition:
        raise InvalidInputError(what)


_NOT_PART_OF_INPUT: Final = frozenset({"job_id", "queue_deadline_at", "time_limit_s"})


def input_sha256(data: BuildRequestedData | SubstitutionRequestedData) -> str:
    """Hash of the problem itself (not of the job), so bench can match a request to its input."""
    canonical = data.model_dump(mode="json", exclude=set(_NOT_PART_OF_INPUT))
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def params_for(  # noqa: PLR0913 -- keyword-only server-side knobs
    data: BuildRequestedData,
    *,
    min_time_limit_s: int,
    max_time_limit_s: int,
    num_workers: int,
    seed: int,
    max_memory_mb: int | None = None,
    stall_min_s: float = 0.0,
    stall_ratio: float = 1.0,
) -> Params:
    """Server-side bounds: the requested limit is clamped, clients never pick solver knobs."""
    limit = clamp_time_limit(data.time_limit_s, min_time_limit_s, max_time_limit_s)
    return Params(
        time_limit_s=float(limit),
        num_workers=num_workers,
        random_seed=seed,
        max_memory_mb=max_memory_mb,
        stall_min_s=stall_min_s or None,
        stall_ratio=stall_ratio,
    )


def clamp_time_limit(requested: int, min_time_limit_s: int, max_time_limit_s: int) -> int:
    """The only clamp of the time limit in production code (contracts README item 5)."""
    return max(min_time_limit_s, min(max_time_limit_s, requested))


def substitution_params(
    data: SubstitutionRequestedData,
    *,
    min_time_limit_s: int,
    max_time_limit_s: int,
    seed: int,
    max_memory_mb: int | None = None,
) -> SubstitutionParams:
    """One worker and a deterministic budget of ``time_limit_s`` deterministic seconds (ADR)."""
    limit = clamp_time_limit(data.time_limit_s, min_time_limit_s, max_time_limit_s)
    return SubstitutionParams(
        time_limit_s=float(limit), random_seed=seed, max_memory_mb=max_memory_mb
    )


# ---------------------------------------------------------------- outbound


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _encode(document: dict[str, object]) -> bytes:
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()


def _envelope(address: Address, event_type: str, data: dict[str, object], now: datetime) -> bytes:
    return _encode(
        {
            "event_id": str(event_id(address.job_id, event_type)),
            "type": event_type,
            "schema_version": SCHEMA_VERSION,
            "source": SOURCE,
            "time": _iso(now),
            "correlation_id": address.correlation_id,
            "school_id": str(address.school_id),
            "data": data,
        }
    )


def started_message(address: Address, started_at: datetime, topic: str) -> OutboundMessage:
    data: dict[str, object] = {"job_id": str(address.job_id), "started_at": _iso(started_at)}
    return OutboundMessage(topic, address.key, _envelope(address, STARTED, data, started_at))


@dataclass(frozen=True, slots=True)
class Provenance:
    wall_time_ms: int
    input_sha256: str


def completed_data(
    job_id: UUID, result: Result, provenance: Provenance | None
) -> dict[str, object]:
    data: dict[str, object] = {
        "job_id": str(job_id),
        "status": result.status.value,
        "lessons": [
            {
                "class_id": lesson.class_id,
                "day": lesson.day,
                "lesson_no": lesson.lesson_no,
                "subject_id": lesson.subject_id,
                "teacher_id": lesson.teacher_id,
            }
            for lesson in result.lessons
        ],
        "day_formats": [
            {"class_id": f.class_id, "day": f.day, "in_person": f.in_person}
            for f in result.day_formats
        ],
        "conflicts": [
            {
                key: value
                for key, value in {
                    "rule": c.rule.value,
                    "class_id": c.class_id,
                    "teacher_id": c.teacher_id,
                    "subject_id": c.subject_id,
                    "day": c.day,
                    "condition": c.condition.value if c.condition is not None else None,
                    "hint": True if c.hint else None,
                }.items()
                if value is not None
            }
            for c in result.conflicts
        ],
    }
    if provenance is not None:
        stats = result.stats
        data["stats"] = {
            "objective": stats.objective,
            "best_bound": stats.best_bound,
            "primary": stats.primary,
            "secondary": stats.secondary,
            "wall_time_ms": provenance.wall_time_ms,
            "num_workers": stats.num_workers,
            "random_seed": stats.random_seed,
            "solver_version": __version__,
            "model_version": MODEL_VERSION,
            "input_sha256": provenance.input_sha256,
            "stop_reason": stats.stop_reason.value,
            "quality_relaxed": stats.quality_relaxed,
        }
    return data


def _proposal(p: Proposal) -> dict[str, object]:
    row: dict[str, object] = {
        "date": p.slot.date.isoformat(),
        "class_id": p.slot.class_id,
        "lesson_no": p.slot.lesson_no,
        "kind": p.kind.value,
    }
    match p.kind:
        case ProposalKind.TEACHER:
            row |= {"teacher_id": p.teacher_id, "off_profile": p.off_profile}
        case ProposalKind.MOVED:
            assert p.moved_to is not None  # noqa: S101 -- the engine sets it for moved rows
            row["moved_to"] = {
                "date": p.moved_to.date.isoformat(),
                "lesson_no": p.moved_to.lesson_no,
            }
        case ProposalKind.SELF_STUDY:
            pass
    row["step"] = p.step
    return row


def substitution_data(
    job_id: UUID, result: SubstitutionResult, provenance: Provenance | None
) -> dict[str, object]:
    """``solve.completed`` of a substitution: no timetable, no conflicts (contracts 0.5.0)."""
    if len(result.proposals) > MAX_PROPOSALS:
        return rejected_data(job_id, ErrorCode.RESULT_TOO_LARGE)
    data: dict[str, object] = {
        "job_id": str(job_id),
        "status": result.status.value,
        "lessons": [],
        "day_formats": [],
        "conflicts": [],
        "substitution": {
            "step": result.step,
            "proposals": [_proposal(p) for p in result.proposals],
        },
    }
    if provenance is not None:
        stats = result.stats
        data["stats"] = {
            "wall_time_ms": provenance.wall_time_ms,
            "num_workers": stats.num_workers,
            "random_seed": stats.random_seed,
            "solver_version": __version__,
            "model_version": MODEL_VERSION,
            "input_sha256": provenance.input_sha256,
            "stop_reason": stats.stop_reason.value,
        }
    return data


def rejected_data(job_id: UUID, code: ErrorCode) -> dict[str, object]:
    data = completed_data(job_id, Result(status=SolveStatus.MODEL_INVALID), None)
    data["conflicts"] = [{"rule": "invalid_input"}] if code in _INPUT_ERRORS else []
    data["error"] = {"code": code.value}
    return data


@dataclass(frozen=True, slots=True)
class Completed:
    message: OutboundMessage
    status: str
    error_code: ErrorCode | None


def completed_message(
    address: Address,
    data: dict[str, object],
    *,
    now: datetime,
    topic: str,
    max_bytes: int,
) -> Completed:
    """Serialize ``solve.completed``; a result over ``max_bytes`` becomes result_too_large."""
    value = _envelope(address, COMPLETED, data, now)
    status = str(data["status"])
    error = data.get("error")
    code = ErrorCode(error["code"]) if isinstance(error, dict) else None
    if len(value) > max_bytes:
        too_large = rejected_data(address.job_id, ErrorCode.RESULT_TOO_LARGE)
        value = _envelope(address, COMPLETED, too_large, now)
        status, code = SolveStatus.MODEL_INVALID.value, ErrorCode.RESULT_TOO_LARGE
    return Completed(OutboundMessage(topic, address.key, value), status, code)


# ---------------------------------------------------------------- DLQ


def _models_in(annotation: object) -> Iterator[type[BaseModel]]:
    if isinstance(annotation, TypeAliasType):
        yield from _models_in(annotation.__value__)
        return
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
        return
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        yield from _models_in(args[0])
    elif origin in {Union, UnionType, tuple, list} or isinstance(origin, TypeAliasType):
        for arg in args:
            yield from _models_in(arg)


def _redacted(value: object) -> str:
    return f"<redacted:{type(value).__name__}>"


def _is_scalar(item: object) -> bool:
    if isinstance(item, float):
        return math.isfinite(item)  # NaN / Infinity would make the DLQ copy invalid JSON
    return isinstance(item, (bool, int, str))


def _is_plain(item: object) -> bool:
    """A JSON scalar or a flat list of scalars (e.g. unavailable_days)."""
    if item is None or _is_scalar(item):
        return True
    if isinstance(item, list):
        items: list[object] = item  # pyright: ignore[reportUnknownVariableType]
        return all(_is_scalar(x) for x in items)
    return False


_LEAF_CONFIG: Final = ConfigDict(hide_input_in_errors=True)


def _annotated(model: type[BaseModel], name: str) -> object:
    info = model.model_fields[name]
    return Annotated[info.annotation, *info.metadata] if info.metadata else info.annotation


# Free-text envelope fields of inbound events are checked as strictly as the envelope reader does;
# ``source`` has no contract format, so it is kept only when it looks like an identifier.
_ENVELOPE_LEAVES: Final[dict[str, object]] = {
    "schema_version": _annotated(EnvelopeHead, "schema_version"),
    "correlation_id": _annotated(EnvelopeHead, "correlation_id"),
    "source": Id | None,
}


_ENVELOPE_MODELS: Final = frozenset({BuildRequested, SubstitutionHead})

# DLQ projections per consumed topic: a field outside these models never reaches the DLQ copy.
BUILD_DLQ_MODELS: Final[tuple[type[BaseModel], ...]] = (BuildRequested,)
SUBSTITUTION_DLQ_MODELS: Final[tuple[type[BaseModel], ...]] = (SubstitutionHead,)


@cache
def _leaf_adapter(model: type[BaseModel], name: str) -> TypeAdapter[Any]:
    override = _ENVELOPE_LEAVES.get(name) if model in _ENVELOPE_MODELS else None
    annotation: Any = override or _annotated(model, name)  # a type form built at runtime
    return TypeAdapter[Any](annotation, config=_LEAF_CONFIG)


def _leaf(model: type[BaseModel], name: str, item: object) -> object:
    """A contract leaf is copied only when it is a valid value of that field: a field name alone
    does not make its value safe (a signature typed into ``teachers[].id`` stays out)."""
    if not _is_plain(item):
        return _redacted(item)
    try:
        _leaf_adapter(model, name).validate_python(item)
    except ValidationError:
        return _redacted(item)
    return item


MAX_DLQ_DEPTH: Final = 16  # the contract nests at most 4 levels; deeper input is redacted


def _project(value: object, models: list[type[BaseModel]], depth: int = 0) -> object:
    """Keep only fields the contract knows; anything else (e.g. a signature) is dropped.

    Depth is bounded: json.loads accepts ~10^5 levels, far beyond the recursion limit.
    """
    if depth >= MAX_DLQ_DEPTH:
        return "<redacted:too_deep>"
    if isinstance(value, list):
        items: list[object] = value  # pyright: ignore[reportUnknownVariableType]
        return [_project(item, models, depth + 1) for item in items]
    if not isinstance(value, dict):
        return _redacted(value)
    document: dict[str, object] = value  # pyright: ignore[reportUnknownVariableType]
    projected: dict[str, object] = {}
    for model in models:
        for name, info in model.model_fields.items():
            if name not in document or name in projected:
                continue
            item = document[name]
            nested = list(_models_in(info.annotation))
            if nested:
                projected[name] = _project(item, nested, depth + 1)
            else:
                projected[name] = _leaf(model, name, item)
    return projected


MAX_DLQ_PROJECTION_BYTES: Final = 512 * 1024  # the DLQ copy must stay below the 1 MB limit


def _minimal(projected: object) -> dict[str, object]:
    """Envelope scalars and the job id only, for a projection too large to produce."""
    if not isinstance(projected, dict):
        return {}
    document: dict[str, object] = projected  # pyright: ignore[reportUnknownVariableType]
    head: dict[str, object] = {
        key: value for key, value in document.items() if not isinstance(value, (dict, list))
    }
    data = document.get("data")
    if isinstance(data, dict) and "job_id" in data:
        head["data"] = {"job_id": data["job_id"]}
    return head


def _dump(document: dict[str, object]) -> bytes:
    # ASCII escapes: a lone surrogate from the input must not break the diagnostic copy.
    return json.dumps(document, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def dlq_value(raw: bytes | None, models: Sequence[type[BaseModel]]) -> bytes | None:
    """DLQ copy for diagnostics. JSON is projected onto the contract's fields, so personal data
    added by mistake does not live on in the DLQ for 14 days. Bytes that are not JSON (truncated,
    wrong encoding) cannot be projected, so only their size and SHA-256 are kept."""
    if raw is None:
        return None
    try:
        document: object = json.loads(raw, parse_int=_parse_int)
    except ValueError, RecursionError:
        return _dump(
            {"unparseable": True, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    projected = _project(document, list(models))
    value = _dump({"projected": projected})
    if len(value) > MAX_DLQ_PROJECTION_BYTES:
        value = _dump({"projected": _minimal(projected), "truncated": True})
    return value


def dlq_message(
    message: InboundMessage,
    *,
    topic: str,
    error_class: str,
    models: Sequence[type[BaseModel]],
) -> OutboundMessage:
    headers = (
        ("x-error-class", error_class.encode()),
        ("x-original-topic", message.topic.encode()),
        ("x-original-partition", str(message.partition).encode()),
        ("x-original-offset", str(message.offset).encode()),
    )
    return OutboundMessage(topic, message.key, dlq_value(message.value, models), headers)
