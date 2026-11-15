"""Substitution job: the request, the week it changes and the proposals (spec §06, stage 2).

ADR 2026-11-15-solver-substitution-engine. Pure value types and the week view the model reads, no
I/O and no clock (``not_before`` comes with the request). Identifiers are opaque strings and an
absence carries no reason (personal-data rules).

The baseline week is the template in force on the job's date laid on the dates of its ISO week,
with the approved substitutions applied in order, the way timetable builds the week it validates
(contracts ``vectors/substitution``): non-school dates hold no lessons; ``teacher`` gives the slot
another teacher, ``moved`` moves the lesson to another hour of the week, ``self_study`` keeps the
class in its slot without a teacher, ``cancelled`` frees the slot.
"""

import datetime as dt
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final

from lad_solver.domain.problem import Problem
from lad_solver.domain.result import DayFormat, Lesson, SolveStatus, StopReason

STEPS: Final = 3  # 1 free teacher, 2 rebuild of the date, 3 neighbour dates


class InvalidRequestError(ValueError):
    """The request does not describe one consistent week. The message never quotes input values."""


class OverlayKind(StrEnum):
    """Kinds of an approved substitution (overlay) of the baseline week."""

    TEACHER = "teacher"
    MOVED = "moved"
    SELF_STUDY = "self_study"
    CANCELLED = "cancelled"  # chosen only by a person, never proposed


class ProposalKind(StrEnum):
    TEACHER = "teacher"  # another teacher in the same slot
    MOVED = "moved"  # the same lesson at another hour of the week
    SELF_STUDY = "self_study"  # no teacher; the class keeps its slot and its day format


@dataclass(frozen=True, slots=True, order=True)
class Hour:
    """A lesson hour of the week."""

    date: dt.date
    lesson_no: int


@dataclass(frozen=True, slots=True, order=True)
class Slot:
    """A class at a lesson hour: where a lesson of the week is held."""

    date: dt.date
    class_id: str
    lesson_no: int

    @property
    def hour(self) -> Hour:
        return Hour(self.date, self.lesson_no)

    def at(self, hour: Hour) -> Slot:
        return Slot(hour.date, self.class_id, hour.lesson_no)


@dataclass(frozen=True, slots=True)
class Overlay:
    """An approved substitution of one slot (``teacher_id`` / ``moved_to`` as its kind needs)."""

    slot: Slot
    kind: OverlayKind
    teacher_id: str | None = None
    moved_to: Hour | None = None

    def __post_init__(self) -> None:
        if (self.teacher_id is not None) != (self.kind is OverlayKind.TEACHER):
            msg = "teacher_id is set exactly for kind=teacher"
            raise ValueError(msg)
        if (self.moved_to is not None) != (self.kind is OverlayKind.MOVED):
            msg = "moved_to is set exactly for kind=moved"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Absence:
    """A teacher absent on some lessons of a date (``None``: the whole date). No reason, ever."""

    teacher_id: str
    date: dt.date
    lesson_nos: frozenset[int] | None = None


@dataclass(frozen=True, slots=True)
class SubstitutionRequest:
    """One substitution job: one date of an absence (timetable sends a job per date)."""

    school: Problem  # the snapshot; its kind is the template's (main 1.1, alert 1.2)
    date: dt.date
    absent_teacher_id: str
    lessons: tuple[Slot, ...]  # the absent teacher's lessons on ``date`` to cover, request order
    week: tuple[dt.date, ...]  # the seven dates of the job's ISO week, Monday first
    not_before: dt.date  # step 3 moves only to dates >= not_before ("today" in Europe/Kyiv)
    absence_dates: frozenset[dt.date]
    template_lessons: tuple[Lesson, ...]
    template_formats: tuple[DayFormat, ...]
    approved: tuple[Overlay, ...] = ()
    absences: tuple[Absence, ...] | None = None  # None: the absent teacher, absence_dates, all day
    non_school_dates: frozenset[dt.date] | None = None  # None: step 3 is off
    # true (after the defence) would allow chains of other teachers' lessons; the engine does not
    # implement them yet and answers as with false (conditional deviation, contracts README item 9).
    allow_support_moves: bool = False

    @property
    def step3(self) -> bool:
        """Step 3 runs only when timetable sends the non-school dates (an empty list is fine)."""
        return self.non_school_dates is not None

    @property
    def last_step(self) -> int:
        """The last step a full run reaches: the ``step`` of a self-study proposal."""
        return 3 if self.step3 else 2

    def window(self) -> tuple[dt.date, ...]:
        """Step 3 dates in the order they are tried: nearest first, on a tie the later date.

        Dates of the ISO week that are school days, >= ``not_before``, not non-school dates and
        not the job's date (the later date on a tie gives more time to warn parents; Roman,
        15.11.2026). Empty when step 3 is off.
        """
        if self.non_school_dates is None:
            return ()
        held = [
            x
            for x in self.week
            if x.isoweekday() in self.school.school_days
            and x >= self.not_before
            and x not in self.non_school_dates
            and x != self.date
        ]
        return tuple(sorted(held, key=lambda x: (abs((x - self.date).days), -x.toordinal())))


@dataclass(frozen=True, slots=True)
class Proposal:
    """One proposed change of one lesson slot of the baseline week (contracts 0.5.0)."""

    slot: Slot
    kind: ProposalKind
    step: int
    teacher_id: str | None = None  # kind=teacher
    off_profile: bool | None = (
        None  # kind=teacher: no curriculum row of the subject ("не за фахом")
    )
    moved_to: Hour | None = None  # kind=moved

    def as_overlay(self) -> Overlay:
        """The proposal as an overlay of the week (self-study is never applied mid-run)."""
        kind = OverlayKind(self.kind.value)
        return Overlay(self.slot, kind, self.teacher_id, self.moved_to)


@dataclass(frozen=True, slots=True)
class SubstitutionParams:
    """Engine knobs; the worker clamps ``time_limit_s`` (clients never pick solver knobs)."""

    time_limit_s: float  # wall-clock safety net of the whole job
    random_seed: int = 0
    max_memory_mb: int | None = None
    max_deterministic_time: float | None = None  # the step budgets; None: time_limit_s
    last_step: int = STEPS  # stop after this step (tests and bench); the worker runs all

    def __post_init__(self) -> None:
        for name, value in (
            ("time_limit_s", self.time_limit_s),
            ("max_deterministic_time", self.max_deterministic_time),
        ):
            if value is not None and not (math.isfinite(value) and value > 0):
                msg = f"{name} must be a finite number > 0, got {value!r}"
                raise ValueError(msg)
        if not 1 <= self.last_step <= STEPS:
            msg = f"last_step must be within [1, {STEPS}], got {self.last_step!r}"
            raise ValueError(msg)

    @property
    def deterministic_time(self) -> float:
        return (
            self.time_limit_s
            if self.max_deterministic_time is None
            else self.max_deterministic_time
        )


@dataclass(frozen=True, slots=True)
class SubstitutionStats:
    """Provenance and telemetry (counts and enums only: they go to the worker logs)."""

    wall_time_s: float = 0.0
    deterministic_time: float = 0.0  # CP-SAT deterministic seconds of all runs
    stop_reason: StopReason = StopReason.COMPLETED
    num_workers: int = 1
    random_seed: int = 0
    runs: int = 0  # CP-SAT runs
    anomalies: int = 0  # runs that ended INFEASIBLE / MODEL_INVALID: an engine bug, never input
    covered: tuple[int, ...] = (0, 0, 0)  # lessons covered by steps 1, 2, 3


@dataclass(frozen=True, slots=True)
class SubstitutionResult:
    """Always a solution (OPTIMAL or FEASIBLE): every lesson to cover has exactly one proposal."""

    status: SolveStatus
    proposals: tuple[Proposal, ...]
    stats: SubstitutionStats

    def __post_init__(self) -> None:
        if self.status not in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE} or not self.proposals:
            msg = "a substitution result is OPTIMAL or FEASIBLE and has proposals"
            raise ValueError(msg)

    @property
    def step(self) -> int:
        """The last step the job reached: max(proposals[].step)."""
        return max(p.step for p in self.proposals)


_ISO_WEEK: Final = re.compile(r"(\d{4})-W(\d{2})")


def iso_week_dates(text: str) -> tuple[dt.date, ...]:
    """The seven dates of ``2026-W47`` (Monday first); a week the year does not have is invalid."""
    match = _ISO_WEEK.fullmatch(text)
    if match is None:
        msg = "iso_week is not YYYY-Www"
        raise InvalidRequestError(msg)
    year, week = int(match[1]), int(match[2])
    try:
        return tuple(dt.date.fromisocalendar(year, week, day) for day in range(1, 8))
    except ValueError:
        msg = "iso_week does not exist"
        raise InvalidRequestError(msg) from None


# ---------------------------------------------------------------- the week


@dataclass(frozen=True, slots=True)
class WeekLesson:
    slot: Slot  # where it is held this week
    subject_id: str
    teacher_id: str
    substitute: bool = False  # taught by a substitute (validator-lib origin=substitution)


type Formats = Mapping[tuple[str, int], bool]  # (class_id, weekday) -> in person


class Week:
    """Lessons of one ISO week, indexed for the model. Immutable: overlays give a new week."""

    __slots__ = (
        "_absent",
        "_busy",
        "_formats",
        "_load",
        "_occupied",
        "_teacher_days",
        "lessons",
        "request",
        "self_study",
    )

    def __init__(
        self,
        request: SubstitutionRequest,
        lessons: Mapping[Slot, WeekLesson],
        self_study: frozenset[Slot],
        *,
        formats: Formats,
        absent: Mapping[tuple[str, dt.date], frozenset[int] | None],
    ) -> None:
        self.request = request
        self.lessons: Mapping[Slot, WeekLesson] = dict(lessons)
        self.self_study = self_study
        self._formats = formats
        self._absent = absent
        self._busy: set[tuple[str, Hour]] = set()
        self._load: Counter[str] = Counter()
        self._occupied: Counter[tuple[str, dt.date]] = Counter()
        self._teacher_days: defaultdict[tuple[str, dt.date], set[str]] = defaultdict(set)
        for slot, lesson in self.lessons.items():
            self._busy.add((lesson.teacher_id, slot.hour))
            self._load[lesson.teacher_id] += 1
            self._occupied[slot.class_id, slot.date] += 1
            self._teacher_days[lesson.teacher_id, slot.date].add(slot.class_id)
        for slot in self_study:
            self._occupied[slot.class_id, slot.date] += 1

    # ------------------------------------------------------------ queries

    def in_person(self, class_id: str, date: dt.date) -> bool | None:
        """The fixed day format of a class (``None``: the template has no format for that day)."""
        return self._formats.get((class_id, date.isoweekday()))

    def absent(self, teacher_id: str, hour: Hour) -> bool:
        if (teacher_id, hour.date) not in self._absent:
            return False
        lesson_nos = self._absent[teacher_id, hour.date]
        return lesson_nos is None or hour.lesson_no in lesson_nos

    def busy(self, teacher_id: str, hour: Hour) -> bool:
        return (teacher_id, hour) in self._busy

    def class_busy(self, slot: Slot) -> bool:
        """A lesson or a self-study holds the slot (a cancelled lesson frees it)."""
        return slot in self.lessons or slot in self.self_study

    def teaches_on(self, teacher_id: str, date: dt.date) -> bool:
        return (teacher_id, date) in self._teacher_days

    def teacher_formats(self, teacher_id: str, date: dt.date) -> frozenset[bool]:
        """Formats of the classes the teacher teaches that date (two values: mixed, stale)."""
        classes = self._teacher_days.get((teacher_id, date), set[str]())
        return frozenset(f for c in classes if (f := self.in_person(c, date)) is not None)

    def on_site(self, teacher_id: str, date: dt.date) -> bool:
        """At least one lesson of an in-person class that date (spec §04)."""
        return True in self.teacher_formats(teacher_id, date)

    def load(self, teacher_id: str) -> int:
        """Lessons the teacher teaches this week (validator-lib's weekly limit count)."""
        return self._load[teacher_id]

    def occupied(self, class_id: str, date: dt.date) -> int:
        """Slots of the class that date held by a lesson or a self-study."""
        return self._occupied[class_id, date]

    def pupils(self, date: dt.date) -> int:
        """Pupils of the classes in person that date (fixed formats: the 1.2 seats load)."""
        return sum(c.students for c in self.request.school.classes if self.in_person(c.id, date))

    def teachers_on_site(self, date: dt.date) -> int:
        teachers = {t for (t, x) in self._teacher_days if x == date}
        return sum(1 for t in teachers if self.on_site(t, date))

    # ------------------------------------------------------------ changes

    def with_overlays(self, overlays: Iterable[Overlay]) -> Week:
        lessons = dict(self.lessons)
        self_study = set(self.self_study)
        for overlay in overlays:
            _apply(lessons, self_study, overlay, self.request.week)
        return Week(
            self.request,
            lessons,
            frozenset(self_study),
            formats=self._formats,
            absent=self._absent,
        )


def baseline_week(request: SubstitutionRequest) -> Week:
    """The week without this job's proposals; raises ``InvalidRequestError`` if inconsistent."""
    if request.absent_teacher_id not in {t.id for t in request.school.teachers}:
        msg = "the absent teacher is not in the snapshot"
        raise InvalidRequestError(msg)
    if request.date not in request.week:
        msg = "the date is not in the ISO week"
        raise InvalidRequestError(msg)
    formats: dict[tuple[str, int], bool] = {}
    for f in request.template_formats:
        if (f.class_id, f.day) in formats:
            msg = "two day formats for one class and day"
            raise InvalidRequestError(msg)
        formats[f.class_id, f.day] = f.in_person
    date_of = {x.isoweekday(): x for x in request.week}
    holidays = request.non_school_dates or frozenset[dt.date]()
    lessons: dict[Slot, WeekLesson] = {}
    for lesson in request.template_lessons:
        date = date_of[lesson.day]
        if date in holidays:
            continue  # not held: no lesson on a non-school date
        slot = Slot(date, lesson.class_id, lesson.lesson_no)
        if slot in lessons:
            msg = "two template lessons in one class slot"
            raise InvalidRequestError(msg)
        lessons[slot] = WeekLesson(slot, lesson.subject_id, lesson.teacher_id)
    week = Week(
        request, lessons, frozenset(), formats=formats, absent=_absences(request)
    ).with_overlays(_approved(request))
    _check_lessons_to_cover(request, week)
    return week


def _approved(request: SubstitutionRequest) -> tuple[Overlay, ...]:
    seen: set[Slot] = set()
    for overlay in request.approved:
        if overlay.slot in seen:
            msg = "two approved substitutions on one slot"
            raise InvalidRequestError(msg)
        seen.add(overlay.slot)
    return request.approved


def _apply(
    lessons: dict[Slot, WeekLesson],
    self_study: set[Slot],
    overlay: Overlay,
    week: tuple[dt.date, ...],
) -> None:
    lesson = lessons.get(overlay.slot)
    if lesson is None:
        msg = "an overlay on a slot without a lesson"
        raise InvalidRequestError(msg)
    match overlay.kind:
        case OverlayKind.TEACHER:
            assert overlay.teacher_id is not None  # noqa: S101 -- Overlay checks its shape
            lessons[overlay.slot] = replace(lesson, teacher_id=overlay.teacher_id, substitute=True)
        case OverlayKind.MOVED:
            assert overlay.moved_to is not None  # noqa: S101 -- Overlay checks its shape
            target = overlay.slot.at(overlay.moved_to)
            if target.date not in week:
                msg = "a lesson moved outside the ISO week"
                raise InvalidRequestError(msg)
            if target in lessons or target in self_study:
                msg = "a lesson moved into an occupied class slot"
                raise InvalidRequestError(msg)
            del lessons[overlay.slot]
            lessons[target] = replace(lesson, slot=target)
        case OverlayKind.SELF_STUDY:
            del lessons[overlay.slot]
            self_study.add(overlay.slot)
        case OverlayKind.CANCELLED:
            del lessons[overlay.slot]


def _absences(request: SubstitutionRequest) -> dict[tuple[str, dt.date], frozenset[int] | None]:
    """(teacher, date) -> absent lesson numbers (``None``: the whole date).

    Without ``absences`` only the absent teacher is absent, the whole of every absence date. The
    lessons to cover are his absences by definition, whatever ``absences`` says about them.
    """
    rows = request.absences
    if rows is None:
        rows = tuple(Absence(request.absent_teacher_id, x) for x in sorted(request.absence_dates))
    absent: dict[tuple[str, dt.date], frozenset[int] | None] = {}
    for row in rows:
        key = (row.teacher_id, row.date)
        if key in absent:
            msg = "two absences of one teacher on one date"
            raise InvalidRequestError(msg)
        absent[key] = row.lesson_nos
    own = (request.absent_teacher_id, request.date)
    needed = frozenset(s.lesson_no for s in request.lessons)
    if own not in absent:
        absent[own] = needed
    elif (lesson_nos := absent[own]) is not None:
        absent[own] = lesson_nos | needed
    return absent


def _check_lessons_to_cover(request: SubstitutionRequest, week: Week) -> None:
    approved = {o.slot for o in request.approved}
    if len(set(request.lessons)) != len(request.lessons):
        msg = "a lesson to cover is listed twice"
        raise InvalidRequestError(msg)
    for slot in request.lessons:
        lesson = week.lessons.get(slot)
        if slot.date != request.date or lesson is None:
            msg = "a lesson to cover is not a lesson of the date"
            raise InvalidRequestError(msg)
        if lesson.teacher_id != request.absent_teacher_id:
            msg = "a lesson to cover is not the absent teacher's"
            raise InvalidRequestError(msg)
        if slot in approved:  # one approved overlay per slot (partial unique index in timetable)
            msg = "a lesson to cover already has an approved substitution"
            raise InvalidRequestError(msg)
