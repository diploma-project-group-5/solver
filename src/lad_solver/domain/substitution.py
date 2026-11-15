"""Substitution job: the request, the week it changes and the proposals (spec §06, stage 2).

ADR 2026-11-15-solver-substitution-engine. Pure value types and the week view the model reads, no
I/O and no clock (``not_before`` comes with the request). Identifiers are opaque strings and an
absence carries no reason (personal-data rules).

The baseline week is the template in force on the job's date laid on the dates of its ISO week,
with the approved substitutions applied, the way timetable builds the week it validates (contracts
``vectors/substitution``): non-school dates hold no lessons; ``teacher`` gives the slot another
teacher, ``moved`` moves the lesson to another hour of the week, ``self_study`` keeps the class in
its slot without a teacher, ``cancelled`` frees the slot. The result does not depend on the order
of the approved substitutions.

Like validator-lib, the week keeps lessons whose ids left the snapshot (an archived class, a
deleted teacher: timetable projects only active ids, the published template keeps them): they stay
where they are, a lesson of an unknown class counts for nothing, an unknown teacher still counts on
site, and neither is ever a candidate.

``baseline_week`` is the one gate: whatever it accepts is one consistent week the engine can answer.
"""

import datetime as dt
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal

from lad_solver.domain.problem import Problem
from lad_solver.domain.result import DayFormat, Lesson, SolveStatus, StopReason

STEPS: Final = 3  # 1 free teacher, 2 rebuild of the date, 3 neighbour dates


class InvalidRequestError(ValueError):
    """The request does not describe one consistent week. The message is a fixed literal that
    names the broken rule and never quotes input values (it is logged as the rejection reason)."""


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


_OVERLAY_OF: Final = {
    ProposalKind.TEACHER: OverlayKind.TEACHER,
    ProposalKind.MOVED: OverlayKind.MOVED,
    ProposalKind.SELF_STUDY: OverlayKind.SELF_STUDY,
}


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
            raise InvalidRequestError(msg)
        if (self.moved_to is not None) != (self.kind is OverlayKind.MOVED):
            msg = "moved_to is set exactly for kind=moved"
            raise InvalidRequestError(msg)


@dataclass(frozen=True, slots=True)
class Absence:
    """A teacher absent on some lessons of a date (``None``: the whole date). No reason, ever."""

    teacher_id: str
    date: dt.date
    lesson_nos: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.lesson_nos is not None and not self.lesson_nos:
            msg = "an absence lists at least one lesson or the whole date"
            raise InvalidRequestError(msg)


@dataclass(frozen=True, slots=True)
class SubstitutionRequest:
    """One substitution job: one date of an absence (timetable sends a job per date)."""

    school: Problem  # the snapshot (active ids only); its kind is the template's
    date: dt.date
    absent_teacher_id: str
    lessons: tuple[Slot, ...]  # the absent teacher's lessons on ``date`` to cover, request order
    not_before: dt.date  # step 3 moves only to dates >= not_before ("today" in Europe/Kyiv)
    absence_dates: frozenset[dt.date]  # all dates of this absence (may reach beyond the week)
    template_lessons: tuple[Lesson, ...]
    template_formats: tuple[DayFormat, ...]
    approved: tuple[Overlay, ...] = ()
    absences: tuple[Absence, ...] | None = None  # None: the absent teacher, absence_dates, all day
    non_school_dates: frozenset[dt.date] | None = None  # None: step 3 is off
    # true (after the defence) would allow chains of other teachers' lessons; the engine does not
    # implement them yet and answers as with false (conditional deviation, contracts README item 9).
    allow_support_moves: bool = False

    @property
    def week(self) -> tuple[dt.date, ...]:
        """The seven dates of the ISO week of ``date``, Monday first (the job's ``iso_week``)."""
        monday = self.date - dt.timedelta(days=self.date.isoweekday() - 1)
        return tuple(monday + dt.timedelta(days=i) for i in range(7))

    @property
    def step3(self) -> bool:
        """Step 3 runs only when timetable sends the non-school dates (an empty list is fine)."""
        return self.non_school_dates is not None

    @property
    def self_study_step(self) -> int:
        """The step of a self-study row after a full run: the last step there is (3 or 2)."""
        return 3 if self.step3 else 2

    def window(self) -> tuple[dt.date, ...]:
        """Step 3 dates in the order they are added: nearest first, on a tie the later date.

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

    def __post_init__(self) -> None:
        teacher = self.kind is ProposalKind.TEACHER
        if (self.teacher_id is not None) != teacher or (self.off_profile is not None) != teacher:
            msg = "teacher_id and off_profile are set exactly for kind=teacher"
            raise ValueError(msg)
        if (self.moved_to is not None) != (self.kind is ProposalKind.MOVED):
            msg = "moved_to is set exactly for kind=moved"
            raise ValueError(msg)
        if not 1 <= self.step <= STEPS or (self.kind is ProposalKind.MOVED and self.step < 2):  # noqa: PLR2004
            msg = "step is 1..3 and a moved row comes from step 2 or 3"
            raise ValueError(msg)
        if self.moved_to is not None and self.moved_to == self.slot.hour:
            msg = "a moved lesson goes to another hour"
            raise ValueError(msg)

    def as_overlay(self) -> Overlay:
        """The proposal as an overlay of the week (self-study is never applied mid-run)."""
        return Overlay(self.slot, _OVERLAY_OF[self.kind], self.teacher_id, self.moved_to)


@dataclass(frozen=True, slots=True)
class SubstitutionParams:
    """Engine knobs; the worker clamps ``time_limit_s`` (clients never pick solver knobs)."""

    time_limit_s: float  # wall-clock safety net of the whole job
    random_seed: int = 0  # the ADR fixes 0 for the worker
    max_memory_mb: int | None = None
    max_deterministic_time: float | None = None  # the step budgets; None: time_limit_s
    stop_after_step: int = STEPS  # tests and bench; the worker runs every step

    def __post_init__(self) -> None:
        for name, value in (
            ("time_limit_s", self.time_limit_s),
            ("max_deterministic_time", self.max_deterministic_time),
        ):
            if value is not None and not (math.isfinite(value) and value > 0):
                msg = f"{name} must be a finite number > 0, got {value!r}"
                raise ValueError(msg)
        if not 1 <= self.stop_after_step <= STEPS:
            msg = f"stop_after_step must be within [1, {STEPS}], got {self.stop_after_step!r}"
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
    # (step, status) of runs that ended INFEASIBLE / MODEL_INVALID: an engine bug, never input
    anomalies: tuple[tuple[int, SolveStatus], ...] = ()
    covered: tuple[int, int, int] = (0, 0, 0)  # lessons covered by steps 1, 2, 3
    absences_completed: int = 0  # absence dates of the absent teacher read as the whole date
    overlays_ignored: int = 0  # approved substitutions on non-school dates (nothing to change)


@dataclass(frozen=True, slots=True)
class SubstitutionResult:
    """Always a solution: every lesson to cover has exactly one proposal."""

    status: Literal[SolveStatus.OPTIMAL, SolveStatus.FEASIBLE]
    proposals: tuple[Proposal, ...]
    stats: SubstitutionStats

    def __post_init__(self) -> None:
        if self.status not in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE} or not self.proposals:
            msg = "a substitution result is OPTIMAL or FEASIBLE and has proposals"
            raise ValueError(msg)
        if len({p.slot for p in self.proposals}) != len(self.proposals):
            msg = "one proposal per lesson"
            raise ValueError(msg)
        proven = self.stats.stop_reason is StopReason.COMPLETED and not self.stats.anomalies
        if self.status is SolveStatus.OPTIMAL and not proven:
            msg = "OPTIMAL is a completed run without anomalies"
            raise ValueError(msg)

    @property
    def step(self) -> int:
        """The last step the job reached: max(proposals[].step)."""
        return max(p.step for p in self.proposals)


def iso_week_dates(text: str) -> tuple[dt.date, ...]:
    """The seven dates of ``2026-W47`` (Monday first); a week the year does not have is invalid."""
    year, _, week = text.partition("-W")
    try:
        return tuple(dt.date.fromisocalendar(int(year), int(week), day) for day in range(1, 8))
    except ValueError:
        msg = "iso_week does not exist"
        raise InvalidRequestError(msg) from None


# ---------------------------------------------------------------- the week


@dataclass(frozen=True, slots=True)
class WeekLesson:
    slot: Slot  # where it is held this week
    subject_id: str
    teacher_id: str


type Formats = Mapping[tuple[str, int], bool]  # (class_id, weekday) -> in person
type Absent = Mapping[tuple[str, dt.date], frozenset[int] | None]  # None: the whole date


class Week:
    """Lessons of one ISO week, indexed for the model. Immutable: overlays give a new week."""

    __slots__ = (
        "_absent",
        "_busy",
        "_formats",
        "_held",
        "_load",
        "_occupied",
        "_teacher_days",
        "absences_completed",
        "cancelled",
        "lessons",
        "overlays_ignored",
        "request",
        "self_study",
    )

    def __init__(  # noqa: PLR0913 - keyword-only views shared by the weeks of one job
        self,
        request: SubstitutionRequest,
        lessons: Mapping[Slot, WeekLesson],
        self_study: frozenset[Slot],
        cancelled: frozenset[Slot],
        *,
        formats: Formats,
        absent: Absent,
        overlays_ignored: int = 0,
        absences_completed: int = 0,
    ) -> None:
        self.request: Final = request
        self.lessons: Final[Mapping[Slot, WeekLesson]] = MappingProxyType(dict(lessons))
        self.self_study: Final = self_study
        self.cancelled: Final = cancelled
        self.overlays_ignored: Final = overlays_ignored  # telemetry of the baseline week
        self.absences_completed: Final = absences_completed  # ditto
        self._formats: Final = formats
        self._absent: Final = absent
        known = {c.id for c in request.school.classes}
        busy: set[tuple[str, Hour]] = set()
        load: Counter[str] = Counter()
        occupied: Counter[tuple[str, dt.date]] = Counter()
        teacher_days: defaultdict[tuple[str, dt.date], set[str]] = defaultdict(set)
        for slot, lesson in self.lessons.items():
            busy.add((lesson.teacher_id, slot.hour))  # also a lesson of an unknown class
            occupied[slot.class_id, slot.date] += 1
            if slot.class_id in known:  # validator-lib: such a lesson counts for nothing else
                load[lesson.teacher_id] += 1
                teacher_days[lesson.teacher_id, slot.date].add(slot.class_id)
        for slot in self_study:
            occupied[slot.class_id, slot.date] += 1
        held = Counter(occupied)
        for slot in cancelled:
            held[slot.class_id, slot.date] += 1
        self._busy: Final = frozenset(busy)
        self._load: Final = load
        self._occupied: Final = occupied
        self._held: Final = held
        self._teacher_days: Final = {k: frozenset(v) for k, v in teacher_days.items()}

    # ------------------------------------------------------------ queries

    def in_person(self, class_id: str, date: dt.date) -> bool | None:
        """The fixed day format of a class (``None``: no format that day, or an unknown class)."""
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
        """A lesson of a class of the snapshot that date."""
        return (teacher_id, date) in self._teacher_days

    def teacher_formats(self, teacher_id: str, date: dt.date) -> frozenset[bool]:
        """Formats of the classes the teacher teaches that date (two values: mixed, stale)."""
        classes = self._teacher_days.get((teacher_id, date), frozenset[str]())
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

    def holds(self, class_id: str, date: dt.date) -> int:
        """Slots that keep the class's day held: lessons, self-study and cancelled lessons."""
        return self._held[class_id, date]

    def pupils(self, date: dt.date) -> int:
        """Pupils of the classes in person that date (fixed formats: the 1.2 seats load)."""
        return sum(c.students for c in self.request.school.classes if self.in_person(c.id, date))

    def teachers_on_site(self, date: dt.date) -> int:
        """Teachers with a lesson of an in-person class that date, an unknown teacher included."""
        return len(
            {
                lesson.teacher_id
                for slot, lesson in self.lessons.items()
                if slot.date == date and self.in_person(slot.class_id, date)
            }
        )

    # ------------------------------------------------------------ changes

    def with_overlays(self, overlays: Iterable[Overlay]) -> Week:
        parts = _Parts(dict(self.lessons), set(self.self_study), set(self.cancelled))
        _apply_all(parts, list(overlays))
        return Week(
            self.request,
            parts.lessons,
            frozenset(parts.self_study),
            frozenset(parts.cancelled),
            formats=self._formats,
            absent=self._absent,
            overlays_ignored=self.overlays_ignored,
            absences_completed=self.absences_completed,
        )


def baseline_week(request: SubstitutionRequest) -> Week:
    """The week without this job's proposals; raises ``InvalidRequestError`` if inconsistent.

    The one gate of a request: it checks every rule that makes the week and the job coherent.
    """
    _check_request(request)
    known = {c.id for c in request.school.classes}
    formats: dict[tuple[str, int], bool] = {}
    for f in request.template_formats:
        if (f.class_id, f.day) in formats:
            msg = "two day formats for one class and day"
            raise InvalidRequestError(msg)
        formats[f.class_id, f.day] = f.in_person
    formats = {k: v for k, v in formats.items() if k[0] in known}  # validator-lib: no rule
    date_of = {x.isoweekday(): x for x in request.week}
    holidays = request.non_school_dates or frozenset[dt.date]()
    parts = _Parts({})
    for lesson in request.template_lessons:
        slot = Slot(date_of[lesson.day], lesson.class_id, lesson.lesson_no)
        if slot in parts.lessons or slot in parts.not_held:
            msg = "two template lessons in one class slot"
            raise InvalidRequestError(msg)
        held = parts.not_held if slot.date in holidays else parts.lessons
        held[slot] = WeekLesson(slot, lesson.subject_id, lesson.teacher_id)
    absent, completed = _absences(request)
    ignored = _apply_all(parts, request.approved)
    week = Week(
        request,
        parts.lessons,
        frozenset(parts.self_study),
        frozenset(parts.cancelled),
        formats=MappingProxyType(formats),
        absent=MappingProxyType(absent),
        overlays_ignored=ignored,
        absences_completed=completed,
    )
    _check_lessons_to_cover(request, week)
    return week


def _check_request(request: SubstitutionRequest) -> None:
    school = request.school
    teachers = {t.id for t in school.teachers}
    week = set(request.week)
    checks = (
        (request.absent_teacher_id in teachers, "the absent teacher is not in the snapshot"),
        (bool(request.lessons), "there is no lesson to cover"),
        (request.date.isoweekday() in school.school_days, "the date is not a school day"),
        (request.date in request.absence_dates, "the date is not one of absence_dates"),
        (
            request.non_school_dates is None or request.date not in request.non_school_dates,
            "the date is a non-school date",
        ),
        (
            request.non_school_dates is None or request.non_school_dates <= week,
            "a non-school date outside iso_week",
        ),
        (
            all(o.slot.date in week for o in request.approved),
            "an approved substitution outside iso_week",
        ),
        (
            all(o.moved_to is None or o.moved_to.date in week for o in request.approved),
            "a lesson moved outside iso_week",
        ),
        (
            all(a.teacher_id in teachers and a.date in week for a in request.absences or ()),
            "an absence outside the snapshot or iso_week",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise InvalidRequestError(msg)


@dataclass(slots=True)
class _Parts:
    """The parts of a week under construction, changed in place by the overlays."""

    lessons: dict[Slot, WeekLesson]
    self_study: set[Slot] = field(default_factory=set[Slot])
    cancelled: set[Slot] = field(default_factory=set[Slot])
    not_held: dict[Slot, WeekLesson] = field(default_factory=dict[Slot, WeekLesson])


def _apply_all(parts: _Parts, overlays: Sequence[Overlay]) -> int:
    """Apply overlays whatever their order; returns how many changed nothing.

    In rounds: an overlay acts on the lesson its slot holds (a template lesson, or one that
    arrived by a move of an earlier round); every move of a round is lifted before any is placed,
    so moves may swap slots. One on a non-school date changes nothing, unless it moves the lesson
    to another date (it was moved away before the date stopped being a school day).
    """
    if len({o.slot for o in overlays}) != len(overlays):
        msg = "two approved substitutions on one slot"
        raise InvalidRequestError(msg)
    ignored = 0
    pending = list(overlays)
    while pending:
        waiting, lifted, skipped = _round(parts, pending)
        ignored += skipped
        for lesson, target in lifted:
            if target in parts.lessons or target in parts.self_study:
                msg = "a lesson moved into an occupied class slot"
                raise InvalidRequestError(msg)
            parts.lessons[target] = replace(lesson, slot=target)
        if len(waiting) == len(pending):
            msg = "an overlay on a slot without a lesson"
            raise InvalidRequestError(msg)
        pending = waiting
    return ignored


def _round(
    parts: _Parts, pending: list[Overlay]
) -> tuple[list[Overlay], list[tuple[WeekLesson, Slot]], int]:
    """(overlays still waiting for their lesson, lessons lifted to move, overlays ignored)."""
    waiting: list[Overlay] = []
    lifted: list[tuple[WeekLesson, Slot]] = []
    ignored = 0
    for overlay in pending:
        if overlay.slot in parts.not_held:
            if overlay.moved_to is None:
                ignored += 1
            else:
                lesson = parts.not_held.pop(overlay.slot)
                lifted.append((lesson, overlay.slot.at(overlay.moved_to)))
        elif overlay.slot in parts.lessons:
            lifted.extend(_apply(parts, overlay))
        else:
            waiting.append(overlay)  # its lesson may arrive by a move of this round
    return waiting, lifted, ignored


def _apply(parts: _Parts, overlay: Overlay) -> list[tuple[WeekLesson, Slot]]:
    """Apply one overlay to the lesson of its slot; a move is only lifted (placed by the round)."""
    slot = overlay.slot
    lesson = parts.lessons[slot]
    match overlay.kind:
        case OverlayKind.TEACHER:
            assert overlay.teacher_id is not None  # noqa: S101 -- Overlay checks its shape
            parts.lessons[slot] = replace(lesson, teacher_id=overlay.teacher_id)
        case OverlayKind.MOVED:
            assert overlay.moved_to is not None  # noqa: S101 -- Overlay checks its shape
            del parts.lessons[slot]
            return [(lesson, slot.at(overlay.moved_to))]
        case OverlayKind.SELF_STUDY:
            del parts.lessons[slot]
            parts.self_study.add(slot)
        case OverlayKind.CANCELLED:
            del parts.lessons[slot]
            parts.cancelled.add(slot)
    return []


def _absences(
    request: SubstitutionRequest,
) -> tuple[dict[tuple[str, dt.date], frozenset[int] | None], int]:
    """(teacher, date) -> absent lesson numbers (``None``: the whole date), and how many dates of
    the absent teacher the safe reading completed.

    The safe reading never lets the absent teacher teach while he may be absent: a date of
    ``absence_dates`` in the week without his row is the whole date (as when ``absences`` is
    missing), and the lessons to cover are his absences by definition.
    """
    rows = request.absences or ()
    absent: dict[tuple[str, dt.date], frozenset[int] | None] = {}
    for row in rows:
        key = (row.teacher_id, row.date)
        if key in absent:
            msg = "two absences of one teacher on one date"
            raise InvalidRequestError(msg)
        absent[key] = row.lesson_nos
    teacher = request.absent_teacher_id
    completed = 0
    for date in sorted(request.absence_dates & set(request.week)):
        if (teacher, date) not in absent:
            absent[teacher, date] = None
            completed += request.absences is not None
    own = absent[teacher, request.date]
    needed = frozenset(s.lesson_no for s in request.lessons)
    if own is not None and not needed <= own:
        absent[teacher, request.date] = own | needed
        completed += 1
    return absent, completed


def _check_lessons_to_cover(request: SubstitutionRequest, week: Week) -> None:
    approved = {o.slot for o in request.approved}
    if len(set(request.lessons)) != len(request.lessons):
        msg = "a lesson to cover is listed twice"
        raise InvalidRequestError(msg)
    known = {c.id for c in request.school.classes}
    for slot in request.lessons:
        lesson = week.lessons.get(slot)
        if slot.date != request.date or lesson is None:
            msg = "a lesson to cover is not a lesson of the date"
            raise InvalidRequestError(msg)
        if slot.class_id not in known:
            msg = "a lesson to cover of a class outside the snapshot"
            raise InvalidRequestError(msg)
        if lesson.teacher_id != request.absent_teacher_id:
            msg = "a lesson to cover is not the absent teacher's"
            raise InvalidRequestError(msg)
        if slot in approved:  # one approved overlay per slot (partial unique index in timetable)
            msg = "a lesson to cover already has an approved substitution"
            raise InvalidRequestError(msg)
        if week.in_person(slot.class_id, slot.date) is None:
            msg = "a lesson to cover has no day format"
            raise InvalidRequestError(msg)
    _check_every_absent_lesson_is_listed(request, week)


def _check_every_absent_lesson_is_listed(request: SubstitutionRequest, week: Week) -> None:
    """With explicit absences, every lesson of the absent teacher in an hour his row marks absent
    must be covered: a missing one would stay with him silently."""
    if request.absences is None:
        return  # the whole-date fallback is a guess, not data: nothing to hold the list to
    row = next(
        (
            a
            for a in request.absences
            if (a.teacher_id, a.date) == (request.absent_teacher_id, request.date)
        ),
        None,
    )
    if row is None:
        return
    listed = set(request.lessons)
    for slot, lesson in week.lessons.items():
        mine = lesson.teacher_id == request.absent_teacher_id and slot.date == request.date
        if (
            mine
            and (row.lesson_nos is None or slot.lesson_no in row.lesson_nos)
            and slot not in listed
        ):
            msg = "a lesson of the absent teacher in an absent hour is not listed"
            raise InvalidRequestError(msg)
