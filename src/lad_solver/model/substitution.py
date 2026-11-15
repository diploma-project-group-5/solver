"""Substitution model (ADR 2026-11-15-solver-substitution-engine): one builder, three step modes.

A step is a small CP-SAT model over the lessons still open. Each open lesson picks at most one
option, an option being "teacher t teaches it at hour h" (the class stays the same):

- step 1 (``TEACHER``): another teacher at the lesson's own hour;
- step 2 (``SAME_DAY``): the absent teacher at another hour of the absence date;
- step 3 (``OTHER_DATE``): the absent teacher at an hour of the neighbour dates given, the first
  ``k`` dates of the window (spec §06: neighbour days are added one at a time).

A lesson that picks nothing stays open (self-study once the steps are over), so "no change" is
always a solution and is the hint of every model.

The hard constraints keep the effective week no worse than the baseline week. They only make the
proposals feasible: the verdict is validator-lib's (``newConflicts`` in timetable), there is no
second implementation of the rules here.

- An option's teacher is free at that hour, present (absences, unavailable days) and, on that date,
  teaches only classes of the class's format. A teacher and a class get one lesson per hour.
- Weekly limit: a teacher's new lessons <= max(limit, baseline load) - current load.
- Capacity: the teachers an option brings on site fit max(passport, baseline load) of that date
  (``model/capacity``); the formats are fixed, so the pupils (1.2 seats) never change.
- A lesson moves only into a class hour free in the baseline: a slot freed by another proposal of
  the same job stays free (no chains of moves while allow_support_moves is off, contracts README
  item 9), never past lessons_per_day, never a no-shelter lesson into an in-person day (1.2), and
  an in-person class day keeps a lesson, a self-study or a cancelled lesson.

Objectives, lexicographic by weights: step 1 coverage >> off profile >> new teachers on site >>
teacher rank (id order); steps 2 and 3 coverage >> date rank (window order) >> class windows
(gaps, ADR 2026-11-14-solver-phase-b-no-gaps) >> earliest hour. Step 2's "fewer changed lessons"
equals the coverage while only the absent teacher's lessons move, so it has no term of its own.
"""

import datetime as dt
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ortools.sat.python import cp_model

from lad_solver.domain.problem import Teacher, TimetableKind
from lad_solver.domain.result import SolveStatus
from lad_solver.domain.substitution import Hour, Slot, Week, WeekLesson
from lad_solver.model.capacity import add_day_capacity, baseline_room, day_people
from lad_solver.model.cpsat import SolverConfig, run
from lad_solver.model.stop import StopHandle

type Expr = cp_model.LinearExpr | int

_FOUND = frozenset({SolveStatus.OPTIMAL, SolveStatus.FEASIBLE})


class Mode(StrEnum):
    TEACHER = "teacher"  # step 1: a free teacher in the same slot
    SAME_DAY = "same_day"  # step 2: another hour of the absence date
    OTHER_DATE = "other_date"  # step 3: an hour of the neighbour dates given


@dataclass(frozen=True, slots=True)
class Option:
    """``lesson`` (open) taught by ``teacher_id`` at ``hour``."""

    lesson: WeekLesson
    teacher_id: str
    hour: Hour
    in_person: bool  # the class's (fixed) format on the option's date
    off_profile: bool  # the teacher has no curriculum row of the lesson's subject
    rank: int  # tie-break, smaller first: the teacher's id rank (step 1) or the lesson number
    date_rank: int = 0  # step 3: the date's place in the window (nearest, a tie to the later)

    @property
    def target(self) -> Slot:
        return self.lesson.slot.at(self.hour)


@dataclass(slots=True)
class StepModel:
    model: cp_model.CpModel
    options: tuple[Option, ...]
    picks: list[cp_model.IntVar]  # picks[k]: options[k] is taken


@dataclass(frozen=True, slots=True)
class StepOutcome:
    status: SolveStatus  # of the CP-SAT run; OPTIMAL when there was nothing to search
    picks: tuple[Option, ...]  # at most one per open lesson
    ran: bool  # a CP-SAT run happened (its deterministic time counts)
    deterministic_time: float = 0.0


def solve_step(  # noqa: PLR0913 - keyword-only stop and targets
    baseline: Week,
    current: Week,
    open_lessons: Sequence[WeekLesson],
    mode: Mode,
    config: SolverConfig,
    *,
    stop: StopHandle | None = None,
    targets: Sequence[dt.date] = (),
) -> StepOutcome:
    """Build and solve one step; a model without options is not run (nothing to choose)."""
    built = build_step(baseline, current, open_lessons, mode, targets)
    if not built.options:
        return StepOutcome(SolveStatus.OPTIMAL, (), ran=False)
    solver, info = run(built.model, config, stop)
    picks: tuple[Option, ...] = ()
    if info.status in _FOUND:
        picks = tuple(
            option
            for option, var in zip(built.options, built.picks, strict=True)
            if solver.boolean_value(var)
        )
    return StepOutcome(info.status, picks, ran=True, deterministic_time=info.deterministic_time)


def build_step(
    baseline: Week,
    current: Week,
    open_lessons: Sequence[WeekLesson],
    mode: Mode,
    targets: Sequence[dt.date] = (),
) -> StepModel:
    """``baseline``: the week without this job; ``current``: with the proposals fixed so far.

    ``targets`` (OTHER_DATE only): the dates to move to, in window order.
    """
    if (mode is Mode.OTHER_DATE) != bool(targets):
        msg = "target dates are given exactly for OTHER_DATE"
        raise ValueError(msg)
    options = tuple(_options(baseline, current, open_lessons, mode, targets))
    model = cp_model.CpModel()
    picks = [model.new_bool_var(f"x_{k}") for k in range(len(options))]
    for var in picks:
        model.add_hint(var, 0)  # no change: always feasible
    built = StepModel(model, options, picks)
    _one_option_per_lesson(built)
    _one_lesson_per_hour(built)
    _weekly_limits(built, baseline, current)
    _one_format_per_teacher_and_date(built, current)
    arrivals = _capacity(built, baseline, current)
    if mode is Mode.OTHER_DATE:
        _in_person_days_keep_a_lesson(built, current)
    if mode is Mode.TEACHER:
        _teacher_objective(built, len(open_lessons), arrivals)
    else:
        _move_objective(built, _windows(built, current))
    return built


# ---------------------------------------------------------------- options


def _options(
    baseline: Week,
    current: Week,
    open_lessons: Sequence[WeekLesson],
    mode: Mode,
    targets: Sequence[dt.date],
) -> list[Option]:
    options: list[Option] = []
    for lesson in open_lessons:
        if mode is Mode.TEACHER:
            options.extend(_teacher_options(baseline, current, lesson))
        elif mode is Mode.SAME_DAY:
            options.extend(_move_options(baseline, current, lesson, lesson.slot.date, 0))
        else:
            for date_rank, date in enumerate(targets):
                options.extend(_move_options(baseline, current, lesson, date, date_rank))
    return options


def _teacher_options(baseline: Week, current: Week, lesson: WeekLesson) -> list[Option]:
    """Step 1: a teacher who works that day in the class's format, free and present then."""
    school = current.request.school
    slot = lesson.slot
    in_person = current.in_person(slot.class_id, slot.date)
    if in_person is None:
        return []  # no format that day (the request gate refuses such a lesson to cover)
    profile = {(r.teacher_id, r.subject_id) for r in school.rows}
    options: list[Option] = []
    for rank, teacher in enumerate(sorted(school.teachers, key=lambda t: t.id)):
        if teacher.id == current.request.absent_teacher_id:
            continue
        if not current.teaches_on(teacher.id, slot.date):
            continue  # "works that day" = has a lesson that day (Roman, 15.11.2026)
        if _headroom(baseline, current, teacher) < 1:
            continue
        if _can_teach(current, teacher, slot.hour, in_person=in_person):
            off = (teacher.id, lesson.subject_id) not in profile
            options.append(
                Option(
                    lesson, teacher.id, slot.hour, in_person=in_person, off_profile=off, rank=rank
                )
            )
    return options


def _move_options(
    baseline: Week, current: Week, lesson: WeekLesson, date: dt.date, date_rank: int
) -> list[Option]:
    """Steps 2 and 3: the absent teacher at a free hour of ``date`` where he is present."""
    school = current.request.school
    slot = lesson.slot
    absent = next(t for t in school.teachers if t.id == current.request.absent_teacher_id)
    no_shelter = {s.id for s in school.subjects if s.no_shelter}
    in_person = current.in_person(slot.class_id, date)
    if in_person is None:
        return []  # no format that day: a lesson cannot be held there
    if school.kind is TimetableKind.ALERT and in_person and lesson.subject_id in no_shelter:
        return []  # "not held in the shelter" never goes to an in-person day (1.2)
    options: list[Option] = []
    for lesson_no in range(1, school.lessons_per_day + 1):
        hour = Hour(date, lesson_no)
        moved = slot.at(hour)
        # Free in the baseline too: a slot this job freed stays free (no chains of moves).
        free = not (baseline.class_busy(moved) or current.class_busy(moved))
        if moved == slot or not free or baseline.busy(absent.id, hour):
            continue
        if _can_teach(current, absent, hour, in_person=in_person):
            option = Option(
                lesson,
                absent.id,
                hour,
                in_person=in_person,
                off_profile=False,
                rank=lesson_no,
                date_rank=date_rank,
            )
            options.append(option)
    return options


def _can_teach(week: Week, teacher: Teacher, hour: Hour, *, in_person: bool) -> bool:
    """Present and free at ``hour``, and that date teaching only classes of this format."""
    if hour.date.isoweekday() in teacher.unavailable_days:
        return False
    if week.absent(teacher.id, hour) or week.busy(teacher.id, hour):
        return False
    return week.teacher_formats(teacher.id, hour.date) <= {in_person}


def _headroom(baseline: Week, current: Week, teacher: Teacher) -> int:
    """New lessons the weekly limit still allows: never worse than the baseline week."""
    return max(teacher.weekly_limit, baseline.load(teacher.id)) - current.load(teacher.id)


# ---------------------------------------------------------------- constraints


def _groups[K: Hashable](built: StepModel, key: dict[int, K]) -> dict[K, list[cp_model.IntVar]]:
    groups: defaultdict[K, list[cp_model.IntVar]] = defaultdict(list)
    for k, value in key.items():
        groups[value].append(built.picks[k])
    return groups


def _one_option_per_lesson(built: StepModel) -> None:
    by_lesson = _groups(built, {k: o.lesson.slot for k, o in enumerate(built.options)})
    for picks in by_lesson.values():
        if len(picks) > 1:
            built.model.add(sum(picks) <= 1)


def _one_lesson_per_hour(built: StepModel) -> None:
    """A teacher, and a class, get at most one lesson per hour."""
    options = built.options
    teacher = _groups(built, {k: (o.teacher_id, o.hour) for k, o in enumerate(options)})
    klass = _groups(built, {k: o.target for k, o in enumerate(options)})
    for group in (*teacher.values(), *klass.values()):
        if len(group) > 1:
            built.model.add(sum(group) <= 1)


def _weekly_limits(built: StepModel, baseline: Week, current: Week) -> None:
    """Options that give a teacher a lesson he did not have (moves keep the count)."""
    new = {
        k: o.teacher_id for k, o in enumerate(built.options) if o.teacher_id != o.lesson.teacher_id
    }
    teachers = {t.id: t for t in current.request.school.teachers}
    for teacher_id, picks in _groups(built, new).items():
        headroom = _headroom(baseline, current, teachers[teacher_id])
        if len(picks) > headroom:
            built.model.add(sum(picks) <= headroom)


def _one_format_per_teacher_and_date(built: StepModel, current: Week) -> None:
    """A teacher without lessons that date may take either format, but only one (spec §04)."""
    by_day: defaultdict[tuple[str, dt.date], list[int]] = defaultdict(list)
    for k, o in enumerate(built.options):
        by_day[o.teacher_id, o.hour.date].append(k)
    for (teacher_id, date), ks in by_day.items():
        formats = {built.options[k].in_person for k in ks}
        if formats != {True, False} or current.teacher_formats(teacher_id, date):
            continue  # one format only, or already fixed by his lessons (options filtered)
        in_person = built.model.new_bool_var(f"format_{teacher_id}_{date.isoformat()}")
        for k in ks:
            pick = built.picks[k]
            built.model.add(
                pick <= in_person if built.options[k].in_person else pick <= 1 - in_person
            )


def _capacity(built: StepModel, baseline: Week, current: Week) -> list[cp_model.IntVar]:
    """Teachers an option brings on site (``arrivals``) fit the room of that date."""
    school = current.request.school
    arrivals: defaultdict[dt.date, list[cp_model.IntVar]] = defaultdict(list)
    comes: dict[tuple[str, dt.date], cp_model.IntVar] = {}
    for k, o in enumerate(built.options):
        if not o.in_person or current.on_site(o.teacher_id, o.hour.date):
            continue
        key = (o.teacher_id, o.hour.date)
        if key not in comes:
            comes[key] = built.model.new_bool_var(f"on_site_{o.teacher_id}_{o.hour.date}")
            arrivals[o.hour.date].append(comes[key])
        built.model.add(built.picks[k] <= comes[key])
    max_pupils = sum(c.students for c in school.classes)
    for date, new in arrivals.items():
        pupils = baseline.pupils(date)
        before = day_people(pupils, school.staff, baseline.teachers_on_site(date))
        room = baseline_room(
            school.capacity,
            school.staff,
            before,
            max_pupils=max_pupils,
            max_teachers=len(school.teachers) + current.teachers_on_site(date),
        )
        teachers: Expr = current.teachers_on_site(date) + sum(new)
        add_day_capacity(built.model, date.isoweekday(), current.pupils(date), teachers, room)
    return list(comes.values())


def _in_person_days_keep_a_lesson(built: StepModel, current: Week) -> None:
    """Moving lessons to another date must leave an in-person class day something that holds it
    (a lesson, a self-study or a cancelled lesson, as validator-lib counts it)."""
    leaving = _groups(
        built,
        {
            k: (o.lesson.slot.class_id, o.lesson.slot.date)
            for k, o in enumerate(built.options)
            if o.lesson.slot.date != o.hour.date
        },
    )
    for (class_id, date), picks in leaving.items():
        if current.in_person(class_id, date):
            built.model.add(sum(picks) <= current.holds(class_id, date) - 1)


# ---------------------------------------------------------------- objectives


def _teacher_objective(built: StepModel, lessons: int, arrivals: list[cp_model.IntVar]) -> None:
    """Coverage >> off profile >> new teachers on site >> teacher rank."""
    options, picks = built.options, built.picks
    w_new = _bound(options, lambda o: o.rank) + 1
    w_off = w_new * (len(arrivals) + 1)
    w_cover = w_off * (lessons + 1)
    built.model.maximize(
        sum(
            (w_cover - w_off * int(o.off_profile) - o.rank) * x
            for o, x in zip(options, picks, strict=True)
        )
        - w_new * sum(arrivals)
    )


def _move_objective(built: StepModel, windows: list[cp_model.IntVar]) -> None:
    """Coverage >> date rank >> class windows >> earliest hour."""
    options, picks = built.options, built.picks
    w_window = _bound(options, lambda o: o.rank) + 1
    w_date = w_window * (len(windows) + 1)
    w_cover = w_date * (_bound(options, lambda o: o.date_rank) + 1)
    built.model.maximize(
        sum(
            (w_cover - w_date * o.date_rank - o.rank) * x
            for o, x in zip(options, picks, strict=True)
        )
        - w_window * sum(windows)
    )


def _bound(options: Sequence[Option], value: Callable[[Option], int]) -> int:
    """The largest total of ``value`` over a solution: per lesson its largest option."""
    worst: dict[Slot, int] = {}
    for o in options:
        worst[o.lesson.slot] = max(worst.get(o.lesson.slot, 0), value(o))
    return sum(worst.values())


def _windows(built: StepModel, current: Week) -> list[cp_model.IntVar]:
    """Gap indicators of every class day an option changes (empty hours between two held ones)."""
    days: set[tuple[str, dt.date]] = set()
    for o in built.options:
        class_id = o.lesson.slot.class_id
        days.update({(class_id, o.lesson.slot.date), (class_id, o.hour.date)})
    gaps: list[cp_model.IntVar] = []
    for class_id, date in sorted(days):
        gaps.extend(_gaps(built.model, _occupancy(built, current, class_id, date)))
    return gaps


def _occupancy(built: StepModel, current: Week, class_id: str, date: dt.date) -> list[Expr]:
    """Per hour: 1 if the class holds a lesson or a self-study then, as a 0/1 expression."""
    hours = max(
        [
            current.request.school.lessons_per_day,
            *(s.lesson_no for s in current.lessons if (s.class_id, s.date) == (class_id, date)),
        ]
    )
    occupied: list[Expr] = [
        int(current.class_busy(Slot(date, class_id, n))) for n in range(1, hours + 1)
    ]
    for o, x in zip(built.options, built.picks, strict=True):
        slot = o.lesson.slot
        if slot.class_id != class_id:
            continue
        if slot.date == date:
            occupied[slot.lesson_no - 1] -= x  # the lesson leaves its hour
        if o.hour.date == date:
            occupied[o.hour.lesson_no - 1] += x  # ... for this one
    return occupied


def _gaps(model: cp_model.CpModel, occupied: list[Expr]) -> list[cp_model.IntVar]:
    """g[h] >= before[h] + after[h] - occupied[h] - 1; minimisation makes it the gap count."""
    if all(isinstance(o, int) for o in occupied):
        return []  # the options never change this day
    gaps: list[cp_model.IntVar] = []
    for h in range(1, len(occupied) - 1):
        before, after, gap = (
            model.new_bool_var(f"{name}_{h + 1}") for name in ("in", "out", "gap")
        )
        for o in occupied[:h]:
            _at_least(model, before, o)
        for o in occupied[h + 1 :]:
            _at_least(model, after, o)
        model.add(gap >= before + after - occupied[h] - 1)
        gaps.append(gap)
    return gaps


def _at_least(model: cp_model.CpModel, var: cp_model.IntVar, value: Expr) -> None:
    if isinstance(value, int):
        if value > 0:
            model.add(var >= value)
        return
    model.add(var >= value)
