"""The substitution engine on hand-made weeks (ADR 2026-11-15-solver-substitution-engine).

Budget and determinism, stop and time limit, and the rules the step models follow where the
contract vectors do not reach (stale days over the passport, in-person days, one format per
teacher, windows and tie-breaks). The vectors and the golden checks live in
``tests/adapters/test_substitution_vectors.py`` and ``tests/test_substitution_properties.py``.
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast
from uuid import UUID

import pytest

import substitution_golden
from lad_solver.adapters.kafka.mapping import substitution_data
from lad_solver.app import substitute as substitute_module
from lad_solver.app.substitute import STEP1_SHARE, STEP2_SHARE, substitute
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
from lad_solver.domain.result import DayFormat, Lesson, SolveStatus, StopReason
from lad_solver.domain.substitution import (
    Absence,
    Hour,
    Overlay,
    OverlayKind,
    ProposalKind,
    Slot,
    SubstitutionParams,
    SubstitutionRequest,
    SubstitutionResult,
    baseline_week,
)
from lad_solver.model import substitution as model
from lad_solver.model.cpsat import SolverConfig
from lad_solver.model.cpsat import run as cpsat_run
from lad_solver.model.stop import StopHandle
from substitution_builders import request_json

MON = dt.date(2026, 11, 16)  # ISO week 2026-W47
WEEK = tuple(MON + dt.timedelta(days=i) for i in range(7))
TUE, WED, THU, FRI = WEEK[1], WEEK[2], WEEK[3], WEEK[4]
PARAMS = SubstitutionParams(time_limit_s=30.0)

type Row = tuple[str, int, int, str, str]  # class, day, lesson_no, subject, teacher
type Json = dict[str, Any]


def week(  # noqa: PLR0913 - a keyword-only builder for readable fixtures
    rows: Sequence[Row],
    *,
    cover: Sequence[tuple[str, int]],
    absent: str = "t1",
    date: dt.date = TUE,
    remote: Sequence[tuple[str, int]] = (),
    teachers: dict[str, int] | None = None,
    unavailable: dict[str, tuple[int, ...]] | None = None,
    capacity: Passport | AlertLimits | None = None,
    lessons_per_day: int = 6,
    staff: int = 0,
    no_shelter: Sequence[str] = (),
    absences: Sequence[Absence] | None = None,
    non_school_dates: Sequence[dt.date] | None = (),
    not_before: dt.date = MON,
    approved: Sequence[Overlay] = (),
    exhausted: Sequence[str] = (),
    school_days: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> SubstitutionRequest:
    """A request whose curriculum rows follow the template (so the week has no plan conflict).

    ``exhausted`` teachers get a weekly limit equal to their load: no lesson more (step 1 off).
    """
    hours: dict[tuple[str, str], tuple[str, int]] = {}
    for c, _, _, s, t in rows:
        hours[c, s] = (t, hours.get((c, s), (t, 0))[1] + 1)
    classes = sorted({c for c, *_ in rows})
    subjects = sorted({s for *_, s, _ in rows})
    teacher_ids = sorted({t for *_, t in rows} | set(teachers or {}))
    monday = date - dt.timedelta(days=date.isoweekday() - 1)
    held = {d for d in school_days if monday + dt.timedelta(d - 1) not in (non_school_dates or ())}
    limits = dict(teachers or {})
    for t in exhausted:
        limits[t] = sum(1 for _, d, _, _, x in rows if x == t and d in held)
    alert = isinstance(capacity, AlertLimits)
    school = Problem(
        kind=TimetableKind.ALERT if alert else TimetableKind.MAIN,
        school_days=school_days,
        lessons_per_day=lessons_per_day,
        staff=staff,
        capacity=capacity or Passport(10_000),
        classes=tuple(SchoolClass(c, 10, priority=False) for c in classes),
        subjects=tuple(
            Subject(s, priority_area=False, no_shelter=s in no_shelter) for s in subjects
        ),
        teachers=tuple(
            Teacher(t, limits.get(t, 30), frozenset((unavailable or {}).get(t, ())))
            for t in teacher_ids
        ),
        rows=tuple(CurriculumRow(c, s, t, n) for (c, s), (t, n) in sorted(hours.items())),
    )
    return SubstitutionRequest(
        school=school,
        date=date,
        absent_teacher_id=absent,
        lessons=tuple(Slot(date, c, n) for c, n in cover),
        not_before=not_before,
        absence_dates=frozenset({date}),
        template_lessons=tuple(Lesson(*row) for row in rows),
        template_formats=tuple(
            DayFormat(c, d, in_person=(c, d) not in remote) for c in classes for d in school_days
        ),
        approved=tuple(approved),
        absences=None if absences is None else tuple(absences),
        non_school_dates=None if non_school_dates is None else frozenset(non_school_dates),
    )


def kinds(result: SubstitutionResult) -> list[tuple[str, int, str, Any]]:
    """(class, lesson, kind, teacher or moved_to or step) per proposal."""
    out: list[tuple[str, int, str, Any]] = []
    for p in result.proposals:
        detail: Any = p.teacher_id if p.kind is ProposalKind.TEACHER else p.moved_to or p.step
        out.append((p.slot.class_id, p.slot.lesson_no, p.kind.value, detail))
    return out


def solved(
    request: SubstitutionRequest,
    params: SubstitutionParams = PARAMS,
    stop: StopHandle | None = None,
) -> SubstitutionResult:
    """The engine's answer, checked by validator-lib: the week gets no new conflict (ADR)."""
    result = substitute(request, params, stop)
    data = request_json(request)
    rows = substitution_data(UUID(data["job_id"]), result, None)["substitution"]
    pair = substitution_golden.weeks(data, cast("Json", rows)["proposals"])
    outcome = substitution_golden.new_conflicts({"case": pair})["case"]
    assert substitution_golden.added(outcome) == []
    return result


def daily(*lessons: tuple[str, int, str, str], days: Sequence[int] = (1, 2, 3, 4, 5)) -> list[Row]:
    return [(c, d, n, s, t) for d in days for c, n, s, t in lessons]


# A two-class week: t1 teaches maths in a L1 and b L2 every day; t2 and t3 teach the rest.
BASE = daily(("a", 1, "math", "t1"), ("a", 2, "ukr", "t2"), ("b", 1, "ukr", "t2"),
             ("b", 2, "math", "t1"), ("b", 3, "eng", "t3"))  # fmt: skip


# ---------------------------------------------------------------- steps and budget


@pytest.fixture
def configs(monkeypatch: pytest.MonkeyPatch) -> list[SolverConfig]:
    """Every SolverConfig the step models run with (cpsat.run spied in model.substitution)."""
    seen: list[SolverConfig] = []
    real = cpsat_run

    def spy(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - forwards to cpsat.run
        seen.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(model, "run", spy)
    return seen


def test_every_run_is_deterministic_and_shares_the_budget(configs: list[SolverConfig]) -> None:
    """One worker, the seed and the memory cap on every run; step 3 dates share the rest."""
    # t1 is absent on Tuesday, Wednesday and Thursday; nobody else has room: steps 1 and 2 have
    # nothing to choose (no run), step 3 tries Wednesday (absent, no run), then Monday.
    request = week(BASE, cover=[("a", 1), ("b", 2)], exhausted=("t2", "t3"),
                   absences=[Absence("t1", d) for d in (TUE, WED, THU)])  # fmt: skip
    params = replace(PARAMS, max_deterministic_time=12.0, random_seed=3, max_memory_mb=128)

    result = solved(request, params)

    assert kinds(result) == [("a", 1, "moved", Hour(MON, 3)), ("b", 2, "moved", Hour(MON, 4))]
    [monday] = configs
    assert (monday.num_workers, monday.random_seed, monday.max_memory_mb) == (1, 3, 128)
    assert monday.time_limit_s <= params.time_limit_s
    # The window is Wed, Mon, Thu, Fri; Monday is its second date: 12 / 3 of what is left.
    assert monday.max_deterministic_time == pytest.approx(12.0 / 3)


def test_step1_and_step2_get_their_shares(configs: list[SolverConfig]) -> None:
    # t1 absent in L1 only: t3 takes a L1 at step 1 (one run).
    request = week(BASE, cover=[("a", 1)], absences=[Absence("t1", TUE, frozenset({1}))])
    params = replace(PARAMS, max_deterministic_time=20.0)

    result = solved(request, params)

    assert kinds(result) == [("a", 1, "teacher", "t3")]
    [step1] = configs
    assert step1.max_deterministic_time == pytest.approx(20.0 * STEP1_SHARE)


def test_a_step_gets_what_the_step_before_left(configs: list[SolverConfig]) -> None:
    # Nobody has room: step 1 has nothing to choose, step 2 moves a L1 within Tuesday.
    request = week(BASE, cover=[("a", 1)], exhausted=("t2", "t3"),
                   absences=[Absence("t1", TUE, frozenset({1}))])  # fmt: skip
    params = replace(PARAMS, max_deterministic_time=20.0)

    result = solved(request, params)

    [step2] = configs  # step 1 had no option, so no run: its 10 % moves on
    assert step2.max_deterministic_time == pytest.approx(20.0 * (STEP1_SHARE + STEP2_SHARE))
    assert kinds(result) == [("a", 1, "moved", Hour(TUE, 3))]


def test_a_stop_before_the_first_step_leaves_every_lesson_self_study() -> None:
    stop = StopHandle()
    stop.stop()

    result = solved(week(BASE, cover=[("a", 1), ("b", 2)]), PARAMS, stop)

    assert [(p.kind, p.step) for p in result.proposals] == [(ProposalKind.SELF_STUDY, 1)] * 2
    assert (result.status, result.stats.stop_reason) == (SolveStatus.FEASIBLE, StopReason.STOPPED)
    assert result.stats.runs == 0


def test_a_stop_after_step_1_keeps_what_step_1_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = StopHandle()
    real = model.solve_step

    def then_stop(*args: Any, **kwargs: Any) -> model.StepOutcome:  # noqa: ANN401
        outcome = real(*args, **kwargs)
        stop.stop()
        return outcome

    monkeypatch.setattr(substitute_module, "solve_step", then_stop)
    # a L1 -> t3 at step 1; in L2 t3 teaches c, so b L2 would wait for step 2.
    request = week([*BASE, ("c", 2, 2, "art", "t3")], cover=[("a", 1), ("b", 2)],
                   exhausted=("t2",), absences=[Absence("t1", TUE, frozenset({1, 2}))])  # fmt: skip

    result = solved(request, PARAMS, stop)

    assert kinds(result) == [("a", 1, "teacher", "t3"), ("b", 2, "self_study", 1)]
    assert (result.status, result.stats.stop_reason) == (SolveStatus.FEASIBLE, StopReason.STOPPED)


def test_out_of_wall_time_every_step_left_is_skipped() -> None:
    result = solved(week(BASE, cover=[("a", 1)]), SubstitutionParams(time_limit_s=1e-9))

    assert kinds(result) == [("a", 1, "self_study", 1)]
    assert (result.status, result.stats.stop_reason) == (
        SolveStatus.FEASIBLE,
        StopReason.TIME_LIMIT,
    )


@pytest.mark.parametrize(
    ("status", "reason", "anomalies"),
    [
        (SolveStatus.UNKNOWN, StopReason.TIME_LIMIT, ()),
        (SolveStatus.INFEASIBLE, StopReason.COMPLETED, ((1, SolveStatus.INFEASIBLE),)),
        (SolveStatus.MODEL_INVALID, StopReason.COMPLETED, ((1, SolveStatus.MODEL_INVALID),)),
    ],
)
def test_a_step_without_a_solution_proposes_nothing_and_the_next_steps_run(
    monkeypatch: pytest.MonkeyPatch,
    status: SolveStatus,
    reason: StopReason,
    anomalies: tuple[tuple[int, SolveStatus], ...],
) -> None:
    real = model.solve_step
    calls: list[model.Mode] = []

    def first_fails(*args: Any, **kwargs: Any) -> model.StepOutcome:  # noqa: ANN401
        calls.append(args[3])
        if len(calls) == 1:
            return model.StepOutcome(status, (), ran=True)
        return real(*args, **kwargs)

    monkeypatch.setattr(substitute_module, "solve_step", first_fails)
    request = week(BASE, cover=[("a", 1)], absences=[Absence("t1", TUE, frozenset({1}))])

    result = solved(request, PARAMS)

    assert calls == [model.Mode.TEACHER, model.Mode.SAME_DAY]
    assert kinds(result) == [("a", 1, "moved", Hour(TUE, 3))]
    assert (result.status, result.stats.stop_reason) == (SolveStatus.FEASIBLE, reason)
    assert result.stats.anomalies == anomalies


def test_a_proven_run_is_optimal_with_its_telemetry() -> None:
    request = week(BASE, cover=[("a", 1)], absences=[Absence("t1", TUE, frozenset({1}))])

    result = solved(request, PARAMS)

    assert result.status is SolveStatus.OPTIMAL
    assert result.stats.stop_reason is StopReason.COMPLETED
    assert (result.stats.runs, result.stats.covered) == (1, (1, 0, 0))
    assert result.stats.deterministic_time > 0


def test_support_moves_requested_change_nothing() -> None:
    request = week(BASE, cover=[("a", 1), ("b", 2)], exhausted=("t2", "t3"))

    asked = solved(replace(request, allow_support_moves=True), PARAMS)

    assert asked.proposals == solved(request, PARAMS).proposals


# ---------------------------------------------------------------- step 1

THIRD_CLASS = daily(("c", 1, "art", "t5"), ("c", 2, "math", "t4"), ("c", 3, "ukr", "t6"))


def test_step1_prefers_a_teacher_on_profile() -> None:
    # In Tuesday L1 t3, t4 and t6 are free; only t4 has a maths row.
    request = week([*BASE, *THIRD_CLASS], cover=[("a", 1)],
                   absences=[Absence("t1", TUE, frozenset({1}))])  # fmt: skip

    assert kinds(solved(request, PARAMS)) == [("a", 1, "teacher", "t4")]


def test_step1_breaks_a_tie_by_the_lower_teacher_id() -> None:
    # Without t4 (no room left) t3 and t6 are both off profile: t3 comes first by id.
    request = week([*BASE, *THIRD_CLASS], cover=[("a", 1)], exhausted=("t4",),
                   absences=[Absence("t1", TUE, frozenset({1}))])  # fmt: skip

    [proposal] = solved(request, PARAMS).proposals
    assert (proposal.teacher_id, proposal.off_profile) == ("t3", True)


def test_step1_respects_the_headroom_of_a_substitute() -> None:
    # t4 (maths) is free in L1 and L2 but has room for one more lesson only.
    request = week([*BASE, *daily(("c", 3, "math", "t4"))], cover=[("a", 1), ("b", 2)],
                   teachers={"t4": 6}, exhausted=("t2", "t3"), non_school_dates=None)  # fmt: skip

    result = solved(request, PARAMS)

    teachers = [p.teacher_id for p in result.proposals]
    assert teachers.count("t4") == 1
    assert [p.kind for p in result.proposals].count(ProposalKind.SELF_STUDY) == 1


def test_step1_never_takes_a_teacher_whose_day_is_already_mixed() -> None:
    # On Tuesday t3 teaches b in person and c remotely (a stale week): no new lesson that day.
    request = week([*BASE, ("c", 2, 4, "art", "t3")], cover=[("a", 1)], remote=[("c", 2)],
                   exhausted=("t2",), absences=[Absence("t1", TUE, frozenset({1}))])  # fmt: skip

    result = solved(request, PARAMS)

    assert kinds(result) == [("a", 1, "moved", Hour(TUE, 3))]


# ---------------------------------------------------------------- steps 2 and 3


def test_step2_picks_the_hour_without_a_window_then_the_earliest() -> None:
    # a on Tuesday: L1 maths (t1 absent), L2 ukr, L3 eng; L4..L6 are free (L5, L6 leave a gap).
    request = week(
        [*BASE, *daily(("a", 3, "eng", "t3"))],
        cover=[("a", 1)],
        exhausted=("t2", "t3"),
        absences=[Absence("t1", TUE, frozenset({1}))],
    )

    assert kinds(solved(request, PARAMS)) == [("a", 1, "moved", Hour(TUE, 4))]


def test_step3_moves_the_lesson_whose_leaving_opens_no_window() -> None:
    # a on Tuesday: L1 ukr, L2 maths (t1), L3 eng, L4 maths (t1); on Monday t1 and a are both
    # free in L3 only. Moving L2 away would open a window on Tuesday, moving L4 would not.
    tuesday = [("a", 2, 1, "ukr", "t2"), ("a", 2, 2, "math", "t1"), ("a", 2, 3, "eng", "t3"),
               ("a", 2, 4, "math", "t1")]  # fmt: skip
    monday = [("a", 1, 1, "ukr", "t2"), ("a", 1, 2, "ukr", "t2"),
              *(("a", 1, n, "eng", "t3") for n in (4, 5, 6)),
              *(("b", 1, n, "math", "t1") for n in (1, 2, 4, 5, 6))]  # fmt: skip
    request = week([*tuesday, *monday], cover=[("a", 2), ("a", 4)], exhausted=("t2", "t3"),
                   non_school_dates=[WED, THU, FRI])  # fmt: skip

    result = solved(request, PARAMS)

    assert kinds(result) == [("a", 2, "self_study", 3), ("a", 4, "moved", Hour(MON, 3))]


def test_step3_never_empties_an_in_person_class_day() -> None:
    # c's only Tuesday lesson is t1's maths in L3; Monday is the only date of the window.
    rows = [*BASE, ("c", 2, 3, "math", "t1")]
    common: dict[str, Any] = {
        "cover": [("c", 3)],
        "exhausted": ("t2", "t3"),
        "non_school_dates": [WED, THU, FRI],
    }

    in_person = solved(week(rows, **common), PARAMS)
    remote = solved(week(rows, remote=[("c", 2)], **common), PARAMS)

    assert kinds(in_person) == [("c", 3, "self_study", 3)]
    [(_, _, kind, target)] = kinds(remote)
    assert (kind, target.date) == ("moved", MON)


def test_step3_gives_a_teacher_one_format_on_a_date_without_his_lessons() -> None:
    # t1 has no Wednesday lesson; a is in person and b remote that day: only one of them goes.
    # Moving b's L2 would open a window on Tuesday (b L1, L3), so a moves, to its earliest hour.
    rows = [r for r in BASE if not (r[1] == 3 and r[4] == "t1")]
    request = week(rows, cover=[("a", 1), ("b", 2)], remote=[("b", 3)], exhausted=("t2", "t3"),
                   non_school_dates=[MON, THU, FRI], not_before=TUE)  # fmt: skip

    result = solved(request, PARAMS)

    assert kinds(result) == [("a", 1, "moved", Hour(WED, 1)), ("b", 2, "self_study", 3)]


def test_step3_lets_a_stale_day_over_the_passport_stay_but_not_grow() -> None:
    """Capacity, "not worse than the baseline": a day already over the passport takes a lesson
    of a teacher who is on site anyway (no change), never one that brings him on site."""
    rows = [r for r in BASE if not (r[1] == 3 and r[4] == "t1")]  # t1 has no Wednesday lesson
    # Wednesday: 20 pupils + 2 staff + t2 and t3 = 24 > passport 23 (stale).
    common: dict[str, Any] = {
        "cover": [("a", 1)],
        "exhausted": ("t2", "t3"),
        "staff": 2,
        "capacity": Passport(23),
        "non_school_dates": [MON, THU, FRI],
        "not_before": TUE,
    }

    off_site = solved(week(rows, **common), PARAMS)
    on_site = solved(week([*rows, ("b", 3, 5, "math", "t1")], **common), PARAMS)

    assert kinds(off_site) == [("a", 1, "self_study", 3)]
    [(_, _, kind, target)] = kinds(on_site)
    assert (kind, target.date) == ("moved", WED)
    for result in (off_site, on_site):  # refused by the room, not by a broken model
        assert (result.status, result.stats.anomalies) == (SolveStatus.OPTIMAL, ())


def test_moves_respect_unavailable_days_absences_and_lessons_per_day() -> None:
    # Window: Wednesday (absent), Monday (unavailable), Thursday: t1 teaches L1-L2 and is
    # absent in L3-L4, and there is no L5 with lessons_per_day 4.
    request = week(BASE, cover=[("a", 1), ("b", 2)], exhausted=("t2", "t3"), lessons_per_day=4,
                   unavailable={"t1": (1,)}, non_school_dates=[FRI],
                   absences=[Absence("t1", TUE), Absence("t1", WED),
                             Absence("t1", THU, frozenset({3, 4}))])  # fmt: skip

    assert kinds(solved(request, PARAMS)) == [
        ("a", 1, "self_study", 3),
        ("b", 2, "self_study", 3),
    ]
    longer = solved(replace(request, school=replace(request.school, lessons_per_day=6)), PARAMS)
    targets = {p.moved_to for p in longer.proposals}
    assert targets == {Hour(THU, 5), Hour(THU, 6)}  # one lesson per hour for t1


def test_a_no_shelter_lesson_never_moves_into_an_in_person_day() -> None:
    # b's maths becomes PE (no shelter); t1 teaches nothing on Wednesday.
    rows = [
        (c, d, n, "pe" if (c, s) == ("b", "math") else s, t)
        for c, d, n, s, t in BASE
        if not (d == 3 and t == "t1")
    ]
    common: dict[str, Any] = {
        "cover": [("b", 2)],
        "exhausted": ("t2", "t3"),
        "capacity": AlertLimits(seats=100, passport=1000),
        "no_shelter": ("pe",),
        "non_school_dates": [MON, THU, FRI],
        "not_before": TUE,
    }

    in_person = solved(week(rows, **common), PARAMS)
    remote = solved(week(rows, remote=[("b", 3)], **common), PARAMS)

    assert kinds(in_person) == [("b", 2, "self_study", 3)]
    assert [p.kind for p in remote.proposals] == [ProposalKind.MOVED]


# ---------------------------------------------------------------- the week and its dates


def test_the_baseline_week_applies_every_kind_of_approved_substitution() -> None:
    request = week(BASE, cover=[("a", 1)], approved=[
        Overlay(Slot(MON, "a", 1), OverlayKind.TEACHER, "t3"),
        Overlay(Slot(MON, "b", 2), OverlayKind.MOVED, moved_to=Hour(MON, 5)),
        Overlay(Slot(WED, "b", 3), OverlayKind.SELF_STUDY),
        Overlay(Slot(THU, "b", 3), OverlayKind.CANCELLED),
    ])  # fmt: skip

    w = baseline_week(request)

    assert w.lessons[Slot(MON, "a", 1)].teacher_id == "t3"
    assert Slot(MON, "b", 2) not in w.lessons
    assert w.lessons[Slot(MON, "b", 5)].teacher_id == "t1"
    assert w.class_busy(Slot(WED, "b", 3))  # self-study holds the slot
    assert not w.class_busy(Slot(THU, "b", 3))  # cancelled frees it
    assert (w.load("t3"), w.load("t1")) == (5 + 1 - 1 - 1, 10 - 1)
    assert w.absent("t1", Hour(TUE, 5))  # without absences: the whole of every absence date
    assert not w.absent("t1", Hour(WED, 1))


def test_non_school_dates_hold_no_lesson_and_leave_the_window() -> None:
    request = week(BASE, cover=[("a", 1)], non_school_dates=[WED], not_before=MON)

    w = baseline_week(request)

    assert not any(s.date == WED for s in w.lessons)
    assert w.load("t1") == 8
    assert request.window() == (MON, THU, FRI)  # Monday and Wednesday tie: the later is gone


@pytest.mark.parametrize(
    ("date", "not_before", "window"),
    [
        (TUE, MON, (WED, MON, THU, FRI)),  # equal distance: the later date first
        (TUE, TUE, (WED, THU, FRI)),  # never before not_before
        (FRI, FRI, ()),  # the window never leaves the ISO week
        (WED, MON, (THU, TUE, FRI, MON)),
    ],
)
def test_step3_window_order(
    date: dt.date, not_before: dt.date, window: tuple[dt.date, ...]
) -> None:
    request = week(BASE, cover=[], date=date, not_before=not_before)

    assert request.window() == window


def test_without_non_school_dates_step3_is_off() -> None:
    request = week(BASE, cover=[("a", 1)], non_school_dates=None, exhausted=("t2", "t3"))

    result = solved(request, PARAMS)

    assert request.window() == ()
    assert kinds(result) == [("a", 1, "self_study", 2)]


@pytest.mark.parametrize(
    "monday",
    [dt.date(2026, 3, 23), dt.date(2026, 10, 19)],
    ids=["dst-starts-sunday-29-03", "dst-ends-sunday-25-10"],
)
def test_weeks_with_a_dst_change_count_calendar_days(monday: dt.date) -> None:
    """Dates are calendar dates (Europe/Kyiv): a 23- or 25-hour Sunday changes no distance."""
    dates = tuple(monday + dt.timedelta(days=i) for i in range(7))
    request = week(BASE, cover=[("a", 1), ("b", 2)], date=dates[4], not_before=dates[0],
                   exhausted=("t2", "t3"))  # fmt: skip

    result = solved(request, PARAMS)

    assert request.window() == dates[3::-1]  # Thursday, Wednesday, Tuesday, Monday
    assert {p.moved_to.date for p in result.proposals if p.moved_to} == {dates[3]}


# ---------------------------------------------------------------- review fixes (15.11.2026)


def test_a_feasible_step_keeps_the_best_proposals_it_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a step budget cut (FEASIBLE) the best solution found stays: it is not thrown away."""

    def feasible(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - forwards to cpsat.run
        solver, info = cpsat_run(*args, **kwargs)
        return solver, replace(info, status=SolveStatus.FEASIBLE)

    monkeypatch.setattr(model, "run", feasible)
    # a L1 -> t3 at step 1 (t3 teaches c in L2); b L2 waits for step 2 and moves to L4.
    request = week([*BASE, ("c", 2, 2, "art", "t3")], cover=[("a", 1), ("b", 2)],
                   exhausted=("t2",), absences=[Absence("t1", TUE, frozenset({1, 2}))])  # fmt: skip

    result = solved(request, PARAMS)

    assert kinds(result) == [("a", 1, "teacher", "t3"), ("b", 2, "moved", Hour(TUE, 4))]
    assert (result.status, result.stats.stop_reason) == (
        SolveStatus.FEASIBLE,
        StopReason.TIME_LIMIT,
    )


def test_step3_places_the_lessons_jointly_over_the_dates_added() -> None:
    """Dates are added one at a time and the lessons placed over all of them together: b takes
    Wednesday's only free hour, a goes to Monday (placing a on Wednesday first would leave b
    nowhere, since b is busy on Monday in that hour)."""
    rows = [r for r in BASE if not (r[0] == "b" and r[2] == 3 and r[1] == 3)]  # b free Wed L3
    rows += [("c", d, n, "math", "t1") for d in (1, 3) for n in (4, 5, 6)]  # t1 free: L3 only
    request = week(rows, cover=[("a", 1), ("b", 2)], exhausted=("t2", "t3"),
                   absences=[Absence("t1", TUE)], non_school_dates=[THU, FRI])  # fmt: skip

    result = solved(request, PARAMS)

    assert request.window() == (WED, MON)
    assert kinds(result) == [("a", 1, "moved", Hour(MON, 3)), ("b", 2, "moved", Hour(WED, 3))]
    assert result.stats.runs == 2  # Wednesday alone placed one lesson; Wednesday + Monday both


def test_a_cancelled_lesson_keeps_an_in_person_day_held() -> None:
    """validator-lib: self-study and cancelled lessons keep an in-person day held, so the last
    real lesson of the day may move away when a cancelled one stays."""
    rows = [*BASE, ("c", 2, 3, "math", "t1"), ("c", 2, 4, "ukr", "t2")]
    common: dict[str, Any] = {
        "cover": [("c", 3)],
        "exhausted": ("t3",),
        "teachers": {"t2": 0},  # over his limit already: no new lesson even after the cancel
        "non_school_dates": [WED, THU, FRI],
        "absences": [
            Absence("t1", TUE, frozenset({3, 4, 5, 6})),
            Absence("t2", TUE, frozenset({4})),
        ],
    }
    cancelled = [Overlay(Slot(TUE, "c", 4), OverlayKind.CANCELLED)]
    self_studied = [Overlay(Slot(TUE, "c", 4), OverlayKind.SELF_STUDY)]
    alone = [r for r in rows if r[:3] != ("c", 2, 4)]

    kept = solved(week(alone, **common), PARAMS)
    after_cancel = solved(week(rows, approved=cancelled, **common), PARAMS)
    after_self_study = solved(week(rows, approved=self_studied, **common), PARAMS)

    assert kinds(kept) == [("c", 3, "self_study", 3)]
    for result in (after_cancel, after_self_study):
        [(_, _, kind, target)] = kinds(result)
        assert (kind, target.date) == ("moved", MON)


def test_approved_substitutions_apply_in_any_order() -> None:
    """A move into a slot another overlay frees, and a swap of two moves, whatever the order."""
    into_cancelled = [
        Overlay(Slot(THU, "b", 3), OverlayKind.CANCELLED),
        Overlay(Slot(MON, "a", 2), OverlayKind.MOVED, moved_to=Hour(THU, 3)),
    ]
    swap = [
        Overlay(Slot(MON, "b", 1), OverlayKind.MOVED, moved_to=Hour(MON, 2)),
        Overlay(Slot(MON, "b", 2), OverlayKind.MOVED, moved_to=Hour(MON, 1)),
    ]
    chain = [  # b's Wednesday L3 goes to Thursday L6, a lesson of Tuesday takes its slot
        Overlay(Slot(WED, "b", 3), OverlayKind.MOVED, moved_to=Hour(THU, 6)),
        Overlay(Slot(TUE, "b", 3), OverlayKind.MOVED, moved_to=Hour(WED, 3)),
    ]
    for overlays in (into_cancelled, swap, chain):
        forward = baseline_week(week(BASE, cover=[("a", 1)], approved=overlays))
        backward = baseline_week(week(BASE, cover=[("a", 1)], approved=overlays[::-1]))
        assert dict(forward.lessons) == dict(backward.lessons)
    swapped = baseline_week(week(BASE, cover=[("a", 1)], approved=swap))
    assert swapped.lessons[Slot(MON, "b", 1)].subject_id == "math"
    assert swapped.lessons[Slot(MON, "b", 2)].subject_id == "ukr"


def test_approved_substitutions_on_a_non_school_date_change_nothing_but_moves() -> None:
    """A lesson of a non-school date is not held: an overlay on it is moot, unless it moved the
    lesson to another date before the date stopped being a school day."""
    overlays = [
        Overlay(Slot(WED, "a", 2), OverlayKind.TEACHER, "t3"),
        Overlay(Slot(WED, "b", 3), OverlayKind.MOVED, moved_to=Hour(THU, 6)),
    ]

    w = baseline_week(week(BASE, cover=[("a", 1)], approved=overlays, non_school_dates=[WED]))

    assert not any(s.date == WED for s in w.lessons)
    assert w.lessons[Slot(THU, "b", 6)].teacher_id == "t3"  # held on Thursday
    assert w.overlays_ignored == 1


def test_ids_that_left_the_snapshot_stay_in_the_week_like_in_validator_lib() -> None:
    """timetable projects active ids only; the template keeps the old ones. t3 was deleted and
    class b archived: their lessons stay put, b counts for nothing, t3 still counts on site and
    is never a candidate."""
    request = week(BASE, cover=[("a", 1)], absences=[Absence("t1", TUE, frozenset({1}))])
    school = request.school
    stale = replace(
        request,
        school=replace(
            school,
            classes=tuple(c for c in school.classes if c.id != "b"),
            teachers=tuple(t for t in school.teachers if t.id != "t3"),
            rows=tuple(r for r in school.rows if r.class_id != "b" and r.teacher_id != "t3"),
        ),
    )

    w = baseline_week(stale)
    result = solved(stale, PARAMS)

    assert w.load("t1") == 5  # a's maths only: b is unknown
    assert w.teachers_on_site(TUE) == 2  # t1 and t2 in a; b's lessons count for nothing
    assert all(p.teacher_id != "t3" for p in result.proposals)
    assert result.status is SolveStatus.OPTIMAL


def test_absences_are_read_the_safe_way() -> None:
    """A date of the absence without the absent teacher's row is the whole date; a row that
    misses a lesson to cover is extended to it. Both are counted for the logs."""
    absences = [Absence("t1", TUE, frozenset({2}))]  # misses a L1, and Wednesday has no row
    request = replace(
        week(BASE, cover=[("a", 1), ("b", 2)], absences=absences),
        absence_dates=frozenset({TUE, WED}),
    )

    w = baseline_week(request)

    assert w.absent("t1", Hour(TUE, 1))
    assert w.absent("t1", Hour(WED, 5))
    assert not w.absent("t1", Hour(TUE, 3))
    assert w.absences_completed == 2


def test_a_six_day_school_has_saturday_in_the_window() -> None:
    request = week(BASE, cover=[("a", 1), ("b", 2)], date=FRI, not_before=FRI,
                   exhausted=("t2", "t3"), school_days=(1, 2, 3, 4, 5, 6))  # fmt: skip

    result = solved(request, PARAMS)

    assert request.window() == (WEEK[5],)
    assert {p.moved_to.date for p in result.proposals if p.moved_to} == {WEEK[5]}
