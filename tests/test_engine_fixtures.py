"""Hand fixtures with OPTIMAL objective values (ADR 2026-11-04 §5, testing.md capacity cases).

Assertions are on objective values and formats, not on exact grids; every solved result also
passes the independent oracle.
"""

import pytest

import oracle
from builders import OTHER, OUTSIDE, PRIO, klass, problem, row, teacher
from lad_solver.app.solve import solve
from lad_solver.domain.problem import AlertLimits, Params, Passport, Problem, TimetableKind
from lad_solver.domain.result import Result, SolveStatus, StopReason

PARAMS = Params(time_limit_s=10, num_workers=1, random_seed=0)
ALERT = TimetableKind.ALERT


def _solve(p: Problem) -> Result:
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.OPTIMAL
    assert oracle.violations(p, result) == []
    recomputed = oracle.objective(p, result)
    assert result.stats.objective == recomputed.value
    assert (result.stats.primary, result.stats.secondary) == (
        recomputed.primary,
        recomputed.secondary,
    )
    assert result.stats.stop_reason is StopReason.COMPLETED
    return result


def _one_class(
    capacity: Passport | AlertLimits, kind: TimetableKind = TimetableKind.MAIN
) -> Problem:
    # 20 pupils + 5 staff + 1 teacher = 26 people, 20 seats.
    return problem(
        classes=(klass("a", 20),),
        teachers=(teacher("t1"),),
        rows=(row("a", PRIO, "t1", 2),),
        capacity=capacity,
        kind=kind,
        days=(1,),
        staff=5,
    )


@pytest.mark.parametrize(("passport", "objective"), [(26, 40), (25, 0)])
def test_passport_exactly_at_limit_is_in_person_one_more_is_remote(
    passport: int, objective: int
) -> None:
    assert _solve(_one_class(Passport(passport))).stats.objective == objective


@pytest.mark.parametrize(
    ("seats", "passport", "objective"),
    [(20, 26, 40), (19, 26, 0), (20, 25, 0)],
    ids=["both-at-limit", "seats-violated", "passport-violated"],
)
def test_alert_conditions_seats_and_passport_separately(
    seats: int, passport: int, objective: int
) -> None:
    assert _solve(_one_class(AlertLimits(seats, passport), ALERT)).stats.objective == objective


def test_model_1_1_vs_1_2_differ_by_seats() -> None:
    assert _solve(_one_class(Passport(1000))).stats.objective == 40
    assert _solve(_one_class(AlertLimits(19, 1000), ALERT)).stats.objective == 0


@pytest.mark.parametrize(
    ("capacity", "kind"),
    [
        (Passport(2**64), TimetableKind.MAIN),
        (AlertLimits(seats=2**63, passport=2**64), ALERT),
    ],
    ids=["1.1", "1.2"],
)
def test_capacity_beyond_int64_is_an_unlimited_shelter(
    capacity: Passport | AlertLimits, kind: TimetableKind
) -> None:
    # CP-SAT takes int64 constants only: such limits must not crash the model build.
    p = problem(
        classes=(klass("a", 20),),
        teachers=(teacher("t1"),),
        rows=(row("a", PRIO, "t1", 2),),
        capacity=capacity,
        kind=kind,
        days=(1,),
        staff=2**63,
    )
    assert _solve(p).stats.objective == 40


def test_teacher_with_several_in_person_lessons_is_counted_once() -> None:
    p = problem(
        classes=(klass("a", 20),),
        teachers=(teacher("t1"),),
        rows=(row("a", PRIO, "t1", 3),),
        capacity=Passport(21),
        days=(1,),
    )
    assert _solve(p).stats.objective == 60


def test_teacher_with_only_remote_lessons_is_not_counted() -> None:
    p = problem(
        classes=(klass("a", 20), klass("b", 30)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 2), row("b", PRIO, "t2", 2)),
        capacity=Passport(21),  # a + t1 fit exactly; counting t2 as well would not
        days=(1,),
    )
    result = _solve(p)
    assert result.stats.objective == 40
    assert oracle.in_person_days(result) == {("a", 1)}


@pytest.mark.parametrize(("second", "objective"), [("t1", 0), ("t2", 20)], ids=["shared", "own"])
def test_shared_teacher_forces_one_format_for_both_classes(second: str, objective: int) -> None:
    p = problem(
        classes=(klass("a", 20), klass("b", 20)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 1), row("b", PRIO, second, 1)),
        capacity=Passport(21),  # one class + its teacher fits, both classes do not
        days=(1,),
    )
    assert _solve(p).stats.objective == objective


@pytest.mark.parametrize(
    ("kind", "objective"), [(ALERT, 3 * 40 + 1), (TimetableKind.MAIN, 3 * 40 + 2)]
)
def test_no_shelter_only_on_remote_days_and_only_for_alert(
    kind: TimetableKind, objective: int
) -> None:
    p = problem(
        classes=(klass("a", 20, priority=True),),  # W = 1 * 2 + 1 = 3
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 2), row("a", OUTSIDE, "t2", 2)),
        capacity=AlertLimits(1000, 1000) if kind is ALERT else Passport(1000),
        kind=kind,
        days=(1, 2),
    )
    result = _solve(p)
    assert result.stats.objective == objective
    if kind is ALERT:
        in_person = oracle.in_person_days(result)
        assert all(
            (x.class_id, x.day) not in in_person for x in result.lessons if x.subject_id == OUTSIDE
        )


def test_empty_day_is_never_in_person() -> None:
    p = problem(
        classes=(klass("a", 20, priority=True),),
        teachers=(teacher("t1"),),
        rows=(row("a", PRIO, "t1", 1),),
        days=(1, 2),
    )
    result = _solve(p)
    assert result.stats.objective == 3 * 20 + 1  # one in-person day, not two
    assert len(oracle.in_person_days(result)) == 1


def test_class_priority_flag_only_breaks_ties() -> None:
    p = problem(
        classes=(klass("a", 20, priority=True), klass("b", 21)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 1), row("b", PRIO, "t2", 1)),
        capacity=Passport(22),  # room for one class and its teacher
        days=(1,),
    )
    result = _solve(p)
    assert oracle.in_person_days(result) == {("b", 1)}  # 21 pupils beat the flag
    assert result.stats.objective == 2 * 21


def test_non_priority_lessons_do_not_count_in_primary() -> None:
    p = problem(
        classes=(klass("a", 20),),
        teachers=(teacher("t1"),),
        rows=(row("a", OTHER, "t1", 2),),
        days=(1,),
    )
    assert _solve(p).stats.primary == 0


def test_lessons_are_packed_from_the_first_period() -> None:
    p = problem(
        classes=(klass("a", 20),),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 1), row("a", OTHER, "t2", 1)),
        days=(1,),
        lessons_per_day=5,
    )
    assert sorted(x.lesson_no for x in _solve(p).lessons) == [1, 2]


def test_precheck_conflict_skips_cp_sat() -> None:
    p = _one_class(Passport(4))  # staff 5 > passport 4
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.INFEASIBLE
    assert result.stats.stop_reason is StopReason.PRECHECK
    assert result.lessons == ()


def test_invalid_input_is_model_invalid() -> None:
    p = problem(
        classes=(klass("a", 20),), teachers=(teacher("t1"),), rows=(row("a", PRIO, "zz", 1),)
    )
    assert solve(p, PARAMS).status is SolveStatus.MODEL_INVALID


def test_packing_infeasibility_reports_busiest_teachers() -> None:
    # Pre-checks pass, but both teachers can only come on day 1 and L = 1.
    p = problem(
        classes=(klass("a", 20),),
        teachers=(teacher("t1", unavailable=(2,)), teacher("t2", unavailable=(2,))),
        rows=(row("a", PRIO, "t1", 1), row("a", OTHER, "t2", 1)),
        days=(1, 2),
        lessons_per_day=1,
    )
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.INFEASIBLE
    assert result.stats.quality_relaxed  # the uncapped model was tried as well
    rules = [(c.rule.value, c.teacher_id, c.hint) for c in result.conflicts]
    assert rules == [("packing", None, False), ("packing", "t1", True), ("packing", "t2", True)]


def test_stop_before_start_returns_unknown_without_timetable() -> None:
    from lad_solver.model.stop import StopHandle  # noqa: PLC0415

    handle = StopHandle()
    handle.stop()
    result = solve(_one_class(Passport(100)), PARAMS, handle)
    assert result.status is SolveStatus.UNKNOWN
    assert result.stats.stop_reason is StopReason.STOPPED
