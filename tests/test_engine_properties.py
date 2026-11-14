"""Metamorphic properties of the engine (ADR 2026-11-04 §5), on small timetable-first problems.

Problems are derived from a randomly drawn clash-free weekly grid, so they are feasible by
construction: INFEASIBLE from the engine would be a bug.
"""

from collections import Counter, defaultdict
from dataclasses import replace

from hypothesis import example, given, settings
from hypothesis import strategies as st
from ortools.sat.python import cp_model

import golden
from builders import OTHER, OUTSIDE, PRIO, klass, problem, row, teacher
from lad_solver.app.solve import solve
from lad_solver.domain.problem import (
    AlertLimits,
    Params,
    Passport,
    Problem,
    Subject,
    TimetableKind,
)
from lad_solver.domain.result import Result, SolveStatus
from lad_solver.model.cpsat import SolverConfig
from lad_solver.model.phase_b import DayDemand, Periods, assign_day, assign_periods

PARAMS = Params(time_limit_s=10, num_workers=1, random_seed=0)
SETTINGS = settings(max_examples=40, deadline=None, derandomize=True)
SUBJECT_IDS = (PRIO, OTHER, OUTSIDE)


@st.composite
def timetable_first(draw: st.DrawFn) -> Problem:
    n_days = draw(st.integers(1, 3))
    lessons_per_day = draw(st.integers(1, 3))
    n_classes = draw(st.integers(1, 3))
    n_teachers = draw(st.integers(1, 3))
    days = tuple(range(1, n_days + 1))
    hours = Counter[tuple[str, str, str]]()
    busy_days: dict[str, set[int]] = {f"t{t}": set() for t in range(n_teachers)}
    # One teacher per class x subject (spec 04, no split groups): a second one is a broken snapshot.
    owner: dict[tuple[str, str], str] = {}
    for d in days:
        for _ in range(lessons_per_day):
            free = [f"t{t}" for t in range(n_teachers)]
            for c in range(n_classes):
                pick = draw(st.none() | st.sampled_from(free)) if free else None
                if pick is None:
                    continue
                subjects = [s for s in SUBJECT_IDS if owner.get((f"c{c}", s), pick) == pick]
                if not subjects:
                    continue
                subject = draw(st.sampled_from(subjects))
                owner[f"c{c}", subject] = pick
                free.remove(pick)
                busy_days[pick].add(d)
                hours[f"c{c}", subject, pick] += 1
    load = Counter[str]()
    for (_, _, t), h in hours.items():
        load[t] += h
    teachers = tuple(
        teacher(
            t,
            limit=load[t] + draw(st.integers(0, 3)),
            unavailable=tuple(d for d in days if d not in busy and draw(st.booleans())),
        )
        for t, busy in busy_days.items()
    )
    staff = draw(st.integers(0, 5))
    return problem(
        classes=tuple(
            klass(f"c{c}", draw(st.integers(0, 30)), priority=draw(st.booleans()))
            for c in range(n_classes)
        ),
        teachers=teachers,
        rows=tuple(row(c, s, t, h) for (c, s, t), h in sorted(hours.items())),
        capacity=Passport(staff + draw(st.integers(0, 70))),
        days=days,
        lessons_per_day=lessons_per_day,
        staff=staff,
    )


def _optimal(p: Problem) -> Result:
    result = solve(p, PARAMS)
    assert result.status is SolveStatus.OPTIMAL, result.status
    assert golden.violations(p, result) == []
    assert result.stats.objective == golden.objective(p, result).value
    return result


def _objective(result: Result) -> int:
    assert result.stats.objective is not None
    return result.stats.objective


@SETTINGS
@given(timetable_first())
def test_timetable_first_problem_is_never_infeasible(p: Problem) -> None:
    _optimal(p)


@SETTINGS
@given(timetable_first(), st.integers(1, 40))
def test_bigger_passport_never_lowers_the_objective(p: Problem, extra: int) -> None:
    bigger = replace(p, capacity=Passport(p.capacity.passport + extra))
    assert _objective(_optimal(bigger)) >= _objective(_optimal(p))


@SETTINGS
@given(timetable_first())
def test_passport_equal_to_staff_means_no_in_person_priority_lessons(p: Problem) -> None:
    assert _optimal(replace(p, capacity=Passport(p.staff))).stats.primary == 0


@SETTINGS
@given(timetable_first())
def test_alert_with_unlimited_seats_and_no_outside_subjects_equals_main(p: Problem) -> None:
    sheltered = tuple(Subject(s.id, s.priority_area, no_shelter=False) for s in p.subjects)
    main = replace(p, subjects=sheltered)
    alert = replace(
        main,
        kind=TimetableKind.ALERT,
        capacity=AlertLimits(seats=10**6, passport=p.capacity.passport),
    )
    assert _objective(_optimal(alert)) == _objective(_optimal(main))


@st.composite
def alert_first(draw: st.DrawFn) -> Problem:
    """A timetable-first problem as 1.2: seats may bind, and no-shelter rows stay (remote days).

    "Everything remote" satisfies seats, passport >= staff and the no-shelter rule, so these
    problems stay feasible by construction.
    """
    p = draw(timetable_first())
    seats = draw(st.integers(0, sum(c.students for c in p.classes)))
    return replace(
        p,
        kind=TimetableKind.ALERT,
        capacity=AlertLimits(seats=seats, passport=p.capacity.passport),
    )


@SETTINGS
@given(alert_first())
def test_alert_problem_is_never_infeasible_and_respects_seats(p: Problem) -> None:
    _optimal(p)  # validator-lib checks seats, passport and no-shelter lessons on remote days


@SETTINGS
@given(alert_first(), st.integers(1, 40))
def test_more_seats_never_lower_the_objective(p: Problem, extra: int) -> None:
    assert isinstance(p.capacity, AlertLimits)
    more = replace(p, capacity=replace(p.capacity, seats=p.capacity.seats + extra))
    assert _objective(_optimal(more)) >= _objective(_optimal(p))


@st.composite
def day_loads(draw: st.DrawFn) -> tuple[int, list[DayDemand]]:
    """Loads of one day with class-day and teacher-day totals <= L (what phase A produces)."""
    lessons_per_day = draw(st.integers(1, 6))
    n_classes, n_teachers = draw(st.integers(1, 5)), draw(st.integers(1, 5))
    class_load, teacher_load = Counter[int](), Counter[int]()
    counts = Counter[tuple[int, int]]()
    for _ in range(draw(st.integers(0, 25))):
        c, t = draw(st.integers(0, n_classes - 1)), draw(st.integers(0, n_teachers - 1))
        if class_load[c] < lessons_per_day and teacher_load[t] < lessons_per_day:
            class_load[c] += 1
            teacher_load[t] += 1
            counts[c, t] += 1
    demands = [
        DayDemand(i, f"c{c}", f"t{t}", n) for i, ((c, t), n) in enumerate(sorted(counts.items()))
    ]
    return lessons_per_day, demands


@settings(max_examples=60, deadline=None, derandomize=True)
@given(day_loads())
def test_phase_b_is_always_feasible_for_phase_a_loads(case: tuple[int, list[DayDemand]]) -> None:
    lessons_per_day, demands = case
    periods = assign_periods(1, demands, lessons_per_day, config=SolverConfig(time_limit_s=5))
    assert periods is not None
    by_class, by_teacher = Counter[tuple[str, int]](), Counter[tuple[str, int]]()
    for d in demands:
        assert len(periods[d.key]) == d.count
        for p in periods[d.key]:
            assert 1 <= p <= lessons_per_day
            by_class[d.class_id, p] += 1
            by_teacher[d.teacher_id, p] += 1
    assert max((*by_class.values(), *by_teacher.values()), default=1) == 1


def _gap_free_exists(lessons_per_day: int, demands: list[DayDemand]) -> bool:
    """Oracle: a clash-free day where every class has consecutive lessons (hard constraint)."""
    model = cp_model.CpModel()
    periods = range(1, lessons_per_day + 1)
    x = {(d.key, p): model.new_bool_var("") for d in demands for p in periods}
    by_class: defaultdict[str, list[DayDemand]] = defaultdict(list)
    by_teacher: defaultdict[str, list[DayDemand]] = defaultdict(list)
    for d in demands:
        model.add(sum(x[d.key, p] for p in periods) == d.count)
        by_class[d.class_id].append(d)
        by_teacher[d.teacher_id].append(d)
    for group in by_teacher.values():
        for p in periods:
            model.add_at_most_one(x[d.key, p] for d in group)
    for group in by_class.values():
        k = sum(d.count for d in group)
        starts = {s: model.new_bool_var("") for s in range(1, lessons_per_day - k + 2)}
        model.add_exactly_one(starts.values())
        for p in periods:  # period p is busy iff the chosen block [s, s + k) covers it
            covering = [b for s, b in starts.items() if s <= p < s + k]
            model.add(sum(x[d.key, p] for d in group) == sum(covering))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    status = solver.solve(model)
    assert status in {cp_model.OPTIMAL, cp_model.INFEASIBLE}
    return status == cp_model.OPTIMAL


def _gaps(demands: list[DayDemand], periods: Periods) -> int:
    by_class: defaultdict[str, list[int]] = defaultdict(list)
    for d in demands:
        by_class[d.class_id].extend(periods[d.key])
    return sum(max(ps) - min(ps) + 1 - len(ps) for ps in by_class.values())


@settings(max_examples=100, deadline=None, derandomize=True)
@given(day_loads())
# the old objective (lessons after the k-th period) was happy with c1 at 2 and 4
@example((4, [DayDemand(0, "c0", "t0", 1), DayDemand(1, "c1", "t0", 2)]))
def test_phase_b_has_no_gaps_whenever_a_gap_free_day_exists(
    case: tuple[int, list[DayDemand]],
) -> None:
    lessons_per_day, demands = case
    assigned = assign_day(1, demands, lessons_per_day, config=SolverConfig(time_limit_s=5))
    assert assigned.status is SolveStatus.OPTIMAL  # tiny days: proven, not a timeout
    assert assigned.periods is not None  # the grid is always full, gaps or not
    if _gap_free_exists(lessons_per_day, demands):
        assert _gaps(demands, assigned.periods) == 0


def test_phase_b_respects_fixed_lessons_and_hint() -> None:
    demands = [DayDemand(0, "a", "t1", 1), DayDemand(1, "a", "t2", 1)]
    config = SolverConfig(time_limit_s=5)
    periods = assign_periods(1, demands, 3, fixed=[(0, 3)], hint={1: [1]}, config=config)
    assert periods is not None
    assert periods[0] == (3,)
    assert periods[1] in {(1,), (2,)}
