"""Metamorphic properties of the engine (ADR 2026-11-04 §5), on small timetable-first problems.

Problems are derived from a randomly drawn clash-free weekly grid, so they are feasible by
construction: INFEASIBLE from the engine would be a bug.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
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

# OPTIMAL is asserted below, so the budget is deterministic (one worker, fixed seed): a loaded
# runner cannot turn a proof into a timeout. The wall clock is only a safety net.
PARAMS = Params(time_limit_s=60, num_workers=1, random_seed=0, max_deterministic_time=10)
DAY_CONFIG = SolverConfig(time_limit_s=60, num_workers=1, random_seed=0, max_deterministic_time=5)
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
    periods = assign_periods(1, demands, lessons_per_day, config=DAY_CONFIG)
    assert periods is not None
    by_class, by_teacher = Counter[tuple[str, int]](), Counter[tuple[str, int]]()
    for d in demands:
        assert len(periods[d.key]) == d.count
        for p in periods[d.key]:
            assert 1 <= p <= lessons_per_day
            by_class[d.class_id, p] += 1
            by_teacher[d.teacher_id, p] += 1
    assert max((*by_class.values(), *by_teacher.values()), default=1) == 1


type Pins = list[tuple[int, int]]  # (demand key, lesson number), as ``assign_day(fixed=...)``


@st.composite
def pinned_day_loads(draw: st.DrawFn) -> tuple[int, list[DayDemand], Pins]:
    """``day_loads`` plus clash-free pins (substitutions) that may force holes.

    Pins never put two lessons of one class or one teacher into a period, but they need not
    extend to a full day: then the day is infeasible (pins are hard), and the oracle says so.
    """
    lessons_per_day, demands = draw(day_loads())
    pins: Pins = []
    teacher_busy: set[tuple[str, int]] = set()
    by_class: defaultdict[str, list[DayDemand]] = defaultdict(list)
    for d in demands:
        by_class[d.class_id].extend([d] * d.count)  # one entry per lesson
    for group in by_class.values():
        # maybe pin two lessons of the class at least two periods apart: a hole unless the
        # class has a free lesson for each idle period between them (and teachers allow it)
        if len(group) < 2 or lessons_per_day < 3 or draw(st.booleans()):
            continue
        first = draw(st.integers(1, lessons_per_day - 2))
        second = draw(st.integers(first + 2, lessons_per_day))
        for d, p in ((group[0], first), (group[1], second)):
            if (d.teacher_id, p) not in teacher_busy:
                teacher_busy.add((d.teacher_id, p))
                pins.append((d.key, p))
    return lessons_per_day, demands, pins


def _min_holes(
    lessons_per_day: int, demands: list[DayDemand], fixed: Sequence[tuple[int, int]] = ()
) -> int | None:
    """Oracle: the fewest holes of a clash-free day with ``fixed`` pins; ``None`` = infeasible.

    Counted independently of phase B's span model: a hole is an idle period of a class with a
    lesson before it and a lesson after it, all as hard constraints, minimised.
    """
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
    for key, p in fixed:
        model.add(x[key, p] == 1)
    holes: list[cp_model.IntVar] = []
    for group in by_class.values():
        busy = {p: sum(x[d.key, p] for d in group) for p in periods}
        for p in periods:
            model.add(busy[p] <= 1)
            before, after, hole = (model.new_bool_var("") for _ in range(3))
            for q in periods:  # before / after >= "a lesson at q", tight at the minimum
                if q != p:
                    model.add((before if q < p else after) >= busy[q])
            model.add(hole >= before + after - busy[p] - 1)
            holes.append(hole)
    model.minimize(sum(holes))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    status = solver.solve(model)
    assert status in {cp_model.OPTIMAL, cp_model.INFEASIBLE}
    return round(solver.objective_value) if status == cp_model.OPTIMAL else None


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
    assigned = assign_day(1, demands, lessons_per_day, config=DAY_CONFIG)
    assert assigned.status is SolveStatus.OPTIMAL  # tiny days: proven, not a timeout
    assert assigned.periods is not None  # the grid is always full, gaps or not
    fewest = _min_holes(lessons_per_day, demands)
    assert fewest is not None  # phase A loads always fit (König)
    assert _gaps(demands, assigned.periods) == fewest  # in particular 0 when a gap-free day exists


@settings(max_examples=400, deadline=None, derandomize=True)
@given(pinned_day_loads())
# pins at 1 and 4 of a 3-lesson class: one hole is forced, not two
@example(
    (
        4,
        [DayDemand(0, "a", "t1", 1), DayDemand(1, "a", "t2", 1), DayDemand(2, "a", "t3", 1)],
        [(0, 1), (1, 4)],
    )
)
# pins at 1 and 3 leave room for the free lesson at 2: no hole at all
@example(
    (
        3,
        [DayDemand(0, "a", "t1", 1), DayDemand(1, "a", "t2", 1), DayDemand(2, "a", "t3", 1)],
        [(0, 1), (1, 3)],
    )
)
# b's hole (1, 3) is forced; t2 is pinned to b at 3, so a gets t2's 1-2 without a hole
@example(
    (
        3,
        [DayDemand(0, "b", "t1", 1), DayDemand(1, "b", "t2", 1), DayDemand(2, "a", "t2", 2)],
        [(0, 1), (1, 3)],
    )
)
def test_phase_b_holes_are_minimal_even_when_pins_force_them(
    case: tuple[int, list[DayDemand], Pins],
) -> None:
    lessons_per_day, demands, pins = case
    fewest = _min_holes(lessons_per_day, demands, pins)

    assigned = assign_day(1, demands, lessons_per_day, pins, config=DAY_CONFIG)

    if fewest is None:  # the pins do not extend to a clash-free day
        assert assigned.status is SolveStatus.INFEASIBLE
        assert assigned.periods is None
        return
    assert assigned.status is SolveStatus.OPTIMAL  # tiny days: proven, not a timeout
    assert assigned.periods is not None
    assert all(p in assigned.periods[key] for key, p in pins)
    assert _gaps(demands, assigned.periods) == fewest


def test_phase_b_respects_fixed_lessons_and_hint() -> None:
    demands = [DayDemand(0, "a", "t1", 1), DayDemand(1, "a", "t2", 1)]
    periods = assign_periods(1, demands, 3, fixed=[(0, 3)], hint={1: [1]}, config=DAY_CONFIG)
    assert periods is not None
    assert periods[0] == (3,)
    assert periods[1] in {(1,), (2,)}
