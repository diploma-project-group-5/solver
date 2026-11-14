"""Phase B leaves no gaps ("windows") inside a class day (bug 11, Trello #72).

The demo timetable had remote class-days like lessons 1, 3, 6: the old objective (lessons after
the class's k-th period) priced a late start and a hole in the middle the same. Holes now come
first, a late start second; feasibility (König) is untouched since nothing here is a hard rule.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from lad_solver.app.solve import PHASE_B_DAY_DET
from lad_solver.domain.result import SolveStatus
from lad_solver.model.cpsat import SolverConfig
from lad_solver.model.phase_b import DayAssignment, DayDemand, Periods, assign_day

DEMO = Path(__file__).resolve().parent / "data" / "demo_phase_b_days.json"
FOUND = {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}


def demo_day(day: int) -> tuple[int, list[DayDemand]]:
    """Phase A loads of one day of the demo school (copied from the platform seed)."""
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    demands = [DayDemand(k, c, t, n) for k, c, t, n in data["days"][str(day)]]
    return data["lessons_per_day"], demands


def class_periods(demands: Sequence[DayDemand], periods: Periods) -> dict[str, list[int]]:
    by_class: defaultdict[str, list[int]] = defaultdict(list)
    for d in demands:
        by_class[d.class_id].extend(periods[d.key])
    return {c: sorted(ps) for c, ps in by_class.items()}


def gapped(demands: Sequence[DayDemand], periods: Periods) -> dict[str, list[int]]:
    """Classes whose lessons are not consecutive (max - min + 1 != count)."""
    return {
        c: ps for c, ps in class_periods(demands, periods).items() if ps[-1] - ps[0] + 1 != len(ps)
    }


def assert_valid_grid(
    demands: Sequence[DayDemand], lessons_per_day: int, assigned: DayAssignment
) -> Periods:
    assert assigned.status in FOUND
    periods = assigned.periods
    assert periods is not None
    by_class, by_teacher = Counter[tuple[str, int]](), Counter[tuple[str, int]]()
    for d in demands:
        assert len(periods[d.key]) == d.count
        for p in periods[d.key]:
            assert 1 <= p <= lessons_per_day
            by_class[d.class_id, p] += 1
            by_teacher[d.teacher_id, p] += 1
    assert max((*by_class.values(), *by_teacher.values()), default=1) == 1
    return periods


def _assign(
    demands: Sequence[DayDemand],
    lessons_per_day: int,
    config: SolverConfig,
    fixed: Sequence[tuple[int, int]] = (),
    hint: Mapping[int, Sequence[int]] | None = None,
) -> Periods:
    assigned = assign_day(1, demands, lessons_per_day, fixed, hint, config=config)
    return assert_valid_grid(demands, lessons_per_day, assigned)


# ---------------------------------------------------------------- demo school


@pytest.mark.slow
@pytest.mark.parametrize("day", [1, 2, 3, 4, 5])
def test_phase_b_demo_days_have_no_holes(day: int) -> None:
    lessons_per_day, demands = demo_day(day)
    config = SolverConfig(time_limit_s=30, max_deterministic_time=PHASE_B_DAY_DET)

    periods = _assign(demands, lessons_per_day, config)

    assert gapped(demands, periods) == {}


@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    [
        # what the worker runs with a deterministic budget (the seed build)
        SolverConfig(time_limit_s=30, random_seed=0, max_deterministic_time=PHASE_B_DAY_DET),
        # the wall-clock day limit of a 60 s solve: min(15, 0.15 * 60) / 5 days, 2 workers
        SolverConfig(time_limit_s=1.8, num_workers=2, random_seed=0),
    ],
    ids=["deterministic", "wall-1.8s"],
)
def test_phase_b_demo_day_reaches_no_holes_within_the_production_budget(
    config: SolverConfig,
) -> None:
    lessons_per_day, demands = demo_day(1)

    assigned = assign_day(1, demands, lessons_per_day, config=config, fallback=False)

    periods = assert_valid_grid(demands, lessons_per_day, assigned)
    assert gapped(demands, periods) == {}


# ---------------------------------------------------------------- hand-made days


def test_phase_b_holes_first_then_late_start() -> None:
    # Shared teacher "s" teaches all three classes once: periods 1, 2, 3. Each class also has one
    # lesson of its own teacher. A gap-free day exists only if one class starts with lesson 2.
    demands = [
        *(DayDemand(i, f"c{i}", "s", 1) for i in range(3)),
        *(DayDemand(10 + i, f"c{i}", f"own{i}", 1) for i in range(3)),
    ]

    periods = _assign(demands, 3, SolverConfig(time_limit_s=10))

    assert gapped(demands, periods) == {}
    firsts = [ps[0] for ps in class_periods(demands, periods).values()]
    assert sorted(firsts) == [1, 1, 2]


def test_phase_b_starts_at_1_when_possible() -> None:
    # Two classes, two teachers, two lessons each: both fit into periods 1-2 of a 4-period day.
    demands = [
        DayDemand(0, "a", "t1", 1),
        DayDemand(1, "a", "t2", 1),
        DayDemand(2, "b", "t1", 1),
        DayDemand(3, "b", "t2", 1),
    ]

    periods = _assign(demands, 4, SolverConfig(time_limit_s=10))

    assert class_periods(demands, periods) == {"a": [1, 2], "b": [1, 2]}


def test_fixed_forcing_hole_still_returns_grid() -> None:
    # Substitutions pin lessons 1 and 3: the hole is unavoidable, the day must still come back.
    demands = [DayDemand(0, "a", "t1", 1), DayDemand(1, "a", "t2", 1)]

    assigned = assign_day(1, demands, 3, fixed=[(0, 1), (1, 3)], config=SolverConfig(10))

    assert assigned.status in FOUND
    assert assigned.periods == {0: (1,), 1: (3,)}
