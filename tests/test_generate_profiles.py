"""Generator profiles: ``top2026`` follows the 2026/27 typical plans; ``legacy`` stays the default."""

import pytest

from lad_solver.cli.generate import PRIMARY_GRADES, PROFILES, generate_problem
from lad_solver.domain.prechecks import precheck

# weekly hours incl. physical education: orders No 743 (1-4), No 235/1430 (5-9), No 408/890 (10-11)
TYPICAL_TOTAL = {1: 23, 2: 25, 3: 26, 4: 26, 5: 31, 6: 34, 7: 35, 8: 36, 9: 36, 10: 38, 11: 38}


def test_top2026_weekly_hours_match_the_typical_plans() -> None:
    plan = PROFILES["top2026"].curriculum
    assert {grade: sum(hours.values()) for grade, hours in plan.items()} == TYPICAL_TOTAL


def test_top2026_week_fits_the_grid() -> None:
    prof = PROFILES["top2026"]
    assert max(TYPICAL_TOTAL.values()) <= 5 * prof.lessons_per_day


@pytest.mark.parametrize(("classes", "seed"), [(12, 0), (24, 1), (36, 2)])
def test_top2026_class_sizes_within_law_463_art_12(classes: int, seed: int) -> None:
    problem = generate_problem(classes, 0.55, seed=seed, profile="top2026")
    for c in problem.classes:
        grade = int(c.id[1:3])
        limit = 24 if grade <= PRIMARY_GRADES else 30
        assert 5 <= c.students <= limit, c


@pytest.mark.parametrize(("classes", "seed"), [(12, 0), (24, 1), (36, 2)])
def test_top2026_passes_prechecks(classes: int, seed: int) -> None:
    problem = generate_problem(classes, 0.55, seed=seed, profile="top2026")
    assert problem.lessons_per_day == PROFILES["top2026"].lessons_per_day
    assert precheck(problem) == ()


def test_default_profile_is_legacy() -> None:
    assert generate_problem(12, 0.5, seed=3) == generate_problem(12, 0.5, seed=3, profile="legacy")
    assert generate_problem(12, 0.5, seed=3).lessons_per_day == PROFILES["legacy"].lessons_per_day


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        generate_problem(12, 0.5, profile="nope")
