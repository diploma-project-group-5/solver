"""The golden checks really go through validator-lib: they catch broken timetables too."""

import json
from dataclasses import replace

import pytest

import golden
from builders import PRIO, klass, problem, row, teacher
from lad_solver.app.solve import solve
from lad_solver.domain.problem import Params, Passport, Problem
from lad_solver.domain.result import DayFormat, Lesson, Result

PARAMS = Params(time_limit_s=10, num_workers=1, random_seed=0)


def _solved() -> tuple[Problem, Result]:
    p = problem(
        classes=(klass("a", 20), klass("b", 10)),
        teachers=(teacher("t1"), teacher("t2")),
        rows=(row("a", PRIO, "t1", 2), row("b", PRIO, "t2", 2)),
        capacity=Passport(40),
        days=(1, 2),
        staff=3,
    )
    result = solve(p, PARAMS)
    assert golden.violations(p, result) == []
    return p, result


def _rules(lines: list[str]) -> set[str]:
    return {json.loads(line)["rule"] for line in lines}


def test_a_double_booked_class_is_reported() -> None:
    p, result = _solved()
    broken = replace(result, lessons=(*result.lessons, result.lessons[0]))
    assert "class_double_booked" in _rules(golden.violations(p, broken))


def test_an_overfull_shelter_is_reported() -> None:
    p, result = _solved()
    all_in_person = tuple(DayFormat(f.class_id, f.day, in_person=True) for f in result.day_formats)
    broken = replace(result, day_formats=all_in_person)
    tight = replace(p, capacity=Passport(p.staff + 1))
    assert "capacity_exceeded" in _rules(golden.violations(tight, broken))


def test_day_load_is_the_validator_load() -> None:
    p, result = _solved()
    in_person = golden.in_person_days(result)
    for d in p.school_days:
        load = golden.day_load(p, result, d)
        pupils = sum(c.students for c in p.classes if (c.id, d) in in_person)
        assert load.pupils == pupils
        assert load.people == pupils + p.staff + load.teachers


def test_a_lesson_of_an_unknown_class_is_reported() -> None:
    p, result = _solved()
    stray = Lesson("ghost", 1, 1, PRIO, "t1")
    broken = replace(result, lessons=(*result.lessons, stray))
    assert "class_not_active" in _rules(golden.violations(p, broken))


def test_without_the_cli_the_checks_fail_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(golden.CLI_ENV, raising=False)
    with pytest.raises(AssertionError, match=golden.CLI_ENV):
        golden.cli_path()
