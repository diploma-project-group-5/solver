"""Golden checks of an engine ``Result``: the verdict comes from the validator-lib CLI (test-only).

Timetable rules live in validator-lib only (ADR 2026-11-04 §5, Trello #73): the solver never
re-implements them in Python. ``violations`` and ``day_load`` run ``lad-validate`` in JSONL mode on
the problem (the snapshot) and the result (the timetable). ``objective`` and ``in_person_days`` are
the engine's own scoring and a data view, not validation rules.

The CLI is ``LAD_VALIDATE_CLI`` (path to validator-lib ``dist/cli/main.js``, run with ``node``);
without it the checks fail loudly instead of passing silently.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import cast

from lad_solver.cli.io import problem_to_json
from lad_solver.domain.problem import Problem
from lad_solver.domain.result import Result, SolveStatus

type Json = dict[str, object]

CLI_ENV = "LAD_VALIDATE_CLI"
_TIMEOUT_S = 120


@dataclass(frozen=True, slots=True)
class Objective:
    primary: int
    secondary: int
    value: int


@dataclass(frozen=True, slots=True)
class DayLoad:
    pupils: int  # students of in-person classes (the 1.2 seats load)
    teachers: int  # distinct teachers with an in-person lesson
    people: int  # pupils + staff + teachers (the passport load)


def in_person_days(result: Result) -> set[tuple[str, int]]:
    """(class_id, day) of every in-person class day."""
    return {(f.class_id, f.day) for f in result.day_formats if f.in_person}


def violations(problem: Problem, result: Result) -> list[str]:
    """Every conflict validator-lib reports, one line each; empty means the result is admissible."""
    if result.status not in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}:
        return [f"status {result.status} has no timetable"]
    outcome = _outcome(problem, result)
    status = outcome["status"]
    if status == "INVALID_INPUT":
        return [f"validator-lib INVALID_INPUT: {json.dumps(outcome.get('issues'))}"]
    conflicts = cast("list[Json]", outcome["conflicts"])
    lines = [json.dumps(c, sort_keys=True) for c in conflicts]
    if status != "ACCEPTED" and not lines:
        return [f"validator-lib {status} without conflicts"]
    return lines


def day_load(problem: Problem, result: Result, day: int) -> DayLoad:
    """The load of one school day as validator-lib reports it for ``/load``."""
    outcome = _outcome(problem, result)
    report = outcome.get("load")
    if not isinstance(report, dict):
        msg = f"validator-lib returned no load report ({outcome['status']})"
        raise AssertionError(msg)  # noqa: TRY004 - a failed test check, not a type error
    days = cast("list[Json]", cast("Json", report)["days"])
    (entry,) = (d for d in days if d["day"] == day)
    pupils = cast("int", entry["students_in_person"])
    teachers = cast("int", entry["teachers_on_site"])
    passport = cast("Json", cast("Json", entry["conditions"])["passport"])
    return DayLoad(pupils, teachers, cast("int", passport["load"]))


def objective(problem: Problem, result: Result) -> Objective:
    """The engine's objective recomputed from the result (spec 06: W * primary + secondary)."""
    students = {c.id: c.students for c in problem.classes}
    flagged = {c.id for c in problem.classes if c.priority}
    priority_subjects = {s.id for s in problem.subjects if s.priority_area}
    in_person = in_person_days(result)
    primary = sum(
        students[x.class_id]
        for x in result.lessons
        if x.subject_id in priority_subjects and (x.class_id, x.day) in in_person
    )
    secondary = sum(1 for c, _ in in_person if c in flagged)
    weight = len(flagged) * len(problem.school_days) + 1
    return Objective(primary, secondary, weight * primary + secondary)


def validation_input(problem: Problem, result: Result) -> Json:
    """validation.input.v1: the problem is the snapshot, lessons and day formats the timetable."""
    return {
        "schema_version": "1.0",
        "snapshot": problem_to_json(problem),
        "timetable": {
            "lessons": [
                {
                    "class_id": x.class_id,
                    "day": x.day,
                    "lesson_no": x.lesson_no,
                    "subject_id": x.subject_id,
                    "teacher_id": x.teacher_id,
                }
                for x in result.lessons
            ],
            "day_formats": [
                {"class_id": f.class_id, "day": f.day, "in_person": f.in_person}
                for f in result.day_formats
            ],
        },
    }


def _outcome(problem: Problem, result: Result) -> Json:
    line = {"case_id": "golden", "input": validation_input(problem, result)}
    return json.loads(_run(json.dumps(line, sort_keys=True, ensure_ascii=False)))


@cache
def _run(line: str) -> str:
    """One JSONL line through ``lad-validate``; cached because a test asks per day."""
    completed = subprocess.run(  # noqa: S603 - fixed argv, input is test data
        [_node(), str(cli_path())],
        input=line + "\n",
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        msg = f"lad-validate exited {completed.returncode}: {completed.stderr.strip()}"
        raise AssertionError(msg)
    (out,) = completed.stdout.splitlines()
    return out


def cli_path() -> Path:
    raw = os.environ.get(CLI_ENV)
    if not raw:
        msg = f"{CLI_ENV} is not set: point it at validator-lib dist/cli/main.js (see README)"
        raise AssertionError(msg)
    path = Path(raw)
    if not path.is_file():
        msg = f"{CLI_ENV}={raw} is not a file"
        raise AssertionError(msg)
    return path


@cache
def _node() -> str:
    node = shutil.which("node")
    if node is None:
        msg = "node is not on PATH: the validator-lib CLI needs Node 24"
        raise AssertionError(msg)
    return node
