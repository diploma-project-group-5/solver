"""JSON (de)serialisation of engine types for the CLI and bench.

This is the CLI/bench file format, NOT the Kafka contract (that lives in ``contracts`` and is
mapped by the Kafka adapter). Parsing raises ``TypeError`` for a wrong JSON type and
``ValueError`` for a bad enum value.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from lad_solver.domain.problem import (
    AlertLimits,
    Capacity,
    CurriculumRow,
    Passport,
    Problem,
    SchoolClass,
    Subject,
    Teacher,
    TimetableKind,
)
from lad_solver.domain.result import (
    CapacityCondition,
    Conflict,
    ConflictRule,
    DayFormat,
    Lesson,
    Result,
    SolveStatus,
    Stats,
    StopReason,
)

type Json = dict[str, object]


# ------------------------------------------------------------------ writing


def problem_to_json(problem: Problem) -> Json:
    capacity: Json = (
        {"seats": problem.capacity.seats, "passport": problem.capacity.passport}
        if isinstance(problem.capacity, AlertLimits)
        else {"passport": problem.capacity.passport}
    )
    return {
        "kind": problem.kind.value,
        "school_days": list(problem.school_days),
        "lessons_per_day": problem.lessons_per_day,
        "staff": problem.staff,
        "capacity": capacity,
        "classes": [asdict(c) for c in problem.classes],
        "subjects": [asdict(s) for s in problem.subjects],
        "teachers": [
            {
                "id": t.id,
                "weekly_limit": t.weekly_limit,
                "unavailable_days": sorted(t.unavailable_days),
            }
            for t in problem.teachers
        ],
        "rows": [asdict(r) for r in problem.rows],
    }


def result_to_json(result: Result) -> Json:
    return {
        "status": result.status.value,
        "lessons": [asdict(x) for x in result.lessons],
        "day_formats": [asdict(x) for x in result.day_formats],
        "conflicts": [
            {
                k: v
                for k, v in asdict(c).items()
                if v is not None and not (k == "hint" and v is False)
            }
            for c in result.conflicts
        ],
        "stats": asdict(result.stats),
    }


def write_json(path: Path, data: Json) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def read_json(path: Path) -> Json:
    return _obj(json.loads(path.read_text(encoding="utf-8")), str(path))


# ------------------------------------------------------------------ reading


def problem_from_json(data: Json) -> Problem:
    return Problem(
        kind=TimetableKind(_str(data, "kind")),
        school_days=tuple(_int_item(d, "school_days") for d in _list(data, "school_days")),
        lessons_per_day=_int(data, "lessons_per_day"),
        staff=_int(data, "staff"),
        capacity=_capacity(_obj(data.get("capacity"), "capacity")),
        classes=tuple(
            SchoolClass(_str(o, "id"), _int(o, "students"), _bool(o, "priority"))
            for o in _objects(data, "classes")
        ),
        subjects=tuple(
            Subject(_str(o, "id"), _bool(o, "priority_area"), _bool(o, "no_shelter"))
            for o in _objects(data, "subjects")
        ),
        teachers=tuple(
            Teacher(
                _str(o, "id"),
                _int(o, "weekly_limit"),
                frozenset(_int_item(d, "unavailable_days") for d in _list(o, "unavailable_days")),
            )
            for o in _objects(data, "teachers")
        ),
        rows=tuple(
            CurriculumRow(
                _str(o, "class_id"), _str(o, "subject_id"), _str(o, "teacher_id"), _int(o, "hours")
            )
            for o in _objects(data, "rows")
        ),
    )


def result_from_json(data: Json) -> Result:
    return Result(
        status=SolveStatus(_str(data, "status")),
        lessons=tuple(
            Lesson(
                _str(o, "class_id"),
                _int(o, "day"),
                _int(o, "lesson_no"),
                _str(o, "subject_id"),
                _str(o, "teacher_id"),
            )
            for o in _objects(data, "lessons")
        ),
        day_formats=tuple(
            DayFormat(_str(o, "class_id"), _int(o, "day"), _bool(o, "in_person"))
            for o in _objects(data, "day_formats")
        ),
        conflicts=tuple(_conflict(o) for o in _objects(data, "conflicts")),
        stats=_stats(_obj(data.get("stats"), "stats")),
    )


def _capacity(o: Json) -> Capacity:
    if "seats" in o:
        return AlertLimits(seats=_int(o, "seats"), passport=_int(o, "passport"))
    return Passport(_int(o, "passport"))


def _conflict(o: Json) -> Conflict:
    condition = _opt_str(o, "condition")
    return Conflict(
        rule=ConflictRule(_str(o, "rule")),
        class_id=_opt_str(o, "class_id"),
        teacher_id=_opt_str(o, "teacher_id"),
        subject_id=_opt_str(o, "subject_id"),
        day=_opt_int(o, "day"),
        condition=None if condition is None else CapacityCondition(condition),
        hint=bool(o.get("hint", False)),
    )


def _stats(o: Json) -> Stats:
    best_bound = o.get("best_bound")
    first = o.get("first_solution_s")
    return Stats(
        objective=_opt_int(o, "objective"),
        best_bound=None if best_bound is None else _float(best_bound, "best_bound"),
        primary=_opt_int(o, "primary"),
        secondary=_opt_int(o, "secondary"),
        wall_time_s=_float(o.get("wall_time_s"), "wall_time_s"),
        first_solution_s=None if first is None else _float(first, "first_solution_s"),
        phase_a_status=_opt_str(o, "phase_a_status"),
        quality_relaxed=_bool(o, "quality_relaxed"),
        stop_reason=StopReason(_str(o, "stop_reason")),
        num_workers=_int(o, "num_workers"),
        random_seed=_int(o, "random_seed"),
    )


# ------------------------------------------------------------------ typed accessors


def _obj(value: object, where: str) -> Json:
    if not isinstance(value, dict):
        msg = f"{where}: expected an object"
        raise TypeError(msg)
    return cast("Json", value)


def _list(o: Json, key: str) -> list[object]:
    value = o.get(key)
    if not isinstance(value, list):
        msg = f"{key}: expected a list"
        raise TypeError(msg)
    return cast("list[object]", value)


def _objects(o: Json, key: str) -> list[Json]:
    return [_obj(v, key) for v in _list(o, key)]


def _str(o: Json, key: str) -> str:
    value = o.get(key)
    if not isinstance(value, str):
        msg = f"{key}: expected a string"
        raise TypeError(msg)
    return value


def _opt_str(o: Json, key: str) -> str | None:
    return None if o.get(key) is None else _str(o, key)


def _int_item(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{where}: expected an integer"
        raise TypeError(msg)
    return value


def _int(o: Json, key: str) -> int:
    return _int_item(o.get(key), key)


def _opt_int(o: Json, key: str) -> int | None:
    return None if o.get(key) is None else _int(o, key)


def _float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{where}: expected a number"
        raise TypeError(msg)
    return float(value)


def _bool(o: Json, key: str) -> bool:
    value = o.get(key)
    if not isinstance(value, bool):
        msg = f"{key}: expected a boolean"
        raise TypeError(msg)
    return value
