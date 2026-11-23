"""Golden check of a substitution: validator-lib ``newConflicts`` on two weeks (test-only).

timetable builds the week it validates from the template, the holidays, the absences and the
overlays (contracts ``vectors/substitution``: ``given`` -> ``input``); ``week_input`` does the same,
after ``gen-substitution-vectors.mjs`` (holidays first, then the overlays in order), and is checked
against every ``given -> input`` vector before it checks the engine. ``weeks`` turns a
``solve.substitution.requested`` request into the baseline week (approved substitutions only) and
the effective week (plus the engine's proposals); the verdict is ``lad-validate --mode
new-conflicts`` (validator-lib >= 0.4.0). No timetable rule is re-implemented here.
"""

import copy
import datetime as dt
import json
import subprocess
from collections.abc import Iterable
from typing import Any, cast

from golden import cli_path, node_path

type Json = dict[str, Any]

_TIMEOUT_S = 120


def week_input(
    snapshot: Json,
    template: Json,
    *,
    holidays: Iterable[int] = (),
    absences: Iterable[Json] = (),
    overlays: Iterable[Json] = (),
) -> Json:
    """validation.input.v1 of a week; overlays are ``{day, class_id, lesson_no, kind, ...}``."""
    holidays = list(holidays)
    absences = list(absences)
    lessons: list[Json] = copy.deepcopy(template["lessons"])
    not_held: list[Json] = []
    for day in holidays:  # holidays first: they take the template lessons of the day away
        not_held.extend(_not_held(x, "holiday") for x in lessons if x["day"] == day)
        lessons = [x for x in lessons if x["day"] != day]
    for overlay in overlays:
        [i] = [
            i
            for i, x in enumerate(lessons)
            if (x["class_id"], x["day"], x["lesson_no"])
            == (overlay["class_id"], overlay["day"], overlay["lesson_no"])
        ]
        lesson = lessons[i]
        match overlay["kind"]:
            case "teacher":
                lessons[i] = lesson | {
                    "teacher_id": overlay["new_teacher_id"],
                    "origin": "substitution",
                }
            case "moved":
                target = overlay["moved_to"]
                lessons[i] = lesson | {"day": target["day"], "lesson_no": target["lesson_no"]}
            case kind:
                del lessons[i]
                not_held.append(_not_held(lesson, kind))
    snap = copy.deepcopy(snapshot)
    snap["school_days"] = [d for d in snap["school_days"] if d not in holidays]
    timetable: Json = {
        "lessons": lessons,
        "day_formats": [f for f in template["day_formats"] if f["day"] not in holidays],
    }
    if not_held:
        timetable["not_held"] = not_held
    result: Json = {"schema_version": "1.0", "snapshot": snap, "timetable": timetable}
    if absences:
        result["absences"] = absences
    if holidays:
        result["holidays"] = holidays
    return result


def _not_held(lesson: Json, kind: str) -> Json:
    keys = ("class_id", "day", "lesson_no", "subject_id")
    return {k: lesson[k] for k in keys} | {"kind": kind}


# ---------------------------------------------------------------- a substitution request


def weeks(request: Json, proposals: Iterable[Json]) -> tuple[Json, Json]:
    """(baseline, effective) of a request (``data`` of the event) and the engine's proposals.

    Dates become weekdays of the request's ISO week; non-school dates that are school days are
    the holidays; an absence of a whole date lists every lesson number; without ``absences`` the
    absent teacher is absent all of every absence date of the week. Both weeks carry the very same
    snapshot object, as newConflicts requires.
    """
    week = _week(request["iso_week"])
    snapshot = {
        "kind": request["template"]["kind"],
        "school_days": request["school_days"],
        "lessons_per_day": request["lessons_per_day"],
        "staff": request["staff"],
        "capacity": request["capacity"],
        "classes": request["classes"],
        "subjects": request["subjects"],
        "teachers": request["teachers"],
        "rows": request["rows"],
    }
    template = {
        "lessons": request["template"]["lessons"],
        "day_formats": request["template"]["day_formats"],
    }
    holidays = sorted(
        day
        for x in request.get("non_school_dates", [])
        if (day := _day(week, x)) in request["school_days"]
    )
    every = list(range(1, request["lessons_per_day"] + 1))
    rows = cast("list[Json] | None", request.get("absences"))
    if rows is None:
        rows = [
            {"teacher_id": request["absent_teacher_id"], "date": x, "lesson_nos": None}
            for x in request["absence_dates"]
        ]
    absences: list[Json] = [
        {
            "teacher_id": a["teacher_id"],
            "day": _day(week, a["date"]),
            "lesson_nos": every if a["lesson_nos"] is None else a["lesson_nos"],
        }
        for a in rows
        if a["date"] in week
    ]
    approved = [_overlay(week, a) for a in request["approved_substitutions"]]
    proposed = [_overlay(week, p) for p in proposals]
    baseline = week_input(
        snapshot, template, holidays=holidays, absences=absences, overlays=approved
    )
    effective = week_input(
        snapshot, template, holidays=holidays, absences=absences, overlays=approved + proposed
    )
    effective["snapshot"] = baseline["snapshot"]  # the same object: identical JSON
    return baseline, effective


def _week(iso_week: str) -> dict[str, int]:
    year, week = iso_week.split("-W")
    dates = (dt.date.fromisocalendar(int(year), int(week), d) for d in range(1, 8))
    return {x.isoformat(): x.isoweekday() for x in dates}


def _day(week: dict[str, int], date: str) -> int:
    return week[date]


def _overlay(week: dict[str, int], row: Json) -> Json:
    """An approved substitution or a proposal in the weekday form of ``week_input``."""
    overlay: Json = {
        "day": _day(week, row["date"]),
        "class_id": row["class_id"],
        "lesson_no": row["lesson_no"],
        "kind": row.get("kind", "teacher"),
    }
    if overlay["kind"] == "teacher":
        overlay["new_teacher_id"] = row["teacher_id"]
    if overlay["kind"] == "moved":
        target = row["moved_to"]
        overlay["moved_to"] = {"day": _day(week, target["date"]), "lesson_no": target["lesson_no"]}
    return overlay


# ---------------------------------------------------------------- the CLI


def new_conflicts(cases: dict[str, tuple[Json, Json]]) -> dict[str, Json]:
    """``lad-validate --mode new-conflicts`` on (baseline, effective) pairs, one CLI run."""
    lines = [
        json.dumps({"case_id": case_id, "baseline": b, "effective": e}, ensure_ascii=False)
        for case_id, (b, e) in cases.items()
    ]
    completed = subprocess.run(  # noqa: S603 - fixed argv, input is test data
        [node_path(), str(cli_path()), "--mode", "new-conflicts"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        check=False,
    )
    if completed.returncode != 0:
        msg = f"lad-validate exited {completed.returncode}: {completed.stderr.strip()}"
        raise AssertionError(msg)
    outcomes = [cast("Json", json.loads(line)) for line in completed.stdout.splitlines()]
    return {cast("str", o["case_id"]): o for o in outcomes}


def added(outcome: Json) -> list[Json]:
    """What the proposals add or worsen; a broken week is a failed check, not an empty list."""
    if outcome.get("status") != "CHECKED":
        msg = f"lad-validate did not check the pair: {json.dumps(outcome)[:2000]}"
        raise AssertionError(msg)
    return cast("list[Json]", outcome["added"])
