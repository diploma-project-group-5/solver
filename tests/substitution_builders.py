"""Random ``solve.substitution.requested`` requests for the property tests (test-only).

``random_request(seed)`` builds a small school week in ISO week 2026-W47: a template without
clashes, its curriculum rows, day formats that keep a teacher in one format a day, a capacity near
the busiest day, an absent teacher with a partial or whole-day absence and a few absences of others.
A share of the weeks is stale on purpose (a class with a flipped format, a day over the passport,
a teacher over his limit): the engine must still answer, and never make them worse.
With ``approved`` an earlier job of the same week is solved first and its proposals are approved.
"""

import datetime as dt
import random
from typing import Any

type Json = dict[str, Any]

ISO_WEEK = "2026-W47"
DATES = [dt.date(2026, 11, 16) + dt.timedelta(days=i) for i in range(7)]  # Monday .. Sunday
SCHOOL_DAYS = [1, 2, 3, 4, 5]


def date_of(day: int) -> str:
    return DATES[day - 1].isoformat()


def envelope(data: Json) -> Json:
    """The event around a request's ``data``."""
    return {
        "event_id": "6f1d2c3b-4a5e-4f60-8b71-0c2d3e4f5a6d",
        "type": "solve.substitution.requested",
        "schema_version": "1.0",
        "correlation_id": "req-property",
        "school_id": "7b0f8c1e-3d2a-4c51-9a6e-1f2b3c4d5e6f",
        "data": data,
    }


def random_request(seed: int, *, kind: str = "main") -> Json:  # noqa: C901, PLR0915
    rng = random.Random(seed)  # noqa: S311 - test data
    lessons_per_day = rng.randint(4, 7)
    classes: list[Json] = [
        {"id": f"c-{i}", "students": rng.randint(5, 30), "priority": rng.random() < 0.3}
        for i in range(rng.randint(2, 5))
    ]
    alert = kind == "alert"
    subjects: list[Json] = [
        {
            "id": f"s-{i}",
            "priority_area": rng.random() < 0.4,
            "no_shelter": alert and rng.random() < 0.2,
        }
        for i in range(rng.randint(3, 6))
    ]
    teacher_ids = [f"t-{i}" for i in range(rng.randint(3, 8))]
    owner: dict[tuple[str, str], str] = {}  # (class, subject) -> teacher (one teacher per row)
    busy: set[tuple[str, int, int]] = set()  # (teacher, day, lesson_no)
    lessons: list[Json] = []
    for c in classes:
        for day in SCHOOL_DAYS:
            count = rng.randint(0, lessons_per_day - 1)
            start = rng.randint(1, lessons_per_day - count) if count else 1
            for lesson_no in range(start, start + count):
                pairs = [
                    (s["id"], t)
                    for s in subjects
                    for t in teacher_ids
                    if owner.get((c["id"], s["id"]), t) == t and (t, day, lesson_no) not in busy
                ]
                if not pairs:
                    continue
                subject_id, teacher_id = rng.choice(pairs)
                owner[c["id"], subject_id] = teacher_id
                busy.add((teacher_id, day, lesson_no))
                lessons.append(
                    {
                        "class_id": c["id"],
                        "day": day,
                        "lesson_no": lesson_no,
                        "subject_id": subject_id,
                        "teacher_id": teacher_id,
                    }
                )
    rows = [
        {
            "class_id": c,
            "subject_id": s,
            "teacher_id": t,
            "hours": sum(1 for x in lessons if (x["class_id"], x["subject_id"]) == (c, s)),
        }
        for (c, s), t in sorted(owner.items())
    ]
    formats = _formats(rng, classes, lessons)
    staff = rng.randint(0, 5)
    loads = [_people(classes, lessons, formats, day, staff) for day in SCHOOL_DAYS]
    passport = max(loads) + rng.choice([0, 0, 1, 2, 10])
    if rng.random() < 0.15:  # stale: the busiest day is already over the passport
        passport = max(staff, max(loads) - rng.randint(1, 3))
    capacity: Json = {"passport": passport}
    if alert:
        pupils = max(_pupils(classes, formats, day) for day in SCHOOL_DAYS)
        capacity = {"seats": pupils + rng.choice([0, 1, 5]), "passport": passport}
    loads_by_teacher = {t: sum(1 for x in lessons if x["teacher_id"] == t) for t in teacher_ids}
    teachers = [
        {
            "id": t,
            "weekly_limit": max(0, loads_by_teacher[t] + rng.choice([-1, 0, 0, 1, 2, 4])),
            "unavailable_days": sorted(rng.sample(SCHOOL_DAYS, k=rng.choice([0, 0, 0, 1]))),
        }
        for t in teacher_ids
    ]
    teaching = sorted({(x["teacher_id"], x["day"]) for x in lessons})
    absent_id, day = rng.choice(teaching) if teaching else (teacher_ids[0], 1)
    own_hours = sorted(
        x["lesson_no"] for x in lessons if (x["teacher_id"], x["day"]) == (absent_id, day)
    )
    whole_day = rng.random() < 0.4 or not own_hours
    absent_hours = (
        None if whole_day else sorted(rng.sample(own_hours, k=rng.randint(1, len(own_hours))))
    )
    date = date_of(day)
    to_cover = [
        {"class_id": x["class_id"], "lesson_no": x["lesson_no"]}
        for x in lessons
        if (x["teacher_id"], x["day"]) == (absent_id, day)
        and (absent_hours is None or x["lesson_no"] in absent_hours)
    ]
    absences = [{"teacher_id": absent_id, "date": date, "lesson_nos": absent_hours}]
    absence_dates = [date]
    if rng.random() < 0.3 and day < 5:
        absence_dates.append(date_of(day + 1))
        absences.append({"teacher_id": absent_id, "date": date_of(day + 1), "lesson_nos": None})
    for other in rng.sample(teacher_ids, k=min(2, len(teacher_ids))):
        if other == absent_id:
            continue
        other_day = rng.choice(SCHOOL_DAYS)
        hours = sorted(rng.sample(range(1, lessons_per_day + 1), k=rng.randint(1, 2)))
        absences.append({"teacher_id": other, "date": date_of(other_day), "lesson_nos": hours})
    request: Json = {
        "job_id": f"5d0b5e1c-0000-4000-8000-{seed % 10**12:012d}",
        "date": date,
        "absence_dates": absence_dates,
        "absent_teacher_id": absent_id,
        "lessons": to_cover,
        "not_before": date_of(rng.randint(1, day)),
        "iso_week": ISO_WEEK,
        "snapshot_version": 1,
        "time_limit_s": 30,
        "queue_deadline_at": "2026-11-16T08:10:00+02:00",
        "school_days": SCHOOL_DAYS,
        "lessons_per_day": lessons_per_day,
        "staff": staff,
        "capacity": capacity,
        "classes": classes,
        "subjects": subjects,
        "teachers": teachers,
        "rows": rows,
        "template": {
            "timetable_id": "9e1f7a20-3c4b-4d5e-8f60-7a8b9c0d1e2f",
            "kind": kind,
            "lessons": lessons,
            "day_formats": [
                {"class_id": c["id"], "day": d, "in_person": formats[c["id"], d]}
                for c in classes
                for d in SCHOOL_DAYS
            ],
        },
        "approved_substitutions": [],
        "absences": _merge(absences),
        "allow_support_moves": False,
    }
    if rng.random() < 0.7:  # else step 3 is off
        others = [x.isoformat() for x in DATES[:5] if x.isoformat() != date]
        request["non_school_dates"] = sorted(rng.sample(others, k=rng.choice([0, 0, 1, 2])))
    return request


def _formats(
    rng: random.Random, classes: list[Json], lessons: list[Json]
) -> dict[tuple[str, int], bool]:
    """One format per group of classes that share a teacher that day (a teacher has one format);
    now and then one class flips (a stale week)."""
    formats: dict[tuple[str, int], bool] = {}
    for day in SCHOOL_DAYS:
        groups = _teacher_groups(
            [c["id"] for c in classes], [x for x in lessons if x["day"] == day]
        )
        chosen: dict[str, bool] = {}
        for c in classes:
            root = groups[c["id"]]
            chosen.setdefault(root, rng.random() < 0.6)
            has_lessons = any(x["day"] == day and x["class_id"] == c["id"] for x in lessons)
            formats[c["id"], day] = chosen[root] and has_lessons
    busy = sorted({(x["class_id"], x["day"]) for x in lessons})
    if busy and rng.random() < 0.1:
        key = rng.choice(busy)
        formats[key] = not formats[key]
    return formats


def _teacher_groups(class_ids: list[str], day_lessons: list[Json]) -> dict[str, str]:
    """Class -> a representative of the classes linked to it by shared teachers that day."""
    parent = {c: c for c in class_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x

    by_teacher: dict[str, list[str]] = {}
    for x in day_lessons:
        by_teacher.setdefault(x["teacher_id"], []).append(x["class_id"])
    for group in by_teacher.values():
        for other in group[1:]:
            parent[find(other)] = find(group[0])
    return {c: find(c) for c in class_ids}


def _pupils(classes: list[Json], formats: dict[tuple[str, int], bool], day: int) -> int:
    return sum(c["students"] for c in classes if formats[c["id"], day])


def _people(
    classes: list[Json],
    lessons: list[Json],
    formats: dict[tuple[str, int], bool],
    day: int,
    staff: int,
) -> int:
    on_site = {x["teacher_id"] for x in lessons if x["day"] == day and formats[x["class_id"], day]}
    return _pupils(classes, formats, day) + staff + len(on_site)


def _merge(absences: list[Json]) -> list[Json]:
    """One row per (teacher, date), as the contract requires."""
    merged: dict[tuple[str, str], Json] = {}
    for a in absences:
        key = (a["teacher_id"], a["date"])
        if key not in merged:
            merged[key] = dict(a)
            continue
        old, new = merged[key]["lesson_nos"], a["lesson_nos"]
        merged[key]["lesson_nos"] = None if old is None or new is None else sorted({*old, *new})
    return list(merged.values())
