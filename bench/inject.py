"""Fault injection into accepted timetables: does validator-lib catch each kind of violation?

Input: ``validate_in.jsonl`` of a bench run (validation.input.v1 lines). Only cases validator-lib
ACCEPTED are used as bases. Every operator breaks exactly one rule of the validator vocabulary
(sometimes others follow, e.g. a deleted lesson also breaks the plan hours); a mutant counts as
*detected* when the validator's conflicts contain the operator's target rule. Mutants rejected at
the boundary (INVALID_INPUT) are counted apart. Deterministic per ``--seed``.

    python3 -I bench/inject.py --run ../../goit/AppliedCSProject/experiments/<run-id> \
        --per-op 5 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

type Json = dict[str, object]
type Op = Callable[[Json, random.Random], Json | None]

GHOST = "ghost-0000"


def _tt(inp: Json) -> tuple[list[Json], list[Json]]:
    tt = inp["timetable"]
    return tt["lessons"], tt["day_formats"]  # type: ignore[index,return-value]


def _snap(inp: Json) -> Json:
    return inp["snapshot"]  # type: ignore[return-value]


def _formats(inp: Json) -> dict[tuple[str, int], Json]:
    return {(f["class_id"], f["day"]): f for f in _tt(inp)[1]}  # type: ignore[misc]


def _by(lessons: list[Json], *keys: str) -> dict[tuple, list[Json]]:
    out: dict[tuple, list[Json]] = defaultdict(list)
    for x in lessons:
        out[tuple(x[k] for k in keys)].append(x)
    return out


def _day_load(inp: Json, day: int) -> tuple[int, int]:
    """(pupils in person, pupils + staff + teachers on site) of one day."""
    s = _snap(inp)
    students = {c["id"]: c["students"] for c in s["classes"]}  # type: ignore[index]
    fm = _formats(inp)
    lessons, _ = _tt(inp)
    on = {c for (c, d), f in fm.items() if d == day and f["in_person"]}
    pupils = sum(students[c] for c in on)
    teachers = {x["teacher_id"] for x in lessons if x["day"] == day and x["class_id"] in on}
    return pupils, pupils + int(s["staff"]) + len(teachers)  # type: ignore[arg-type]


# ---------------------------------------------------------------- operators (one target rule each)


def op_capacity(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    s = _snap(m)
    cap = s["capacity"]
    lessons, _ = _tt(m)
    has = {(x["class_id"], x["day"]) for x in lessons}
    fm = _formats(m)
    days = list(s["school_days"])  # type: ignore[call-overload]
    rnd.shuffle(days)
    for day in days:
        remote = [k for k, f in fm.items() if k[1] == day and not f["in_person"] and k in has]
        rnd.shuffle(remote)
        for k in remote:
            fm[k]["in_person"] = True
            pupils, people = _day_load(m, day)
            over_seats = "seats" in cap and pupils > cap["seats"]  # type: ignore[operator,index]
            if people > cap["passport"] or over_seats:  # type: ignore[index]
                return m
        for k in remote:
            fm[k]["in_person"] = False
    return None


def op_class_double(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    groups = [g for g in _by(_tt(m)[0], "class_id", "day").values() if len(g) >= 2]  # noqa: PLR2004
    if not groups:
        return None
    a, b = rnd.sample(rnd.choice(groups), 2)
    b["lesson_no"] = a["lesson_no"]
    return m


def op_teacher_double(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    groups = [
        g for g in _by(_tt(m)[0], "teacher_id", "day").values()
        if len({x["class_id"] for x in g}) >= 2  # noqa: PLR2004
    ]
    if not groups:
        return None
    g = rnd.choice(groups)
    a = rnd.choice(g)
    b = rnd.choice([x for x in g if x["class_id"] != a["class_id"]])
    b["lesson_no"] = a["lesson_no"]
    return m


def op_teacher_unavailable(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    x = rnd.choice(_tt(m)[0])
    t = next(t for t in _snap(m)["teachers"] if t["id"] == x["teacher_id"])  # type: ignore[union-attr]
    t["unavailable_days"] = sorted({*t["unavailable_days"], x["day"]})
    return m


def op_weekly_limit(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    load = Counter(x["teacher_id"] for x in _tt(m)[0])
    busy = [t for t, n in load.items() if n >= 2]  # noqa: PLR2004
    if not busy:
        return None
    tid = rnd.choice(busy)
    t = next(t for t in _snap(m)["teachers"] if t["id"] == tid)  # type: ignore[union-attr]
    t["weekly_limit"] = load[tid] - 1
    return m


def op_lesson_out_of_range(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    rnd.choice(_tt(m)[0])["lesson_no"] = int(_snap(m)["lessons_per_day"]) + 1  # type: ignore[arg-type]
    return m


def op_day_not_school(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    free = [d for d in range(1, 7) if d not in _snap(m)["school_days"]]  # type: ignore[operator]
    if not free:
        return None
    rnd.choice(_tt(m)[0])["day"] = rnd.choice(free)
    return m


def op_format_missing(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    fms = _tt(m)[1]
    fms.pop(rnd.randrange(len(fms)))
    return m


def op_format_duplicate(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    fms = _tt(m)[1]
    fms.append(copy.deepcopy(rnd.choice(fms)))
    return m


def op_in_person_without_lessons(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    has = {(x["class_id"], x["day"]) for x in _tt(m)[0]}
    empty = [f for k, f in _formats(m).items() if k not in has and not f["in_person"]]
    if not empty:
        return None
    rnd.choice(empty)["in_person"] = True
    return m


def op_teacher_mixed(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    fm = _formats(m)
    groups = [
        g for g in _by(_tt(m)[0], "teacher_id", "day").values()
        if len({x["class_id"] for x in g}) >= 2  # noqa: PLR2004
    ]
    if not groups:
        return None
    g = rnd.choice(groups)
    x = rnd.choice(g)
    f = fm[x["class_id"], x["day"]]
    f["in_person"] = not f["in_person"]
    return m


def op_lesson_not_in_plan(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    s = _snap(m)
    x = rnd.choice(_tt(m)[0])
    planned = {r["subject_id"] for r in s["rows"] if r["class_id"] == x["class_id"]}  # type: ignore[union-attr]
    other = [sub["id"] for sub in s["subjects"] if sub["id"] not in planned]  # type: ignore[union-attr]
    if not other:
        return None
    x["subject_id"] = rnd.choice(other)
    return m


def op_teacher_not_in_plan(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    s = _snap(m)
    x = rnd.choice(_tt(m)[0])
    row_teachers = {
        r["teacher_id"] for r in s["rows"]  # type: ignore[union-attr]
        if r["class_id"] == x["class_id"] and r["subject_id"] == x["subject_id"]
    }
    other = [t["id"] for t in s["teachers"] if t["id"] not in row_teachers]  # type: ignore[union-attr]
    if not other:
        return None
    x["teacher_id"] = rnd.choice(other)
    return m


def op_plan_hours(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    lessons = _tt(m)[0]
    lessons.pop(rnd.randrange(len(lessons)))
    return m


def _ghost(field: str) -> Op:
    def op(inp: Json, rnd: random.Random) -> Json | None:
        m = copy.deepcopy(inp)
        rnd.choice(_tt(m)[0])[field] = GHOST
        return m

    return op


def op_no_shelter(inp: Json, rnd: random.Random) -> Json | None:
    m = copy.deepcopy(inp)
    s = _snap(m)
    if s["kind"] != "alert":
        return None
    outside = {sub["id"] for sub in s["subjects"] if sub["no_shelter"]}  # type: ignore[union-attr]
    fm = _formats(m)
    cand = [x for x in _tt(m)[0] if x["subject_id"] in outside
            and not fm[x["class_id"], x["day"]]["in_person"]]
    if not cand:
        return None
    x = rnd.choice(cand)
    fm[x["class_id"], x["day"]]["in_person"] = True
    return m


OPS: dict[str, Op] = {
    "capacity_exceeded": op_capacity,
    "class_double_booked": op_class_double,
    "teacher_double_booked": op_teacher_double,
    "teacher_unavailable_day": op_teacher_unavailable,
    "teacher_weekly_limit": op_weekly_limit,
    "lesson_no_out_of_range": op_lesson_out_of_range,
    "day_not_school_day": op_day_not_school,
    "day_format_missing": op_format_missing,
    "day_format_duplicate": op_format_duplicate,
    "in_person_without_lessons": op_in_person_without_lessons,
    "teacher_mixed_format": op_teacher_mixed,
    "lesson_not_in_plan": op_lesson_not_in_plan,
    "teacher_not_in_plan": op_teacher_not_in_plan,
    "plan_hours_mismatch": op_plan_hours,
    "class_not_active": _ghost("class_id"),
    "subject_not_active": _ghost("subject_id"),
    "teacher_not_active": _ghost("teacher_id"),
    "no_shelter_in_person": op_no_shelter,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--run", type=Path, required=True, help="bench run directory")
    ap.add_argument("--per-op", type=int, default=5, help="mutants per operator and base case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bases", type=int, default=0, help="0 = every accepted case")
    ap.add_argument("--validator", type=Path,
                    default=Path(__file__).resolve().parents[2] / "validator-lib" / "dist" / "cli"
                    / "main.js")
    args = ap.parse_args()

    verdict = {}
    for raw in (args.run / "validate_out.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(raw)
        verdict[rec["case_id"]] = rec["status"]
    bases = [
        json.loads(raw)
        for raw in (args.run / "validate_in.jsonl").read_text(encoding="utf-8").splitlines()
        if raw and verdict.get(json.loads(raw)["case_id"]) == "ACCEPTED"
    ]
    if args.max_bases:
        bases = bases[: args.max_bases]
    rnd = random.Random(args.seed)
    lines, meta = [], {}
    skipped: Counter[str] = Counter()
    for base in bases:
        for rule, op in OPS.items():
            for k in range(args.per_op):
                mutant = op(base["input"], rnd)
                if mutant is None:
                    skipped[rule] += 1
                    continue
                cid = f"{base['case_id']}#{rule}#{k}"
                meta[cid] = rule
                lines.append(json.dumps({"case_id": cid, "input": mutant}, ensure_ascii=False))

    out_dir = args.run / f"inject-s{args.seed}"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "mutants_in.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    done = subprocess.run(  # noqa: S603 - fixed argv
        ["node", str(args.validator)], input="\n".join(lines) + "\n", capture_output=True,
        text=True, check=False, timeout=3600,
    )
    (out_dir / "mutants_out.jsonl").write_text(done.stdout, encoding="utf-8")

    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for raw in done.stdout.splitlines():
        rec = json.loads(raw)
        rule = meta[rec["case_id"]]
        rules = {c.get("rule") for c in rec.get("conflicts") or []}
        if rec["status"] == "INVALID_INPUT":
            outcome = "invalid_input"
        elif rule in rules:
            outcome = "detected"
        elif rec["status"] == "ACCEPTED":
            outcome = "missed_accepted"
        else:
            outcome = "other_rule"
        stats[rule][outcome] += 1
        stats[rule]["mutants"] += 1

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "mutants", "detected", "detection_rate", "invalid_input",
                    "other_rule", "missed_accepted", "not_applicable"])
        tot = Counter()
        for rule in OPS:
            s = stats[rule]
            rate = round(s["detected"] / s["mutants"], 4) if s["mutants"] else None
            w.writerow([rule, s["mutants"], s["detected"], rate, s["invalid_input"],
                        s["other_rule"], s["missed_accepted"], skipped[rule]])
            tot.update(s)
        rate = round(tot["detected"] / tot["mutants"], 4) if tot["mutants"] else None
        w.writerow(["TOTAL", tot["mutants"], tot["detected"], rate, tot["invalid_input"],
                    tot["other_rule"], tot["missed_accepted"], sum(skipped.values())])
    print(f"bases={len(bases)} mutants={len(lines)} -> {out_dir / 'summary.csv'}")
    print((out_dir / "summary.csv").read_text(encoding="utf-8"))
    return 0 if done.returncode == 0 else done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
