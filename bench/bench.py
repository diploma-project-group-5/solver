"""Thesis bench: the joint model vs. two baselines on realistic and synthetic schools.

Not shipped with the engine (lives outside ``src``). Every variant gets the same wall-clock budget
``T`` and the same phase B, and every timetable it returns goes through validator-lib in one
JSONL batch, so the comparison is on admissible timetables only.

Variants
--------
``joint``       the engine as shipped: ``lad_solver.app.solve.solve`` (days, formats, capacity and
                objective in one phase A model, coverage step, phase B).
``rotation``    baseline "rotation of parallels": classes of one grade come in person together;
                parallels are taken in a fixed cyclic order and each day is filled greedily while
                pupils + teachers of the parallels fit the shelter (teachers counted per parallel,
                as a deputy head would estimate them). The phase A model then optimises the
                timetable with ``f[c,d] = 0`` outside that rotation (it may drop more classes).
``sequential``  baseline "timetable first, formats second": stage 1 solves phase A with the shelter
                ignored (a usual timetable), stage 2 fixes those lessons per day and only chooses
                the in-person days under the real capacity.

Usage
-----
    uv run python bench/bench.py --out ../../goit/AppliedCSProject/experiments \
        --time-limit 60 --workers 4 --jobs 3 --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "tests", ROOT / "tests" / "realistic"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import golden  # noqa: E402 - tests helper: validation.input.v1 builder
from lad_solver import __version__  # noqa: E402
from lad_solver.app.solve import (  # noqa: E402
    PHASE_B_DAY_MAX_S,
    PROBE_SHARE,
    _day_formats,
    _phase_b_reserve,
    _run_phase_b,
)
from lad_solver.app.solve import solve as solve_joint  # noqa: E402
from lad_solver.cli.generate import generate_problem  # noqa: E402
from lad_solver.domain.prechecks import precheck  # noqa: E402
from lad_solver.domain.problem import AlertLimits, Params, Passport, Problem  # noqa: E402
from lad_solver.domain.result import Result, SolveStatus  # noqa: E402
from lad_solver.model.cpsat import SolverConfig, run  # noqa: E402
from lad_solver.model.index import Index, build_index  # noqa: E402
from lad_solver.model.phase_a import (  # noqa: E402
    DayPlan,
    PhaseAModel,
    _plan,
    build_phase_a,
    score,
    zero_day_classes,
)

FOUND = frozenset({SolveStatus.OPTIMAL, SolveStatus.FEASIBLE})
BIG = 10**6
STAGE1_SHARE = 0.25  # sequential: share of the phase A budget for the shelter-blind timetable

type Builder = Callable[[object], Problem]

# name -> (test module, builder); the realistic schools of tests/realistic (feasible ones only).
REALISTIC: dict[str, tuple[str, Builder]] = {
    "tiny_rural": ("test_tiny_rural_gymnasium", lambda m: m.build_school()),
    "frontline_tight": ("test_frontline_gymnasium_tight_shelter", lambda m: m.gymnasium()),
    "primary_two_parallels": ("test_primary_school_two_parallels", lambda m: m.build_problem()),
    "town_lyceum": ("test_town_lyceum_comfortable_shelter", lambda m: m.build_lyceum()),
    # the test module's ``lyceum`` is a pytest fixture: use the shared builder directly
    "large_city_lyceum": ("_lyceum", lambda m: m.build_lyceum()),
    "senior_profile": ("test_senior_profile_full_grid", lambda m: m.build(m.PLAN_A)),
    "six_day_support": ("test_six_day_week_support_school", lambda m: m.build_problem()),
    "part_time_teachers": ("test_part_time_teachers_unavailable_days", lambda m: m.build_school()),
    "demo_school": ("test_demo_school_poor_priority_list", lambda m: m.demo_school()),
    "large_lyceum_alert": ("test_large_lyceum_alert_tight_seats", lambda m: m.lyceum()),
    "poor_priority_lyceum": ("test_large_lyceum_poor_priority_list", lambda m: m.poor_lyceum()),
    "alert_seats_binding": ("test_alert_day_seats_vs_passport", lambda m: m.build_problem("a")),
    "alert_passport_binding": ("test_alert_day_seats_vs_passport", lambda m: m.build_problem("b")),
}

_SYNTH = re.compile(r"^(gen|top)_c(\d+)_r([0-9.]+)_s(\d+)$")  # gen = legacy, top = top2026
_PROFILE = {"gen": "legacy", "top": "top2026"}


def load_scenario(name: str) -> Problem:
    if m := _SYNTH.match(name):
        return generate_problem(int(m[2]), float(m[3]), seed=int(m[4]), profile=_PROFILE[m[1]])
    module, build = REALISTIC[name]
    return build(importlib.import_module(module))


# ---------------------------------------------------------------- metrics


@dataclass(frozen=True, slots=True)
class Outcome:
    status: SolveStatus
    result: Result
    plan: DayPlan | None
    index: Index
    wall_s: float
    best_bound: float | None = None
    first_solution_s: float | None = None
    note: str = ""


def primary_max(index: Index) -> int:
    """Primary term if every priority lesson were in person (an upper bound, ignores capacity)."""
    return sum(
        index.students[index.row_class[r]] * index.row_hours[r]
        for r in range(len(index.row_hours))
        if index.row_priority[r] and not index.row_outside[r]
    )


def row_of(task: Task, problem: Problem, out: Outcome) -> dict[str, object]:
    index = out.index
    pmax = primary_max(index)
    class_days = sum(1 for c in range(len(problem.classes)) for _ in problem.school_days)
    row: dict[str, object] = {
        "case_id": task.case_id,
        "scenario": task.scenario,
        "source": "synthetic" if _SYNTH.match(task.scenario) else "realistic",
        "kind": problem.kind.value,
        "classes": len(problem.classes),
        "students": sum(c.students for c in problem.classes),
        "teachers": len(problem.teachers),
        "days": len(problem.school_days),
        "capacity": _capacity_text(problem),
        "variant": task.variant,
        "seed": task.seed,
        "time_limit_s": task.time_limit_s,
        "workers": task.workers,
        "status": out.status.value,
        "wall_s": round(out.wall_s, 3),
        "first_solution_s": _round(out.first_solution_s),
        "primary_max": pmax,
        "note": out.note,
    }
    if out.plan is None:
        return row
    sc = score(index, out.plan)
    in_person = len(out.plan.in_person)
    row |= {
        "primary": sc.primary,
        "primary_share": round(sc.primary / pmax, 4) if pmax else None,
        "secondary": sc.secondary,
        "objective": sc.objective,
        "best_bound": _round(out.best_bound),
        "in_person_class_days": in_person,
        "class_days": class_days,
        "in_person_share": round(in_person / class_days, 4) if class_days else None,
        "zero_day_classes": len(zero_day_classes(index, out.plan)),
    }
    return row


def _capacity_text(problem: Problem) -> str:
    cap = problem.capacity
    if isinstance(cap, AlertLimits):
        return f"seats={cap.seats};passport={cap.passport};staff={problem.staff}"
    return f"passport={cap.passport};staff={problem.staff}"


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


# ---------------------------------------------------------------- variants


def run_joint(problem: Problem, params: Params) -> Outcome:
    started = time.monotonic()
    res = solve_joint(problem, params)
    wall = time.monotonic() - started
    index = build_index(problem)
    plan = None
    if res.status in FOUND:
        loads: dict[tuple[int, int], int] = {}
        rows = {(r.class_id, r.subject_id, r.teacher_id): i for i, r in enumerate(problem.rows)}
        for x in res.lessons:
            key = (rows[x.class_id, x.subject_id, x.teacher_id], x.day)
            loads[key] = loads.get(key, 0) + 1
        pos = {c.id: i for i, c in enumerate(problem.classes)}
        in_person = frozenset((pos[f.class_id], f.day) for f in res.day_formats if f.in_person)
        plan = DayPlan(loads, in_person)
    s = res.stats
    return Outcome(res.status, res, plan, index, wall, s.best_bound, s.first_solution_s)


def _phase_a_budget(params: Params) -> tuple[float, float]:
    """(phase A seconds, phase B day limit) exactly as the engine splits ``T``."""
    reserve_b = _phase_b_reserve(params)
    return params.time_limit_s - reserve_b, reserve_b


def _solve_restricted(
    index: Index,
    params: Params,
    seconds: float,
    restrict: Callable[[PhaseAModel, tuple[int, ...]], None],
) -> tuple[SolveStatus, DayPlan | None, float | None, str]:
    """Phase A with extra constraints, on the engine's quality ladder.

    Probe with strict row caps for ``PROBE_SHARE`` of the budget; a found plan is the hint of a
    strict follow-up for the rest (the better one is kept); no plan -> relaxed caps for the rest.
    """
    config = SolverConfig(
        time_limit_s=seconds, num_workers=params.num_workers, random_seed=params.random_seed
    )
    end = time.monotonic() + seconds

    def solve_once(
        caps: tuple[int, ...], hint: DayPlan | None
    ) -> tuple[SolveStatus, DayPlan | None, float | None]:
        built = build_phase_a(index, caps, hint)
        restrict(built, caps)
        left = max(end - time.monotonic(), 0.05)
        budget = min(left, seconds * PROBE_SHARE) if hint is None and caps is strict else left
        solver, info = run(built.model, config.with_time(budget))
        plan = _plan(built, solver) if info.status in FOUND else None
        return info.status, plan, info.best_bound

    strict = index.row_caps(params.row_day_cap_min, relaxed=False)
    status, plan, bound = solve_once(strict, None)
    if plan is not None:
        if status is SolveStatus.OPTIMAL or end - time.monotonic() < 0.1:  # noqa: PLR2004
            return status, plan, bound, ""
        s2, plan2, bound2 = solve_once(strict, plan)
        if plan2 is not None and score(index, plan2).objective >= score(index, plan).objective:
            return s2, plan2, bound2, ""
        return SolveStatus.FEASIBLE, plan, bound2 if bound2 is not None else bound, ""
    relaxed = index.row_caps(params.row_day_cap_min, relaxed=True)
    status, plan, bound = solve_once(relaxed, None)
    return status, plan, bound, "relaxed_caps" if plan is not None else ""


def _finish(
    index: Index, params: Params, started: float, status: SolveStatus, plan: DayPlan | None,
    bound: float | None, note: str,
) -> Outcome:
    """Phase B on the baseline's plan, with the engine's own reserve and day limit."""
    if plan is None:
        return Outcome(status, Result(status), None, index, time.monotonic() - started, bound,
                       note=note)
    _, reserve_b = _phase_a_budget(params)
    day_limit = min(PHASE_B_DAY_MAX_S, reserve_b / len(index.days))
    end = time.monotonic() + reserve_b
    b = _run_phase_b(index, plan, params, None, day_limit, end=end)
    if isinstance(b.outcome, SolveStatus):
        return Outcome(b.outcome, Result(b.outcome), None, index, time.monotonic() - started,
                       bound, note=f"{note};phase_b_failed".strip(";"))
    res = Result(status, b.outcome, _day_formats(index, plan))
    return Outcome(status, res, plan, index, time.monotonic() - started, bound, note=note)


def rotation_allowed(problem: Problem, index: Index) -> set[tuple[int, int]]:
    """(class, day) pairs the rotation lets come in person; see the module docstring."""
    cap = problem.capacity
    if isinstance(cap, AlertLimits):
        room_people, room_pupils = cap.passport - problem.staff, cap.seats
    else:
        room_people, room_pupils = cap.passport - problem.staff, BIG

    grades: dict[str, list[int]] = {}
    for pos, c in enumerate(problem.classes):
        m = re.search(r"\d+", c.id)
        grades.setdefault(m[0] if m else f"#{pos}", []).append(pos)

    def need(classes: list[int]) -> tuple[int, int]:
        teachers = {index.row_teacher[r] for c in classes for r in index.rows_by_class[c]}
        pupils = sum(index.students[c] for c in classes)
        return pupils + len(teachers), pupils

    groups: list[list[int]] = []
    for classes in grades.values():  # a parallel that never fits rotates class by class
        people, pupils = need(classes)
        if people > room_people or pupils > room_pupils:
            groups.extend([c] for c in classes)
        else:
            groups.append(classes)

    allowed: set[tuple[int, int]] = set()
    ptr = 0
    for d in index.days:
        load_people = load_pupils = taken = 0
        while taken < len(groups):
            g = groups[ptr % len(groups)]
            people, pupils = need(g)
            if load_people + people > room_people or load_pupils + pupils > room_pupils:
                break
            allowed.update((c, d) for c in g)
            load_people, load_pupils = load_people + people, load_pupils + pupils
            ptr += 1
            taken += 1
    return allowed


def run_rotation(problem: Problem, params: Params) -> Outcome:
    started = time.monotonic()
    index = build_index(problem)
    allowed = rotation_allowed(problem, index)

    def restrict(m: PhaseAModel, _caps: tuple[int, ...]) -> None:
        for key, f in m.f.items():
            if key not in allowed:
                m.model.add(f == 0)

    seconds, _ = _phase_a_budget(params)
    status, plan, bound, note = _solve_restricted(index, params, seconds, restrict)
    return _finish(index, params, started, status, plan, bound, note)


def run_sequential(problem: Problem, params: Params) -> Outcome:
    started = time.monotonic()
    index = build_index(problem)
    seconds, _ = _phase_a_budget(params)
    blind_cap = (
        AlertLimits(BIG, BIG) if isinstance(problem.capacity, AlertLimits) else Passport(BIG)
    )
    blind = build_index(replace(problem, capacity=blind_cap))
    s1, plan1, _, note1 = _solve_restricted(
        blind, params, seconds * STAGE1_SHARE, lambda _m, _c: None
    )
    if plan1 is None:
        return _finish(index, params, started, s1, None, None, f"stage1_{s1.value}")
    loads = plan1.loads

    def restrict(m: PhaseAModel, _caps: tuple[int, ...]) -> None:
        for key, y in m.y.items():
            m.model.add(y == loads.get(key, 0))

    left = max(seconds - (time.monotonic() - started), 1.0)
    status, plan, bound, note = _solve_restricted(index, params, left, restrict)
    return _finish(index, params, started, status, plan, bound, ";".join(x for x in (note1, note) if x))


VARIANTS: dict[str, Callable[[Problem, Params], Outcome]] = {
    "joint": run_joint,
    "rotation": run_rotation,
    "sequential": run_sequential,
}


# ---------------------------------------------------------------- driver


@dataclass(frozen=True, slots=True)
class Task:
    scenario: str
    variant: str
    seed: int
    time_limit_s: float
    workers: int

    @property
    def case_id(self) -> str:
        return f"{self.scenario}|{self.variant}|s{self.seed}"


def execute(task: Task) -> tuple[dict[str, object], str | None]:
    problem = load_scenario(task.scenario)
    if precheck(problem):
        return {"case_id": task.case_id, "scenario": task.scenario, "variant": task.variant,
                "seed": task.seed, "status": "PRECHECK"}, None
    params = Params(time_limit_s=task.time_limit_s, num_workers=task.workers,
                    random_seed=task.seed)
    out = VARIANTS[task.variant](problem, params)
    line = None
    if out.status in FOUND:
        payload = {"case_id": task.case_id, "input": golden.validation_input(problem, out.result)}
        line = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return row_of(task, problem, out), line


def validate(lines: list[str], cli: Path, out_dir: Path) -> dict[str, tuple[str, int]]:
    (out_dir / "validate_in.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    done = subprocess.run(  # noqa: S603 - fixed argv
        ["node", str(cli)], input="\n".join(lines) + "\n", capture_output=True, text=True,
        check=False, timeout=3600,
    )
    (out_dir / "validate_out.jsonl").write_text(done.stdout, encoding="utf-8")
    if done.returncode != 0:
        (out_dir / "validate_err.txt").write_text(done.stderr, encoding="utf-8")
    verdicts: dict[str, tuple[str, int]] = {}
    for raw in done.stdout.splitlines():
        rec = json.loads(raw)
        verdicts[rec["case_id"]] = (rec["status"], len(rec.get("conflicts") or []))
    return verdicts


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=False).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True, help="parent directory for the run")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--scenarios", default="realistic,synthetic",
                    help="'realistic', 'synthetic' (legacy generator), 'top2026' or names")
    ap.add_argument("--synthetic", default="12,24,36:0.35,0.55:0,1",
                    help="classes:shelter-ratios:generator-seeds")
    ap.add_argument("--validator", type=Path,
                    default=ROOT.parent / "validator-lib" / "dist" / "cli" / "main.js")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    names: list[str] = []
    for item in args.scenarios.split(","):
        if item == "realistic":
            names += list(REALISTIC)
        elif item == "synthetic":
            cls, ratios, gseeds = (part.split(",") for part in args.synthetic.split(":"))
            names += [f"gen_c{c}_r{r}_s{s}" for c in cls for r in ratios for s in gseeds]
        elif item == "top2026":
            cls, ratios, gseeds = (part.split(",") for part in args.synthetic.split(":"))
            names += [f"top_c{c}_r{r}_s{s}" for c in cls for r in ratios for s in gseeds]
        else:
            names.append(item)
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")
    tasks = [Task(n, v, s, args.time_limit, args.workers) for n in names for s in seeds
             for v in variants]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"bench-{stamp}" + (f"-{args.tag}" if args.tag else "")
    out_dir = args.out / run_id
    out_dir.mkdir(parents=True)
    config = {
        "run_id": run_id,
        "started_utc": stamp,
        "solver_version": __version__,
        "solver_git": _git("rev-parse", "--short", "HEAD"),
        "solver_branch": _git("branch", "--show-current"),
        "solver_dirty": bool(_git("status", "--porcelain")),
        "ortools": version("ortools"),
        "python": platform.python_version(),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "validator_git": subprocess.run(
            ["git", "-C", str(args.validator.parents[2]), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "scenarios": names,
        "tasks": len(tasks),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"{run_id}: {len(tasks)} tasks, jobs={args.jobs}, workers={args.workers}", flush=True)

    rows: list[dict[str, object]] = []
    lines: list[str] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(execute, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                row, line = fut.result()
            except Exception as exc:  # noqa: BLE001 - a bench keeps going and records the crash
                row, line = {"case_id": t.case_id, "scenario": t.scenario, "variant": t.variant,
                             "seed": t.seed, "status": "CRASH", "note": repr(exc)[:300]}, None
            rows.append(row)
            if line:
                lines.append(line)
            with (out_dir / "progress.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(tasks)}] {t.case_id} {row.get('status')} "
                  f"share={row.get('primary_share')} wall={row.get('wall_s')}", flush=True)

    verdicts = validate(lines, args.validator, out_dir) if lines else {}
    for row in rows:
        status, n = verdicts.get(str(row["case_id"]), ("", 0))
        row["validator_status"], row["validator_conflicts"] = status, n
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    rows.sort(key=lambda r: (str(r.get("scenario")), str(r.get("variant")), int(r.get("seed", 0))))
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"done: {out_dir / 'runs.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
