"""Confirmatory analysis of the bench, exactly as in experiments/PREREG.md (edition 2).

    uv run --with scipy python bench/analyze.py --main <run> --rerun <run> --out <dir>

Reads only runs.csv and validate_in.jsonl of the two runs, plus the scenario inputs (bench.py).
The share is recomputed from the final timetable for every variant; for joint it must match
runs.csv. Writes analysis.md, per_scenario.csv, comparison.csv into --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FOUND = {"OPTIMAL", "FEASIBLE"}
NO_TIMETABLE = {"UNKNOWN", "INFEASIBLE", "MODEL_INVALID", "PRECHECK"}
BASELINES = ("rotation", "sequential")
ALPHA = 0.05


class AnalysisStop(RuntimeError):
    """A PREREG stop condition (oracle failure, mismatch, unexpected crash)."""


# ---------------------------------------------------------------- pure functions (tested)


def primary_of(snapshot: dict, timetable: dict) -> tuple[int, int]:
    """(primary, primary_max) from a validation.input.v1 snapshot and timetable."""
    alert = snapshot["kind"] == "alert"
    students = {c["id"]: c["students"] for c in snapshot["classes"]}
    subj = {s["id"]: s for s in snapshot["subjects"]}

    def counts(subject_id: str) -> bool:
        s = subj[subject_id]
        return s["priority_area"] and not (alert and s["no_shelter"])

    in_person = {(f["class_id"], f["day"]) for f in timetable["day_formats"] if f["in_person"]}
    primary = sum(
        students[x["class_id"]]
        for x in timetable["lessons"]
        if counts(x["subject_id"]) and (x["class_id"], x["day"]) in in_person
    )
    pmax = sum(students[r["class_id"]] * r["hours"] for r in snapshot["rows"] if counts(r["subject_id"]))
    return primary, pmax


@dataclass(frozen=True)
class Inputs:
    pmax: int
    priority_pupils: int  # sum of pupils of classes with >= 1 countable priority hour
    tight: bool

    @property
    def delta(self) -> float:
        """Share equal to one more in-person priority lesson per class per week."""
        return self.priority_pupils / self.pmax if self.pmax else math.inf


def inputs_of(snapshot: dict) -> Inputs:
    alert = snapshot["kind"] == "alert"
    subj = {s["id"]: s for s in snapshot["subjects"]}
    students = {c["id"]: c["students"] for c in snapshot["classes"]}
    countable = [
        r for r in snapshot["rows"]
        if subj[r["subject_id"]]["priority_area"]
        and not (alert and subj[r["subject_id"]]["no_shelter"]) and r["hours"] > 0
    ]
    pclasses = {r["class_id"] for r in countable}
    pupils = sum(students.values())
    people = pupils + len(snapshot["teachers"]) + snapshot["staff"]
    cap = snapshot["capacity"]
    tight = cap["passport"] < people or (alert and cap.get("seats", math.inf) < pupils)
    pmax = sum(students[r["class_id"]] * r["hours"] for r in countable)
    return Inputs(pmax, sum(students[c] for c in pclasses), bool(tight))


def run_share(row: dict, line: dict | None) -> tuple[float, str]:
    """Share of one run under PREREG §4, and the reason code."""
    status, verdict = row.get("status", ""), row.get("validator_status", "")
    if status == "CRASH":
        raise AnalysisStop(f"unexpected CRASH in analysis set: {row['case_id']}")
    if status in FOUND:
        if not verdict:
            raise AnalysisStop(f"timetable without validator verdict: {row['case_id']}")
        if verdict != "ACCEPTED":
            return 0.0, "rejected"
        if line is None:
            raise AnalysisStop(f"accepted run without validate_in line: {row['case_id']}")
        primary, pmax = primary_of(line["input"]["snapshot"], line["input"]["timetable"])
        if not 0 <= primary <= pmax:
            raise AnalysisStop(f"primary out of range: {row['case_id']}")
        if row["variant"] == "joint" and row.get("primary") not in ("", None) and int(row["primary"]) != primary:
            raise AnalysisStop(f"joint recompute mismatch {row['case_id']}: {row['primary']} vs {primary}")
        return (primary / pmax if pmax else 0.0), "accepted"
    if status in NO_TIMETABLE or status == "":
        return 0.0, "no_timetable"
    raise AnalysisStop(f"unknown status {status!r}: {row['case_id']}")


def holm(pvalues: list[float | None]) -> list[float | None]:
    idx = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    out: list[float | None] = [None] * len(pvalues)
    m, running = len(idx), 0.0
    for k, (p, i) in enumerate(idx):
        running = max(running, min(1.0, (m - k) * p))
        out[i] = running
    return out


def wilcoxon_greater(d: list[float]) -> float | None:
    if sum(1 for x in d if x != 0) < 6:  # noqa: PLR2004 - PREREG: too few non-zero pairs
        return None
    from scipy.stats import wilcoxon  # noqa: PLC0415

    return float(wilcoxon(d, alternative="greater").pvalue)


def verdict_text(stat: bool, practical: bool, base: str) -> str:
    if not stat:
        return f"перевагу над {base} не підтверджено"
    if practical:
        return f"спільна модель дає більше очних пріоритетних уроків, ніж {base}"
    return f"перевага над {base} стійка, але менша за один урок на клас на тиждень"


# ---------------------------------------------------------------- driver


def _read_runs(d: Path) -> dict[str, dict]:
    return {r["case_id"]: r for r in csv.DictReader((d / "runs.csv").open(encoding="utf-8"))}


def _read_lines(d: Path) -> dict[str, dict]:
    p = d / "validate_in.jsonl"
    out = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rec = json.loads(raw)
            out[rec["case_id"]] = rec
    return out


def merge(main: Path, rerun: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    runs, lines = _read_runs(main), _read_lines(main)
    crashed = {k for k, r in runs.items() if r.get("status") == "CRASH"}
    rr, rl = _read_runs(rerun), _read_lines(rerun)
    if set(rr) != crashed:
        raise AnalysisStop(f"rerun must contain exactly the {len(crashed)} CRASH cases")
    runs.update(rr)
    lines.update(rl)
    return runs, lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--rerun", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import scipy  # noqa: PLC0415

    import bench  # noqa: PLC0415 - scenario inputs only
    from golden import validation_input  # noqa: PLC0415
    from lad_solver.domain.result import Result, SolveStatus  # noqa: PLC0415

    runs, lines = merge(args.main, args.rerun)
    scenarios = sorted({r["scenario"] for r in runs.values()})
    inputs = {
        s: inputs_of(validation_input(bench.load_scenario(s), Result(SolveStatus.UNKNOWN))["snapshot"])
        for s in scenarios
    }
    shares: dict[tuple[str, str], list[float]] = defaultdict(list)
    reasons: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cid, row in sorted(runs.items()):
        share, why = run_share(row, lines.get(cid))
        shares[row["scenario"], row["variant"]].append(share)
        reasons[row["scenario"], row["variant"]].append(why)
    med = {k: statistics.median(v) for k, v in shares.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_scenario.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "source", "tight", "delta", "variant", "median_share", "accepted",
                    "rejected", "no_timetable"])
        for s in scenarios:
            for v in ("joint", *BASELINES):
                rs = reasons[s, v]
                w.writerow([s, "synthetic" if s.startswith(("gen_", "top_")) else "realistic", inputs[s].tight,
                            round(inputs[s].delta, 5), v, round(med[s, v], 5), rs.count("accepted"),
                            rs.count("rejected"), rs.count("no_timetable")])

    tight = [s for s in scenarios if inputs[s].tight]
    comp = []
    for base in BASELINES:
        d = [med[s, "joint"] - med[s, base] for s in scenarios]
        units = [x / inputs[s].delta for x, s in zip(d, scenarios, strict=True)]
        above = [s for s, x in zip(scenarios, d, strict=True) if x >= inputs[s].delta]
        q = [s for s in scenarios
             if reasons[s, "joint"].count("accepted") >= 3 and reasons[s, base].count("accepted") >= 3]  # noqa: PLR2004
        dq = [med[s, "joint"] - med[s, base] for s in q]
        comp.append({
            "baseline": base, "n": len(d),
            "better": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d), "worse": sum(x < 0 for x in d),
            "median_diff_share": round(statistics.median(d), 5),
            "median_diff_lessons_per_class_week": round(statistics.median(units), 3),
            "q1_lessons": round(statistics.quantiles(units, n=4)[0], 3),
            "q3_lessons": round(statistics.quantiles(units, n=4)[2], 3),
            "p_raw": wilcoxon_greater(d),
            "above_delta_all": len(above), "tight_n": len(tight),
            "above_delta_tight": sum(1 for s in above if s in tight),
            "q_n": len(q), "q_median_diff": round(statistics.median(dq), 5) if dq else None,
        })
    for c, p in zip(comp, holm([c["p_raw"] for c in comp]), strict=True):
        c["p_holm"] = p
        stat = p is not None and p < ALPHA
        need_tight = math.ceil(c["tight_n"] / 2)
        practical = c["above_delta_all"] >= math.ceil(c["n"] / 2) and (
            c["tight_n"] == 0 or c["above_delta_tight"] >= need_tight)
        c["statistical"], c["practical"] = stat, practical
        c["verdict"] = verdict_text(stat, practical, c["baseline"])
        c["reliability_only"] = bool(stat and c["q_median_diff"] is not None and c["q_median_diff"] <= 0)
    with (args.out / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comp[0]))
        w.writeheader()
        w.writerows(comp)

    md = ["# Аналіз за PREREG (редакція 2)", "",
          f"Прогони: `{args.main.name}` + `{args.rerun.name}`; сценаріїв {len(scenarios)}, з них тісних {len(tight)}; scipy {scipy.__version__}.", "",
          "| порівняння | краща / нічия / гірша | медіана, уроків на клас на тиждень [Q1; Q3] | p (Голм) | поріг: усі / тісні | висновок |",
          "|---|---|---|---|---|---|"]
    for c in comp:
        p = "—" if c["p_holm"] is None else f"{c['p_holm']:.2g}"
        md.append(f"| joint vs {c['baseline']} | {c['better']} / {c['ties']} / {c['worse']} | "
                  f"{c['median_diff_lessons_per_class_week']} [{c['q1_lessons']}; {c['q3_lessons']}] | {p} | "
                  f"{c['above_delta_all']}/{c['n']} · {c['above_delta_tight']}/{c['tight_n']} | {c['verdict']}"
                  + (" (лише надійність: див. PREREG §4)" if c["reliability_only"] else "") + " |")
    bad = {k: v for k, v in reasons.items() if "rejected" in v or "no_timetable" in v}
    md += ["", "## Запуски без зарахованого розкладу (рахуються як 0)", ""]
    md += [f"- {s} / {v}: відхилено {r.count('rejected')}, без розкладу {r.count('no_timetable')} з {len(r)}"
           for (s, v), r in sorted(bad.items())] or ["- немає"]
    (args.out / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
