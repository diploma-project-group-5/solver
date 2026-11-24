"""Summaries of a bench run for the thesis: medians/IQR per scenario and paired comparisons.

    uv run --with scipy python bench/summarize.py --run <run-dir>

Writes ``summary.md`` (tables) and ``per_scenario.csv`` / ``paired.csv`` next to ``runs.csv``.
Only runs with a timetable that validator-lib ACCEPTED enter the quality metrics; status counts
cover every run. Pairs are (scenario, seed): the same instance and seed for both variants.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("primary_share", "in_person_share", "zero_day_classes", "wall_s")


def q(values: list[float], p: float) -> float:
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def med_iqr(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{statistics.median(values):.3f} [{q(values, 0.25):.3f}; {q(values, 0.75):.3f}]"


def wilcoxon(diffs: list[float]) -> float | None:
    nz = [d for d in diffs if d != 0]
    if len(nz) < 6:  # noqa: PLR2004 - too few non-zero pairs for a meaningful p
        return None
    try:
        from scipy.stats import wilcoxon as w  # noqa: PLC0415

        return float(w(nz, alternative="greater").pvalue)
    except ImportError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run directory (output goes here)")
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="supplementary runs; their rows replace same case_id (e.g. re-runs)")
    args = ap.parse_args()
    merged: dict[str, dict[str, str]] = {}
    for d in (args.run, *args.extra):
        for r in csv.DictReader((d / "runs.csv").open(encoding="utf-8")):
            merged[r["case_id"]] = r
    rows = list(merged.values())

    ok = [r for r in rows if r.get("validator_status") == "ACCEPTED"]
    status = defaultdict(lambda: defaultdict(int))
    for r in rows:
        status[r["variant"]][r["status"]] += 1
    rejected = [r for r in rows if r.get("validator_status") not in ("", "ACCEPTED", None)]

    by = defaultdict(lambda: defaultdict(list))
    for r in ok:
        for m in METRICS:
            if r.get(m) not in ("", None):
                by[(r["scenario"], r["variant"])][m].append(float(r[m]))
    scenarios = sorted({r["scenario"] for r in rows})
    variants = [v for v in ("joint", "rotation", "sequential") if any(r["variant"] == v for r in rows)]

    with (args.run / "per_scenario.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "variant", "n_accepted", *[f"{m}_median" for m in METRICS],
                    *[f"{m}_q1" for m in METRICS], *[f"{m}_q3" for m in METRICS]])
        for s in scenarios:
            for v in variants:
                d = by[(s, v)]
                n = len(d["primary_share"])
                w.writerow([s, v, n,
                            *[round(statistics.median(d[m]), 4) if d[m] else "" for m in METRICS],
                            *[round(q(d[m], 0.25), 4) if d[m] else "" for m in METRICS],
                            *[round(q(d[m], 0.75), 4) if d[m] else "" for m in METRICS]])

    share = {(r["scenario"], r["variant"], r["seed"]): float(r["primary_share"]) for r in ok
             if r.get("primary_share")}
    paired_rows = []
    for base in [v for v in variants if v != "joint"]:
        for group, pick in (("realistic", lambda s: not s.startswith(("gen_", "top_"))),
                            ("synthetic", lambda s: s.startswith(("gen_", "top_"))),
                            ("all", lambda _s: True)):
            seed_diffs = [share[(s, "joint", seed)] - share[(s, base, seed)]
                          for (s, v, seed) in share if v == "joint" and pick(s)
                          and (s, base, seed) in share]
            if not seed_diffs:
                continue
            # Seeds of one scenario are not independent instances: the test runs on the
            # per-scenario medians (one pair per instance).
            diffs = []
            for s in scenarios:
                if not pick(s):
                    continue
                j, b = by[(s, "joint")]["primary_share"], by[(s, base)]["primary_share"]
                if j and b:
                    diffs.append(statistics.median(j) - statistics.median(b))
            wins = sum(d > 0 for d in diffs)
            ties = sum(d == 0 for d in diffs)
            paired_rows.append({
                "baseline": base, "group": group, "instances": len(diffs),
                "joint_better": wins, "ties": ties, "joint_worse": len(diffs) - wins - ties,
                "diff_median": round(statistics.median(diffs), 4),
                "diff_q1": round(q(diffs, 0.25), 4), "diff_q3": round(q(diffs, 0.75), 4),
                "wilcoxon_p_greater": wilcoxon(diffs),
                "seed_pairs": len(seed_diffs),
                "seed_pairs_joint_better": sum(d > 0 for d in seed_diffs),
            })
    with (args.run / "paired.csv").open("w", newline="", encoding="utf-8") as fh:
        if paired_rows:
            w = csv.DictWriter(fh, fieldnames=list(paired_rows[0]))
            w.writeheader()
            w.writerows(paired_rows)

    md = [f"# Bench summary: {args.run.name}", "",
          f"Runs: {len(rows)}; validator ACCEPTED: {len(ok)}; validator rejected: {len(rejected)}.",
          "", "## Status by variant", "", "| variant | " + " | ".join(
              sorted({k for d in status.values() for k in d})) + " |"]
    keys = sorted({k for d in status.values() for k in d})
    md.append("|---" * (len(keys) + 1) + "|")
    md += [f"| {v} | " + " | ".join(str(status[v].get(k, 0)) for k in keys) + " |" for v in variants]
    md += ["", "## Share of priority pupil-lessons held in person (median [Q1; Q3])", "",
           "| scenario | " + " | ".join(variants) + " |", "|---" * (len(variants) + 1) + "|"]
    md += [f"| {s} | " + " | ".join(med_iqr(by[(s, v)]["primary_share"]) for v in variants) + " |"
           for s in scenarios]
    md += ["", "## Wall time, s (median [Q1; Q3])", "",
           "| scenario | " + " | ".join(variants) + " |", "|---" * (len(variants) + 1) + "|"]
    md += [f"| {s} | " + " | ".join(med_iqr(by[(s, v)]["wall_s"]) for v in variants) + " |"
           for s in scenarios]
    md += ["", "## Paired: joint − baseline, primary_share (per-scenario medians over seeds)", "",
           "| baseline | group | instances | joint better / ties / worse | median diff [Q1; Q3] | "
           "Wilcoxon p (one-sided) | seed pairs joint better |", "|---|---|---|---|---|---|---|"]
    md += [f"| {p['baseline']} | {p['group']} | {p['instances']} | {p['joint_better']} / "
           f"{p['ties']} / {p['joint_worse']} | {p['diff_median']} [{p['diff_q1']}; "
           f"{p['diff_q3']}] | "
           f"{'—' if p['wilcoxon_p_greater'] is None else f'{p['wilcoxon_p_greater']:.2e}'} | "
           f"{p['seed_pairs_joint_better']}/{p['seed_pairs']} |"
           for p in paired_rows]
    (args.run / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
