# bench – experiments for the thesis

Scripts that produced the numbers in chapter 4 of the thesis (joint scheduling vs. rotation of parallels vs. the sequential approach, and the checks of validator-lib). They live outside `src/` and are not shipped with the engine.

| Script | What it does |
|---|---|
| `bench.py` | Runs the solver on the realistic schools of `tests/realistic` and on generated schools, three variants × seeds, same time limit; every timetable goes through validator-lib. Writes `runs.csv`, `validate_in.jsonl`, `validate_out.jsonl`. |
| `inject.py` | Breaks accepted timetables on purpose, one operator per validator rule, and counts what validator-lib detects (`inject-s<seed>/summary.csv`). |
| `analyze.py`, `summarize.py` | Per-school medians, comparisons with the baselines (Wilcoxon signed-rank, Holm), tables for the thesis. |
| `test_analyze.py` | Tests of the analysis on synthetic data. |

Generated schools come from `lad-solver generate` (profiles `legacy` and `top2026`).

```sh
uv run python bench/bench.py --out <dir> --time-limit 60 --workers 4 --jobs 3 --seeds 0,1,2,3,4
python3 -I bench/inject.py --run <run dir> --per-op 5 --seed 0
uv run --with scipy python bench/summarize.py --run <run dir>
PYTHONPATH=bench uv run --with scipy pytest bench/test_analyze.py
```

The thesis numbers come from the runs of 21.11.2026 on commit 11a9c22 (this branch's base).
