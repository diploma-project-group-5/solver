# lad-solver

CP-SAT solver for «Лад»: builds the main timetable (capacity model 1.1), the alert timetable (model 1.2)
and substitution proposals. Runs as a Kafka worker (`solve.build.requested`, `solve.substitution.requested`
→ `solve.status`) and ships the `bench` evaluation CLI.

## Layout

```
src/lad_solver/
  domain/    pure value types (school snapshot, classes, teachers, curriculum, capacity limits;
             the substitution request, its baseline week and the proposals)
  model/     CP-SAT models: build phases A/B, the substitution steps, the shared day capacity — no I/O
  app/       use cases: solve (build, 1.1 and 1.2), substitute (substitution, stage 2)
  adapters/  Kafka worker, payload mapping from the `contracts` package
  cli/       bench
tests/       pytest + hypothesis; `test_architecture.py` enforces the layering
```

Logs: only the worker logs, as JSON lines on stdout with an allow-list of fields
(`adapters/kafka/jsonlog.py`); job ids come from the job context, also in the solve thread. The engine
(`domain`, `model`, `app`) has no `logging`: it reports through `Stats` (e.g. the coverage step verdict and
the phase B retry days, counts only), which the worker writes to `job.completed`.

## Substitution engine (stage 2)

ADR 2026-11-15-solver-substitution-engine; spec §06 «Відсутність вчителя → заміна»; contracts 0.5.0
(`solve.substitution.requested`, `solve.status` `data.substitution`). One job is one date of an absence.
The solver only **proposes**: timetable builds the week with the proposals, validator-lib (`newConflicts`)
gives the verdict, a person approves (or picks «скасовано», which the solver never proposes).

Steps, in order; each fixes what it proposed (`app/substitute.py`, one model builder `model/substitution.py`):

1. **A free teacher** in the same slot: not the absent teacher, works that day (has at least one lesson that
   day, Roman 15.11.2026) and only in the class's format, free and present (`absences[]`, unavailable days),
   room in the weekly limit. Objective: coverage ≫ «не за фахом» (no curriculum row with the subject) ≫
   new teachers on site ≫ teacher id.
2. **Another hour of the date** for the lessons step 1 left: the absent teacher where he is present and the
   class is free. Objective: coverage ≫ class windows (gaps) ≫ earliest hour.
3. **The neighbour dates** of the ISO week, one at a time: school days, ≥ `not_before`, not in
   `non_school_dates`; nearest first, on a tie the later date. Without `non_school_dates` in the request
   step 3 is off.

What is left is `self_study` (its `step` is the last step that ran: 3 with step 3 on, else 2). Every lesson
of `lessons` gets exactly one proposal on the request's date; no other rows.

Hard constraints keep the week **no worse than the baseline** (template + approved substitutions of every
kind): a teacher and a class get one lesson per hour; weekly limit `max(limit, baseline load)`; day capacity
`max(passport, baseline load)` (`model/capacity.py`, shared with the build; formats are fixed, so seats never
change); a no-shelter lesson only on a remote day (1.2); an in-person class day never loses its last lesson;
a lesson moves only into a slot free in the baseline, so a slot freed by the same job stays free (no chains
of moves). These only make proposals feasible — the rules live in validator-lib.

There is always a solution: OPTIMAL, or FEASIBLE when SIGTERM, the wall clock or a step budget cuts in (the
proposals fixed so far stay, the rest is self-study; `stats.stop_reason` says why). The budget is
deterministic — one worker, the configured seed (`SOLVER_SEED`, 0), `max_deterministic_time` per run: step 1
10 %, step 2 35 %, the step 3 dates share the rest, unused time moves on; the total is `time_limit_s`
deterministic seconds (clamped to [10, 180]). The wall clock is only the safety net. On the platform seed
school (30 classes, 48 teachers, 868 lessons; main and alert) a job of one of its five busiest teachers takes
6–45 ms of wall time and under 0.001 deterministic seconds, also when nobody has room left and step 3 runs.

Deviations from v24 this follows (contracts README items 7–9, accepted in the ADR): `absences[]` of all
teachers of the week (ids only, no reason) and `non_school_dates[]` in the request; **conditional**: with
`allow_support_moves=false` step 2 moves only the absent teacher's lessons. `allow_support_moves=true` is not
implemented yet: the engine answers as with false and logs `job.support_moves_ignored`.

A request is checked before `solve.started` is promised: schema, ids, every date in `iso_week`, the date a
school day of `absence_dates`, and a consistent baseline week (each approved substitution on a real lesson,
the lessons to cover the absent teacher's). Otherwise MODEL_INVALID / `invalid_payload`, and the DLQ copy
keeps only the envelope and the job id. More than 600 proposals (the schema bound) is `result_too_large`.

`model_version` is `a+a2+b+s/2026-11-15`: phase A, its coverage step, phase B and the substitution steps.

Tests: `tests/adapters/test_substitution_vectors.py` runs all 23 `contracts/vectors/substitution-solve`
vectors; every answer also goes through `lad-validate --mode new-conflicts` (no new conflict), with the
test-only week builder `tests/substitution_golden.py` first checked against the contract's `given → input`
vectors. `tests/test_substitution_properties.py` (hypothesis and a 240-week batch) and
`tests/test_substitution_engine.py` cover the rest; the Kafka tests include kill -9 during a substitution.

## Development

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.14 itself).

```bash
uv sync                      # create .venv from uv.lock
uv run pytest                # tests
uv run ruff format --check . # formatting
uv run ruff check .          # lint
uv run pyright               # strict type checking
```

Integration tests (Kafka via testcontainers) are marked `integration` and need Docker.

Results in tests are checked by the validator-lib CLI, not by Python rules (`tests/golden.py`,
`tests/substitution_golden.py`): the timetable rules live in validator-lib only. The tests need Node 24 and
`LAD_VALIDATE_CLI` pointing at the CLI (validator-lib ≥ 0.4.0 for `--mode new-conflicts`); without it they
fail instead of passing silently. Contract tests read the `contracts` checkout from `LAD_CONTRACTS_DIR`
(default `../contracts`). CI uses the release commits pinned in `.github/workflows/ci.yml` (validator-lib
built with `npm ci`, its lockfile); locally the same:

```bash
git clone https://github.com/diploma-project-group-5/validator-lib.git /tmp/validator
git -C /tmp/validator checkout d4952fc4a91d49a1557585853872bb42e3e54744  # v0.4.0
(cd /tmp/validator && npm ci)
export LAD_VALIDATE_CLI=/tmp/validator/dist/cli/main.js

git clone https://github.com/diploma-project-group-5/contracts.git /tmp/contracts
git -C /tmp/contracts checkout 5c0f3dd299dabc6fccd3c675f19e0a199718c640  # v0.5.0
export LAD_CONTRACTS_DIR=/tmp/contracts
```
