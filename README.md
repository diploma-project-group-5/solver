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
3. **The neighbour dates** of the ISO week: school days, ≥ `not_before`, not in `non_school_dates`, added
   one at a time — nearest first, on a tie the later date (spec: «додає сусідні навчальні дні по одному»).
   Each run places all lessons left over all dates added so far, the nearer date preferred (coverage ≫
   date ≫ windows ≫ hour); the first run that places them all ends the step. Without `non_school_dates`
   in the request step 3 is off.

What is left is `self_study` (its `step` is the last step that ran: 3 with step 3 on, else 2; after a stop,
the step reached). Every lesson of `lessons` gets exactly one proposal on the request's date; no other rows.

Hard constraints keep the week **no worse than the baseline** (template + approved substitutions of every
kind): a teacher and a class get one lesson per hour; weekly limit `max(limit, baseline load)`; day capacity
`max(passport, baseline load)` (`model/capacity.py`, shared with the build; formats are fixed, so seats never
change); a no-shelter lesson only on a remote day (1.2); an in-person class day keeps a lesson, a self-study
or a cancelled lesson; a lesson moves only into a slot free in the baseline, so a slot freed by the same job
stays free (no chains of moves). These only make proposals feasible — the rules live in validator-lib.

The baseline week follows timetable's week builder and validator-lib (`domain/substitution.py`):

- approved substitutions apply whatever their order (moves of one round are lifted before they are placed,
  so two moves may swap slots); on a non-school date they change nothing, unless they moved the lesson to
  another date before it stopped being a school day;
- lessons whose ids left the snapshot (an archived class, a deleted teacher: timetable projects active ids
  only, the published template keeps them) stay where they are; a lesson of an unknown class counts for
  nothing, an unknown teacher still counts on site and is never a candidate;
- absences are read the safe way: a date of `absence_dates` in the week without the absent teacher's row is
  the whole date, and his row always covers the lessons to cover (both logged as `job.request_repaired`).

There is always a solution: OPTIMAL, or FEASIBLE when SIGTERM, the wall clock or a step budget cuts in (the
proposals fixed so far stay, the rest is self-study; `stats.stop_reason` says why, and the worker logs
`job.partial`). The budget is deterministic — one worker, random seed 0 (fixed by the ADR; `SOLVER_SEED` is
for builds), `max_deterministic_time` per run: step 1 10 %, step 2 35 %, the step 3 runs share the rest,
unused time moves on; the total is `time_limit_s` deterministic seconds (clamped to [10, 180]). The wall
clock is only the safety net. A step model that ends INFEASIBLE / MODEL_INVALID is an engine bug ("no
change" is always feasible): its lessons go on, the answer is FEASIBLE and `engine.substitution_anomaly` is
logged as an error. On the platform seed school (30 classes, 48 teachers, 868 lessons; main and alert) a job
of one of its five busiest teachers takes 5–45 ms of wall time and under 0.001 deterministic seconds, also
when nobody has room left and step 3 runs.

Deviations from v24 this follows (contracts README items 7–9, accepted in the ADR): `absences[]` of all
teachers of the week (ids only, no reason) and `non_school_dates[]` in the request; **conditional**: with
`allow_support_moves=false` step 2 moves only the absent teacher's lessons. `allow_support_moves=true` is not
implemented yet: the engine answers as with false and logs `job.support_moves_ignored`.

A request is checked before `solve.started` is promised (`baseline_week`, the one gate): schema, ids of the
snapshot, every date in `iso_week`, the date a school day of `absence_dates` and not a non-school date, and a
consistent week. Each lesson to cover is the absent teacher's lesson of the date in the baseline, of a class of
the snapshot, with a day format, without an approved substitution of its own; with `absences[]`, every lesson
of his in an hour his row marks absent is listed. Otherwise MODEL_INVALID / `invalid_payload`: the DLQ copy
keeps only the envelope and the job id, and the broken rule is logged (`job.rejected`, field `reason`) and set
as the `x-error-reason` header — a fixed literal or a field path, never an input value. Unknown keys of `data`
are ignored and counted (`job.unknown_fields`): a misspelt optional field would turn a feature off. More than
600 proposals (the schema bound) is `result_too_large`.

`model_version` is `a+a2+b+s/2026-11-15`: phase A, its coverage step, phase B and the substitution steps.

Tests: `tests/adapters/test_substitution_vectors.py` runs all 23 `contracts/vectors/substitution-solve`
vectors; every answer, and every hand-made scenario of `tests/test_substitution_engine.py`, also goes through
`lad-validate --mode new-conflicts` (no new conflict), with the test-only week builder
`tests/substitution_golden.py` first checked against the contract's `given → input` vectors and against
proposals that must add a conflict. `tests/test_substitution_properties.py` (hypothesis, and one batch of
360 random, chained and capacity-tight weeks) covers the rest; the Kafka tests include kill -9 during a
substitution.

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
