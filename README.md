# lad-solver

CP-SAT solver for «Лад»: builds the main timetable (capacity model 1.1), the alert timetable (model 1.2)
and substitution proposals. Runs as a Kafka worker (`solve.build.requested`, `solve.substitution.requested`
→ `solve.status`) and ships the `bench` evaluation CLI.

## Layout

```
src/lad_solver/
  domain/    pure value types (school snapshot, classes, teachers, curriculum, capacity limits)
  model/     CP-SAT model: variables, constraints, objective, conflicts — no I/O
  app/       use cases: build_main, build_alert, substitution
  adapters/  Kafka worker, payload mapping from the `contracts` package
  cli/       bench
tests/       pytest + hypothesis; `test_architecture.py` enforces the layering
```

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

Results in tests are checked by the validator-lib CLI, not by Python rules (`tests/golden.py`): the timetable
rules live in validator-lib only. The tests need Node 24 and `LAD_VALIDATE_CLI` pointing at the CLI; without it
they fail instead of passing silently. CI installs the tag pinned in `.github/workflows/ci.yml`:

```bash
npm install --prefix /tmp/validator "git+https://github.com/diploma-project-group-5/validator-lib.git#v0.2.0"
export LAD_VALIDATE_CLI=/tmp/validator/node_modules/@lad/validator-lib/dist/cli/main.js
```
