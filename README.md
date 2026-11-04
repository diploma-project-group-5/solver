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
