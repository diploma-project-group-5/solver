# syntax=docker/dockerfile:1.7
# TODO(platform): pin both base images by digest once the registry/CI flow is set up.

# ---- build: resolve the locked environment with uv -------------------------------------------
FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM python:3.14-slim AS build

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app
# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra worker --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra worker --no-editable

# ---- runtime: no uv, no build tools, non-root ------------------------------------------------
FROM python:3.14-slim AS runtime

RUN groupadd --system --gid 10001 solver \
 && useradd --system --uid 10001 --gid solver --no-create-home solver

COPY --from=build --chown=solver:solver /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SOLVER_HEARTBEAT_FILE=/tmp/lad-solver.alive

USER solver
# The worker touches the heartbeat file at least every ~1 s (idle poll) or 0.5 s (while solving).
# Informational: compose does not restart unhealthy containers; crashes are handled by restart.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os,sys,time; p=os.environ.get('SOLVER_HEARTBEAT_FILE','/tmp/lad-solver.alive'); sys.exit(0 if time.time()-os.stat(p).st_mtime<30 else 1)"]
# Run with compose `init: true` (ADR 2026-11-04 §4): tini forwards SIGTERM and reaps children.
ENTRYPOINT ["lad-solver"]
CMD ["worker"]
