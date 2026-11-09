# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
# Base images are pinned by digest (engineering standard); bump tag and digest together.

# ---- build: resolve the locked environment with uv -------------------------------------------
FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv
FROM python:3.14-slim@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f AS build

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
FROM python:3.14-slim@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f AS runtime

RUN groupadd --system --gid 10001 solver \
 && useradd --system --uid 10001 --gid solver --no-create-home solver

# Owned by root: the runtime user can read and run the venv but not rewrite its own code.
COPY --from=build /app/.venv /app/.venv
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
