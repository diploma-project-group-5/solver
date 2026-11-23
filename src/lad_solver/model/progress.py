"""Phase A progress: first solution, last improvement, and the no-improvement stop.

ADR 2026-11-08: stop phase A when ``now - t_last >= max(min_s, ratio * (t_last - t0))``, where
``t0`` is the start of phase A and ``t_last`` the last strict improvement of the best objective
seen in any phase A run (probe, hinted follow-up, relaxed run share one clock).

No threads here (the engine stays single-threaded, see test_architecture): the solution callback
calls ``record`` from CP-SAT's thread, and the caller ticks ``check`` from its own loop (the
worker's main thread already waits on the solve every 0.5 s). Like ``StopHandle``, attribute
reads/writes are atomic under the GIL and ``stop_search`` is safe from another thread.

The stop never touches ``StopHandle``: phase B keeps its compact model and ``stop_reason`` does
not read ``stopped``.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from lad_solver.model.stop import Stoppable


@dataclass(frozen=True, slots=True)
class StallPolicy:
    min_s: float
    ratio: float

    def __post_init__(self) -> None:
        for name, value in (("min_s", self.min_s), ("ratio", self.ratio)):
            if not (math.isfinite(value) and value > 0):
                msg = f"stall {name} must be a finite number > 0, got {value!r}"
                raise ValueError(msg)

    def due(self, *, now: float, started: float, last_improvement: float | None) -> bool:
        if last_improvement is None:
            return False
        window = max(self.min_s, self.ratio * (last_improvement - started))
        return now - last_improvement >= window


class Progress:
    """Live view of one phase A; ``check`` is the only call made from the caller's thread."""

    __slots__ = (
        "_best",
        "_clock",
        "_current",
        "_policy",
        "_stalled",
        "first_solution_at",
        "last_improvement_at",
        "started",
    )

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._policy: StallPolicy | None = None
        self._current: Stoppable | None = None
        self._best: float | None = None
        self._stalled = False
        self.started: float | None = None
        self.first_solution_at: float | None = None
        self.last_improvement_at: float | None = None

    @property
    def stalled(self) -> bool:
        return self._stalled

    def begin(self, policy: StallPolicy | None) -> None:
        """Start of phase A: resets all state; ``None`` records telemetry only."""
        self._policy = None  # no check() may act on half-reset state
        self._current = None
        self._best = None
        self._stalled = False
        self.first_solution_at = None
        self.last_improvement_at = None
        self.started = self._clock()
        self._policy = policy

    def finish(self) -> bool:
        """End of phase A: later ``check`` calls do nothing. Returns whether the rule fired.

        The snapshot is the answer for this phase: a tick racing with ``finish`` cannot relabel
        a phase that ended by time limit or proof.
        """
        self._policy = None
        return self._stalled

    def attach(self, solver: Stoppable) -> None:
        self._current = solver
        if self._stalled:
            solver.stop_search()

    def detach(self) -> None:
        self._current = None

    def record(self, objective: float) -> None:
        """A solution with this objective value was found (solution callback)."""
        now = self._clock()
        if self.first_solution_at is None:
            self.first_solution_at = now
        if self._best is None or objective > self._best:
            self._best = objective
            self.last_improvement_at = now

    def check(self) -> bool:
        """Stop the running phase A solver if the stall rule is due; True once it has fired.

        Once fired, every tick repeats ``stop_search`` on the current solver: CP-SAT ignores the
        call until ``solve()`` has started, so a single call can be lost between two runs.
        """
        policy, started = self._policy, self.started
        if policy is None or started is None:
            return False
        if self._stalled:
            current = self._current
            if current is not None:
                current.stop_search()
            return True
        if not policy.due(
            now=self._clock(), started=started, last_improvement=self.last_improvement_at
        ):
            return False
        self._stalled = True
        current = self._current
        if current is not None:
            current.stop_search()
        return True
