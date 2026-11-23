"""Cooperative stop for a running solve (used by the worker on SIGTERM).

No ``threading`` import: CpSolver.stop_search() is safe to call from another thread, and
attribute reads/writes below are atomic under the GIL. The engine registers each CpSolver it
starts; ``stop()`` interrupts the current one and makes later phases return immediately with
their best solution.
"""

from typing import Protocol


class Stoppable(Protocol):
    """What a stop needs from a running solver (``CpSolver`` or a test double)."""

    def stop_search(self) -> None: ...


class StopHandle:
    __slots__ = ("_current", "_stopped")

    def __init__(self) -> None:
        self._stopped = False
        self._current: Stoppable | None = None

    @property
    def stopped(self) -> bool:
        return self._stopped

    def stop(self) -> None:
        self._stopped = True
        current = self._current
        if current is not None:
            current.stop_search()

    def attach(self, solver: Stoppable) -> None:
        """Register the solver about to run; stops it at once if a stop was already requested."""
        self._current = solver
        if self._stopped:
            solver.stop_search()

    def detach(self) -> None:
        self._current = None
