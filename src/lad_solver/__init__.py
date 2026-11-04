"""Lad timetable solver.

Layers (hexagonal):
- ``domain``   -- pure value types: school snapshot, classes, teachers, curriculum, capacity limits.
- ``model``    -- CP-SAT model building, solving and conflict extraction. No I/O.
- ``app``      -- use cases: build_main (model 1.1), build_alert (model 1.2), substitution.
- ``adapters`` -- Kafka worker (solve.* topics), payload mapping from ``contracts``.
- ``cli``      -- ``bench`` evaluation CLI.

The domain and model layers must never import from adapters.
"""

from importlib.metadata import version

__all__ = ["__version__"]

__version__ = version("lad-solver")
