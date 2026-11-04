"""Make test-only helpers (``builders``, ``oracle``) importable with ``--import-mode=importlib``."""

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
