"""The ``contracts`` checkout for conformance tests (JSON Schema 2020-12, shared vectors).

Read from ``LAD_CONTRACTS_DIR`` (default ``../contracts`` next to this repo). A test module skips
when the checkout is absent (``SKIP_WITHOUT_CONTRACTS``); CI checks out diploma-project-group-5/
contracts at a pinned commit and sets LAD_CONTRACTS_DIR, so there a missing layout fails loudly.
"""

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

_DEFAULT = Path(__file__).resolve().parents[3] / "contracts"
CONTRACTS = Path(os.environ.get("LAD_CONTRACTS_DIR", str(_DEFAULT)))
SCHEMAS = CONTRACTS / "schemas" / "events"
VECTORS = CONTRACTS / "vectors"

# CI sets LAD_CONTRACTS_DIR: there a moved or renamed layout must fail loudly, not skip.
if "LAD_CONTRACTS_DIR" in os.environ and not SCHEMAS.is_dir():
    msg = "LAD_CONTRACTS_DIR is set but the contracts layout is missing"
    raise RuntimeError(msg)

SKIP_WITHOUT_CONTRACTS = pytest.mark.skipif(
    not SCHEMAS.is_dir(),
    reason=(
        f"contracts repo not available in CI (looked in {CONTRACTS}; set LAD_CONTRACTS_DIR) "
        "— see ADR 2026-11-04"
    ),
)


def load(path: Path) -> Any:  # noqa: ANN401 -- arbitrary JSON
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def validator(schema_file: str) -> Draft202012Validator:
    registry: Registry[Any] = Registry()
    for path in sorted(SCHEMAS.glob("*.json")):
        schema = load(path)
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    schema = load(SCHEMAS / schema_file)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())


def schema_errors(schema_file: str, document: Any) -> list[str]:  # noqa: ANN401 -- arbitrary JSON
    found = validator(schema_file).iter_errors(document)  # pyright: ignore[reportUnknownMemberType]
    return [f"{list(e.absolute_path)}: {e.message}" for e in found]


def vectors(folder: str) -> list[Path]:
    """The JSON vectors of ``vectors/<folder>`` (empty without the checkout)."""
    return sorted((VECTORS / folder).glob("*.json")) if SCHEMAS.is_dir() else []
