"""Layering guard: inner layers must not depend on outer ones (adapters, cli)."""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "lad_solver"

FORBIDDEN: dict[str, tuple[str, ...]] = {
    "domain": ("lad_solver.model", "lad_solver.app", "lad_solver.adapters", "lad_solver.cli"),
    "model": ("lad_solver.app", "lad_solver.adapters", "lad_solver.cli"),
    "app": ("lad_solver.adapters", "lad_solver.cli"),
}

# The engine stays pure and single-threaded: no transport, validation frameworks or threads.
FORBIDDEN_THIRD_PARTY = (
    "confluent_kafka",
    "pydantic",
    "jsonschema",
    "signal",
    "threading",
    "lad_solver.adapters",
    "lad_solver.cli",
)


def _imported_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_does_not_import_outer_layers(layer: str) -> None:
    violations = [
        f"{path.relative_to(PACKAGE_ROOT)} imports {module}"
        for path in sorted((PACKAGE_ROOT / layer).rglob("*.py"))
        for module in _imported_modules(path)
        if module.startswith(FORBIDDEN[layer])
    ]
    assert violations == []


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_engine_layer_has_no_transport_or_threads(layer: str) -> None:
    violations = [
        f"{path.relative_to(PACKAGE_ROOT)} imports {module}"
        for path in sorted((PACKAGE_ROOT / layer).rglob("*.py"))
        for module in _imported_modules(path)
        if module == "signal" or module.startswith(FORBIDDEN_THIRD_PARTY)
    ]
    assert violations == []
