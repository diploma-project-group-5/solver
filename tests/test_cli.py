"""CLI: generate -> solve -> result JSON parses back; generator is realistic and deterministic."""

import importlib
import json
import types
from collections import Counter
from pathlib import Path

import pytest

import oracle
from lad_solver.cli.generate import generate_problem
from lad_solver.cli.io import problem_from_json, problem_to_json, read_json, result_from_json
from lad_solver.cli.main import main
from lad_solver.domain.prechecks import precheck
from lad_solver.domain.problem import AlertLimits, TimetableKind
from lad_solver.domain.result import SolveStatus


def test_generate_then_solve_round_trip(tmp_path: Path) -> None:
    problem_path, result_path = tmp_path / "problem.json", tmp_path / "result.json"
    assert main(["generate", "--classes", "4", "--shelter-ratio", "0.55", "--seed", "1",
                 "--out", str(problem_path)]) == 0  # fmt: skip
    assert main(["solve", "--in", str(problem_path), "--out", str(result_path),
                 "--time-limit", "10", "--workers", "1"]) == 0  # fmt: skip

    problem = problem_from_json(read_json(problem_path))
    result = result_from_json(read_json(result_path))
    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert oracle.violations(problem, result) == []
    assert result.stats.objective == oracle.objective(problem, result).value
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == result.status.value


def test_problem_json_round_trip() -> None:
    problem = generate_problem(6, 0.5, seed=3, kind=TimetableKind.ALERT, saturday=True)
    assert problem_from_json(problem_to_json(problem)) == problem


def test_generator_is_deterministic_and_passes_prechecks() -> None:
    a = generate_problem(30, 0.55, seed=1)
    assert a == generate_problem(30, 0.55, seed=1)
    assert a != generate_problem(30, 0.55, seed=2)
    assert precheck(a) == ()


def test_generator_shape() -> None:
    p = generate_problem(24, 0.55, seed=1, kind=TimetableKind.ALERT)
    hours = Counter[str]()
    for r in p.rows:
        hours[r.class_id] += r.hours
    assert len(p.classes) == 24
    assert all(22 <= h <= 33 for h in hours.values())
    assert all(20 <= c.students <= 32 for c in p.classes)
    assert any(s.no_shelter for s in p.subjects)
    assert isinstance(p.capacity, AlertLimits)
    assert p.capacity.seats < p.capacity.passport
    people = sum(c.students for c in p.classes) + p.staff + len(p.teachers)
    assert abs(p.capacity.passport - 0.55 * people) <= 1


@pytest.mark.parametrize("module", ["confluent_kafka", "confluent_kafka.cimpl", "pydantic"])
def test_worker_without_the_worker_extra_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], module: str
) -> None:
    def missing(_name: str) -> object:
        raise ModuleNotFoundError(name=module)

    monkeypatch.setattr(importlib, "import_module", missing)
    assert main(["worker"]) == 2
    assert "Kafka adapter is unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("module", ["lad_solver.adapters.kafka.mapping", "json", None])
def test_other_import_errors_are_not_hidden(
    monkeypatch: pytest.MonkeyPatch, module: str | None
) -> None:
    def broken(_name: str) -> object:
        raise ImportError(name=module)

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(ImportError):
        main(["worker"])


def test_worker_delegates_to_adapter_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def adapter(name: str) -> types.SimpleNamespace:
        assert name == "lad_solver.adapters.kafka.main"
        return types.SimpleNamespace(run=lambda: 7)

    monkeypatch.setattr(importlib, "import_module", adapter)
    assert main(["worker"]) == 7
