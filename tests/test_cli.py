"""CLI: generate -> solve -> result JSON parses back; generator is realistic and deterministic."""

import importlib
import json
import types
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pytest

import lad_solver
import oracle
from lad_solver.adapters.kafka.mapping import MODEL_VERSION
from lad_solver.cli.generate import generate_problem
from lad_solver.cli.io import (
    canonical_sha256,
    problem_from_json,
    problem_to_json,
    read_json,
    result_from_json,
    result_to_json,
)
from lad_solver.cli.main import main
from lad_solver.domain.prechecks import precheck
from lad_solver.domain.problem import AlertLimits, TimetableKind
from lad_solver.domain.result import (
    CapacityCondition,
    Conflict,
    ConflictRule,
    DayFormat,
    Lesson,
    Result,
    SolveStatus,
    Stats,
    StopReason,
)


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


def test_solve_writes_provenance(tmp_path: Path) -> None:
    problem_path = _problem_file(tmp_path)
    result_path = tmp_path / "result.json"
    assert main(["solve", "--in", str(problem_path), "--out", str(result_path),
                 "--time-limit", "10", "--workers", "1", "--seed", "3",
                 "--deterministic-time", "5"]) == 0  # fmt: skip

    provenance = json.loads(result_path.read_text(encoding="utf-8"))["provenance"]
    assert provenance == {
        "solver_version": lad_solver.__version__,
        "model_version": MODEL_VERSION,
        "ortools_version": version("ortools"),
        "input_sha256": canonical_sha256(read_json(problem_path)),
        "params": {
            "time_limit_s": 10.0,
            "max_deterministic_time": 5.0,
            "num_workers": 1,
            "random_seed": 3,
            "interleave_search": False,
        },
    }
    # The provenance does not change how the result itself parses back.
    assert result_from_json(read_json(result_path)).status in {
        SolveStatus.OPTIMAL,
        SolveStatus.FEASIBLE,
    }


def test_input_hash_ignores_key_order_and_whitespace() -> None:
    data = problem_to_json(generate_problem(2, 0.5, seed=1))
    reordered = json.loads(json.dumps(dict(reversed(list(data.items()))), indent=3))
    assert canonical_sha256(reordered) == canonical_sha256(data)
    changed = {**data, "staff": 999}
    assert canonical_sha256(changed) != canonical_sha256(data)


def test_problem_json_round_trip() -> None:
    problem = generate_problem(6, 0.5, seed=3, kind=TimetableKind.ALERT, saturday=True)
    assert problem_from_json(problem_to_json(problem)) == problem


def test_result_json_round_trip() -> None:
    result = Result(
        status=SolveStatus.INFEASIBLE,
        lessons=(Lesson("5-A", 1, 2, "math", "t1"),),
        day_formats=(DayFormat("5-A", 1, in_person=True), DayFormat("5-A", 2, in_person=False)),
        conflicts=(
            Conflict(ConflictRule.STAFF_OVER_PASSPORT, day=3, condition=CapacityCondition.SEATS),
            Conflict(ConflictRule.PACKING, teacher_id="t1", hint=True),
        ),
        stats=Stats(
            objective=42,
            best_bound=43.5,
            primary=4,
            secondary=2,
            wall_time_s=1.25,
            first_solution_s=0.5,
            last_improvement_s=0.75,
            phase_a_status="FEASIBLE",
            quality_relaxed=True,
            stop_reason=StopReason.NO_IMPROVEMENT,
            num_workers=2,
            random_seed=7,
        ),
    )
    assert result_from_json(result_to_json(result)) == result


@pytest.mark.parametrize(
    ("kind", "capacity"),
    [("main", {"seats": 10, "passport": 20}), ("alert", {"passport": 20})],
)
def test_capacity_shape_must_match_kind(kind: str, capacity: dict[str, int]) -> None:
    data = problem_to_json(generate_problem(2, 0.5, seed=1))
    data["kind"] = kind
    data["capacity"] = capacity
    with pytest.raises(ValueError, match="capacity"):
        problem_from_json(data)


def test_conflict_hint_must_be_a_boolean() -> None:
    data = result_to_json(
        Result(SolveStatus.INFEASIBLE, conflicts=(Conflict(ConflictRule.PACKING),))
    )
    conflicts = data["conflicts"]
    assert isinstance(conflicts, list)
    conflicts[0]["hint"] = "yes"
    with pytest.raises(TypeError, match="hint"):
        result_from_json(data)


def _problem_file(tmp_path: Path) -> Path:
    path = tmp_path / "problem.json"
    assert main(["generate", "--classes", "2", "--shelter-ratio", "0.5", "--out", str(path)]) == 0
    return path


def _bad_inputs(tmp_path: Path) -> dict[str, Path]:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    wrong_type = tmp_path / "wrong_type.json"
    data = json.loads(_problem_file(tmp_path).read_text(encoding="utf-8"))
    data["lessons_per_day"] = "seven"
    wrong_type.write_text(json.dumps(data), encoding="utf-8")
    return {"missing": tmp_path / "missing.json", "invalid": invalid, "wrong_type": wrong_type}


@pytest.mark.parametrize("case", ["missing", "invalid", "wrong_type"])
def test_solve_rejects_bad_input_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    source = _bad_inputs(tmp_path)[case]
    capsys.readouterr()
    assert main(["solve", "--in", str(source), "--out", str(tmp_path / "r.json")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("lad-solver solve: ")
    assert "Traceback" not in err


@pytest.mark.parametrize("command", ["solve", "generate"])
def test_missing_output_directory_is_rejected_up_front(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    out = str(tmp_path / "nope" / "out.json")
    argv = (
        ["solve", "--in", str(_problem_file(tmp_path)), "--out", out]
        if command == "solve"
        else ["generate", "--classes", "2", "--shelter-ratio", "0.5", "--out", out]
    )
    capsys.readouterr()
    assert main(argv) == 2
    assert "output directory does not exist" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--time-limit", "0"),
        ("--time-limit", "-1"),
        ("--time-limit", "nan"),
        ("--time-limit", "inf"),
        ("--time-limit", "abc"),
        ("--deterministic-time", "0"),
        ("--deterministic-time", "nan"),
        ("--workers", "0"),
        ("--workers", "-2"),
        ("--workers", "1.5"),
    ],
)
def test_solve_rejects_out_of_range_options(tmp_path: Path, option: str, value: str) -> None:
    argv = ["solve", "--in", str(tmp_path / "p.json"), "--out", str(tmp_path / "r.json")]
    with pytest.raises(SystemExit) as exc:
        main([*argv, option, value])
    assert exc.value.code == 2


@pytest.mark.parametrize("ratio", ["nan", "inf", "-0.1"])
def test_generate_rejects_bad_shelter_ratio_option(tmp_path: Path, ratio: str) -> None:
    argv = ["generate", "--classes", "2", "--shelter-ratio", ratio, "--out", str(tmp_path / "p")]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_generate_rejects_zero_classes_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["generate", "--classes", "0", "--shelter-ratio", "0.5", "--out", str(tmp_path / "p")]
    assert main(argv) == 2
    assert capsys.readouterr().err.startswith("lad-solver generate: classes must be >= 1")


def test_unwritable_output_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "taken"
    out.mkdir()  # a directory where the file should go: write_text raises IsADirectoryError
    argv = ["generate", "--classes", "2", "--shelter-ratio", "0.5", "--out", str(out)]
    assert main(argv) == 2
    assert capsys.readouterr().err.startswith("lad-solver generate: ")


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), -0.1])
def test_generate_problem_rejects_bad_shelter_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="shelter_ratio"):
        generate_problem(2, ratio)


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
