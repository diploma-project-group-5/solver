"""Cross-repo delivery contract (ADR 2026-11-04-kafka-delivery-mvp).

1. The worker's topic configuration matches ``contracts/topics.v1.json``: the solver consumes only
   topics whose consumers include it, publishes only where it is the producer, and its DLQ follows
   the manifest pattern. A drift here would break ``platform/kafka-init`` at integration time.
2. The job-lifecycle scenarios in ``contracts/vectors/job-lifecycle`` are internally consistent:
   a minimal reference job state machine (test-only, mirrors spec §05) reproduces every expected
   state. ``timetable-service`` runs the same scenarios against its real implementation.

Skips when the contracts checkout is absent (see ``test_contracts.py``).
"""

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
from kafka_support import (
    COMPLETED_STATE,
    SUBSTITUTION_TOPIC,
    ReferenceJob,
    decode,
    encode,
    harness,
    inbound,
    instant,
    substitution_request,
)

from lad_solver.adapters.kafka.config import WorkerConfig

_DEFAULT = Path(__file__).resolve().parents[3] / "contracts"
CONTRACTS = Path(os.environ.get("LAD_CONTRACTS_DIR", str(_DEFAULT)))
MANIFEST = CONTRACTS / "topics.v1.json"
LIFECYCLE = CONTRACTS / "vectors" / "job-lifecycle"

# CI sets LAD_CONTRACTS_DIR: there a moved or renamed layout must fail loudly, not skip.
if "LAD_CONTRACTS_DIR" in os.environ and not MANIFEST.is_file():
    msg = "LAD_CONTRACTS_DIR is set but the contracts layout is missing"
    raise RuntimeError(msg)

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason=(
        f"contracts repo not available in CI (looked in {CONTRACTS}; set LAD_CONTRACTS_DIR) "
        "— see ADR 2026-11-04-kafka-delivery-mvp"
    ),
)

Json = dict[str, Any]
SOLVER = "solver"


def _load(path: Path) -> Json:
    return cast("Json", json.loads(path.read_text(encoding="utf-8")))


def _topics() -> dict[str, Json]:
    return {t["name"]: t for t in _load(MANIFEST)["topics"]}


def _config() -> WorkerConfig:
    return WorkerConfig.from_env({"KAFKA_BOOTSTRAP_SERVERS": "kafka:9092"})


# ---------------------------------------------------------------- topic manifest


def test_solver_consumes_only_topics_it_is_a_consumer_of() -> None:
    topics = _topics()
    for name in _config().consumed_topics:
        assert SOLVER in topics[name]["consumers"], name
    assert topics[_config().build_topic]["stage"] == 1


def test_substitution_topic_is_the_stage_2_topic_the_solver_solves() -> None:
    """ADR 2026-11-15-solver-substitution-engine: the stage 2 engine answers these jobs."""
    topic = _topics()[_config().substitution_topic]
    assert topic["stage"] == 2
    assert SOLVER in topic["consumers"]


def test_solver_publishes_status_only_where_it_is_the_producer() -> None:
    assert _topics()[_config().status_topic]["producer"] == SOLVER


def test_solver_dlq_follows_the_manifest_pattern() -> None:
    pattern: str = _load(MANIFEST)["dlq"]["pattern"]
    config = _config()
    for topic in config.consumed_topics:
        assert config.dlq_for(topic) == pattern.replace("<topic>", topic)


def test_manifest_has_no_retry_topics_and_keys_by_school() -> None:
    manifest = _load(MANIFEST)
    assert not [name for name in _topics() if ".retry" in name]
    assert manifest["defaults"]["key"] == "school_id"


def test_every_solver_topic_with_a_schema_points_to_an_existing_file() -> None:
    for topic in _topics().values():
        if SOLVER in (topic["producer"], *topic["consumers"]) and topic["schema"]:
            assert (CONTRACTS / topic["schema"]).is_file(), topic["name"]


# ---------------------------------------------------------------- job lifecycle


def _scenarios() -> list[Path]:
    return sorted(LIFECYCLE.glob("*.json")) if LIFECYCLE.is_dir() else []


def test_lifecycle_scenarios_exist() -> None:
    assert len(_scenarios()) >= 14
    assert (LIFECYCLE / "13-substitution-unsupported.json").is_file()
    assert (LIFECYCLE / "14-substitution-completed.json").is_file()


@pytest.mark.parametrize("path", _scenarios(), ids=lambda p: p.stem)
def test_lifecycle_scenario_is_consistent(path: Path) -> None:
    scenario = _load(path)
    job_spec: Json = scenario["job"]
    job = ReferenceJob(
        queue_deadline_at=instant(job_spec["queue_deadline_at"]),
        time_limit_s=job_spec["time_limit_s"],
        margin_s=scenario["run_deadline_margin_s"],
    )
    ignored: list[int] = []
    for index, step in enumerate(scenario["steps"]):
        changed = (
            job.on_event(step["event"])
            if "event" in step
            else job.on_watchdog(instant(step["watchdog_at"]))
        )
        if not changed:
            ignored.append(index)

    assert job.state == scenario["expected"]["state"]
    assert ignored == scenario["expected"]["ignored_steps"]


def test_solver_only_emits_statuses_covered_by_the_scenarios() -> None:
    statuses = {
        step["event"]["data"]["status"]
        for path in _scenarios()
        for step in _load(path)["steps"]
        if "event" in step and step["event"]["type"] == "solve.completed"
    }
    assert statuses == set(COMPLETED_STATE)


def test_unsupported_job_type_is_covered_by_a_scenario() -> None:
    """Solvers before 0.4.0 refuse substitutions: timetable closes such a job at once as failed
    (ADR 2026-11-15); the scenario stays for them."""
    codes = {
        step["event"]["data"]["error"]["code"]
        for path in _scenarios()
        for step in _load(path)["steps"]
        if "event" in step and "error" in step["event"]["data"]
    }
    assert "unsupported_job_type" in codes


def test_refused_substitution_scenario_fails_at_once_without_started() -> None:
    scenario = _load(LIFECYCLE / "13-substitution-unsupported.json")

    assert scenario["job"]["kind"] == "substitution"
    assert [step["event"]["type"] for step in scenario["steps"]] == ["solve.completed"]
    assert scenario["expected"] == {"state": "failed", "ignored_steps": []}


def test_solved_substitution_scenario_is_started_then_done() -> None:
    scenario = _load(LIFECYCLE / "14-substitution-completed.json")

    types = [step["event"]["type"] for step in scenario["steps"]]
    assert scenario["job"]["kind"] == "substitution"
    assert types == ["solve.started", "solve.completed"]
    assert "substitution" in scenario["steps"][1]["event"]["data"]
    assert scenario["expected"] == {"state": "done", "ignored_steps": []}


def test_the_solvers_substitution_events_take_the_job_to_done() -> None:
    """The worker's real events for a substitution job, replayed as timetable consumes them."""
    request = substitution_request()
    h = harness()
    h.run(inbound(encode(request), topic=SUBSTITUTION_TOPIC))
    job = ReferenceJob(
        queue_deadline_at=instant(request["data"]["queue_deadline_at"]),
        time_limit_s=request["data"]["time_limit_s"],
        margin_s=90,
    )

    changed = [job.on_event(decode(m)) for m in h.producer.sent]

    assert changed == [True, True]
    assert job.state == "done"
