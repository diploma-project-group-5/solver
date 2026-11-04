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
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from lad_solver.adapters.kafka.config import WorkerConfig

_DEFAULT = Path(__file__).resolve().parents[3] / "contracts"
CONTRACTS = Path(os.environ.get("LAD_CONTRACTS_DIR", str(_DEFAULT)))
MANIFEST = CONTRACTS / "topics.v1.json"
LIFECYCLE = CONTRACTS / "vectors" / "job-lifecycle"

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason=(
        f"contracts repo not available in CI (looked in {CONTRACTS}; set LAD_CONTRACTS_DIR) "
        "— see ADR 2026-11-04-kafka-delivery-mvp"
    ),
)

Json = dict[str, Any]
SOLVER = "solver"
TERMINAL = frozenset({"done", "infeasible", "rejected", "timeout", "failed"})
COMPLETED_STATE = {
    "OPTIMAL": "done",
    "FEASIBLE": "done",
    "INFEASIBLE": "infeasible",
    "UNKNOWN": "timeout",
    "MODEL_INVALID": "failed",
    "EXPIRED": "failed",
}


def _load(path: Path) -> Json:
    return cast("Json", json.loads(path.read_text(encoding="utf-8")))


def _topics() -> dict[str, Json]:
    return {t["name"]: t for t in _load(MANIFEST)["topics"]}


def _config() -> WorkerConfig:
    return WorkerConfig.from_env({"KAFKA_BOOTSTRAP_SERVERS": "kafka:9092"})


# ---------------------------------------------------------------- topic manifest


def test_solver_consumes_only_topics_it_is_a_consumer_of() -> None:
    topic = _topics()[_config().build_topic]
    assert SOLVER in topic["consumers"]
    assert topic["stage"] == 1


def test_solver_publishes_status_only_where_it_is_the_producer() -> None:
    assert _topics()[_config().status_topic]["producer"] == SOLVER


def test_solver_dlq_follows_the_manifest_pattern() -> None:
    pattern: str = _load(MANIFEST)["dlq"]["pattern"]
    config = _config()
    assert config.build_dlq_topic == pattern.replace("<topic>", config.build_topic)


def test_manifest_has_no_retry_topics_and_keys_by_school() -> None:
    manifest = _load(MANIFEST)
    assert not [name for name in _topics() if ".retry" in name]
    assert manifest["defaults"]["key"] == "school_id"


def test_every_solver_topic_with_a_schema_points_to_an_existing_file() -> None:
    for topic in _topics().values():
        if SOLVER in (topic["producer"], *topic["consumers"]) and topic["schema"]:
            assert (CONTRACTS / topic["schema"]).is_file(), topic["name"]


# ---------------------------------------------------------------- job lifecycle


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(UTC)


@dataclass
class _Job:
    """Reference job state machine (spec §05) — test-only, used to validate the scenarios."""

    queue_deadline_at: datetime
    time_limit_s: int
    margin_s: int
    state: str = "queued"
    run_deadline_at: datetime | None = None
    seen: set[str] = field(default_factory=set[str])

    def on_event(self, event: Json) -> bool:
        if event["event_id"] in self.seen or self.state in TERMINAL:
            return False
        self.seen.add(event["event_id"])
        data: Json = event["data"]
        if event["type"] == "solve.started":
            if self.state != "queued":
                return False
            self.state = "running"
            limit = timedelta(seconds=self.time_limit_s + self.margin_s)
            self.run_deadline_at = _ts(data["started_at"]) + limit
            return True
        self.state = COMPLETED_STATE[data["status"]]
        return True

    def on_watchdog(self, now: datetime) -> bool:
        overdue_queue = self.state == "queued" and now > self.queue_deadline_at
        overdue_run = (
            self.state == "running"
            and self.run_deadline_at is not None
            and now > self.run_deadline_at
        )
        if overdue_queue or overdue_run:
            self.state = "failed"
            return True
        return False


def _scenarios() -> list[Path]:
    return sorted(LIFECYCLE.glob("*.json")) if LIFECYCLE.is_dir() else []


def test_lifecycle_scenarios_exist() -> None:
    assert len(_scenarios()) >= 12


@pytest.mark.parametrize("path", _scenarios(), ids=lambda p: p.stem)
def test_lifecycle_scenario_is_consistent(path: Path) -> None:
    scenario = _load(path)
    job_spec: Json = scenario["job"]
    job = _Job(
        queue_deadline_at=_ts(job_spec["queue_deadline_at"]),
        time_limit_s=job_spec["time_limit_s"],
        margin_s=scenario["run_deadline_margin_s"],
    )
    ignored: list[int] = []
    for index, step in enumerate(scenario["steps"]):
        changed = (
            job.on_event(step["event"])
            if "event" in step
            else job.on_watchdog(_ts(step["watchdog_at"]))
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
