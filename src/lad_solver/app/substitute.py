"""Use case: propose substitutions for one date of a teacher's absence (spec §06, stage 2).

ADR 2026-11-15-solver-substitution-engine. The steps run in order and each one fixes what it
proposed: 1 a free teacher -> 2 another hour of the date -> 3 the neighbour dates of the ISO week,
added one at a time (nearest first, on a tie the later date; spec: "додає сусідні навчальні дні
по одному") until every lesson is placed -> whatever is left is self-study. ``cancelled`` is a
person's choice and is never proposed. There is always a solution (OPTIMAL or FEASIBLE): when the
budget runs out or SIGTERM arrives, the proposals fixed so far stay and the rest is self-study.

The budget is deterministic (one worker, a fixed seed, ``max_deterministic_time`` per run), so the
same request gives the same proposals on any machine: step 1 gets 10 % of
``params.deterministic_time``, step 2 35 %, the step 3 windows share the rest equally, and what a
run leaves unused moves on. ``time_limit_s`` (clamped by the worker to [10, 180] s) is only the
wall-clock safety net. ``allow_support_moves`` is not implemented yet: the engine answers as with
false (contracts README item 9).
"""

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Final

from lad_solver.domain.result import SolveStatus, StopReason
from lad_solver.domain.substitution import (
    InvalidRequestError,
    Proposal,
    ProposalKind,
    Slot,
    SubstitutionParams,
    SubstitutionRequest,
    SubstitutionResult,
    SubstitutionStats,
    Week,
    WeekLesson,
    baseline_week,
)
from lad_solver.model.cpsat import SolverConfig
from lad_solver.model.stop import StopHandle
from lad_solver.model.substitution import Mode, Option, StepOutcome, solve_step

NUM_WORKERS: Final = 1  # deterministic search (ADR): one worker
STEP1_SHARE: Final = 0.10
STEP2_SHARE: Final = 0.35  # step 3 shares the rest
MIN_RUN_DET: Final = 0.01  # floor of a run's deterministic budget: enough to load the hint
MIN_RUN_S: Final = 0.05  # floor of a run's wall-clock limit (the safety net)

_STEP: Final = {Mode.TEACHER: 1, Mode.SAME_DAY: 2, Mode.OTHER_DATE: 3}
_INTERRUPTED: Final = frozenset({SolveStatus.FEASIBLE, SolveStatus.UNKNOWN})  # limit or stop
_ANOMALIES: Final = frozenset({SolveStatus.INFEASIBLE, SolveStatus.MODEL_INVALID})  # a bug


class EngineInvariantError(RuntimeError):
    """The engine proposed a change its own week refuses: a bug, never bad input."""


def substitute(
    request: SubstitutionRequest,
    params: SubstitutionParams,
    stop: StopHandle | None = None,
) -> SubstitutionResult:
    """Steps 1-3 for ``request``; raises ``InvalidRequestError`` if the week is inconsistent."""
    started = time.monotonic()
    job = _Job(request, params, stop, baseline_week(request), started + params.time_limit_s)
    job.run()
    return job.result(time.monotonic() - started)


@dataclass
class _Job:
    request: SubstitutionRequest
    params: SubstitutionParams
    stop: StopHandle | None
    baseline: Week
    deadline: float  # time.monotonic(): the wall-clock safety net
    current: Week = field(init=False)  # the baseline with the proposals fixed so far
    open: list[WeekLesson] = field(init=False)  # lessons without a proposal yet, request order
    fixed: dict[Slot, Proposal] = field(default_factory=dict[Slot, Proposal])
    used: float = 0.0  # deterministic seconds spent by the runs
    runs: int = 0
    anomalies: list[tuple[int, SolveStatus]] = field(default_factory=list[tuple[int, SolveStatus]])
    interrupted: bool = False  # a run hit its limit or the stop, or a step could not run
    last: int = 0  # the last step that ran: the step of a self-study proposal
    covered: list[int] = field(default_factory=lambda: [0, 0, 0])

    def __post_init__(self) -> None:
        self.current = self.baseline
        self.open = [self.baseline.lessons[slot] for slot in self.request.lessons]

    def run(self) -> None:
        budget = self.params.deterministic_time
        last = self.params.stop_after_step
        self._fix(self._solve(Mode.TEACHER, budget * STEP1_SHARE))
        if last >= 2:  # noqa: PLR2004 - step numbers
            self._fix(self._solve(Mode.SAME_DAY, budget * (STEP1_SHARE + STEP2_SHARE) - self.used))
        if last >= 3 and self.request.step3:  # noqa: PLR2004
            self._step3(budget)

    def _step3(self, budget: float) -> None:
        """Add the window's dates one at a time; each run places all open lessons over the
        dates added so far, the nearest preferred. The first run that places them all ends it."""
        if not self.open or not self._may_run():
            return
        self.last = 3  # step 3 runs, also over an empty window (vector 14)
        window = self.request.window()
        best: StepOutcome | None = None
        for k in range(1, len(window) + 1):
            det = (budget - self.used) / (len(window) - k + 1)
            outcome = self._solve(Mode.OTHER_DATE, det, window[:k])
            if outcome is None:
                break  # stopped or out of time: keep the best placement so far
            if best is None or len(outcome.picks) > len(best.picks):
                best = outcome
            if len(best.picks) == len(self.open) or outcome.status is not SolveStatus.OPTIMAL:
                break
        if best is not None:
            self._fix(best)

    def _solve(
        self, mode: Mode, det: float, targets: tuple[dt.date, ...] = ()
    ) -> StepOutcome | None:
        """One step model over the open lessons; ``None`` when it cannot or need not run."""
        if not self.open or not self._may_run():
            return None
        self.last = _STEP[mode]
        config = SolverConfig(
            time_limit_s=max(self.deadline - time.monotonic(), MIN_RUN_S),
            num_workers=NUM_WORKERS,
            random_seed=self.params.random_seed,
            max_deterministic_time=max(det, MIN_RUN_DET),
            max_memory_mb=self.params.max_memory_mb,
        )
        outcome = solve_step(
            self.baseline,
            self.current,
            self.open,
            mode,
            config,
            stop=self.stop,
            targets=targets,
        )
        if outcome.ran:
            self.runs += 1
            self.used += outcome.deterministic_time
        if outcome.status in _INTERRUPTED:
            self.interrupted = True
        if outcome.status in _ANOMALIES:
            self.anomalies.append((_STEP[mode], outcome.status))
        return outcome

    def _fix(self, outcome: StepOutcome | None) -> None:
        if outcome is None or not outcome.picks:
            return
        proposals = [self._proposal(option) for option in outcome.picks]
        for proposal in proposals:
            self.fixed[proposal.slot] = proposal
        step = proposals[0].step
        self.covered[step - 1] += len(proposals)
        try:
            self.current = self.current.with_overlays(p.as_overlay() for p in proposals)
        except InvalidRequestError as exc:
            msg = "the engine proposed a change its week refuses"
            raise EngineInvariantError(msg) from exc
        self.open = [lesson for lesson in self.open if lesson.slot not in self.fixed]

    def _may_run(self) -> bool:
        """False once stopped or out of wall time: the steps left are not run."""
        stopped = self.stop is not None and self.stop.stopped
        if stopped or time.monotonic() >= self.deadline:
            self.interrupted = True
            return False
        return True

    def _proposal(self, option: Option) -> Proposal:
        slot = option.lesson.slot
        if option.teacher_id != option.lesson.teacher_id:  # step 1: another teacher
            return Proposal(
                slot,
                ProposalKind.TEACHER,
                1,
                teacher_id=option.teacher_id,
                off_profile=option.off_profile,
            )
        step = 2 if option.hour.date == slot.date else 3
        return Proposal(slot, ProposalKind.MOVED, step, moved_to=option.hour)

    def result(self, wall_time_s: float) -> SubstitutionResult:
        step = max(1, self.last)  # a stop before step 1 leaves step 1 (the contract's minimum)
        proposals = tuple(
            self.fixed.get(slot) or Proposal(slot, ProposalKind.SELF_STUDY, step)
            for slot in self.request.lessons
        )
        stopped = self.stop is not None and self.stop.stopped
        if not self.interrupted:
            reason = StopReason.COMPLETED
        else:
            reason = StopReason.STOPPED if stopped else StopReason.TIME_LIMIT
        stats = SubstitutionStats(
            wall_time_s=wall_time_s,
            deterministic_time=self.used,
            stop_reason=reason,
            num_workers=NUM_WORKERS,
            random_seed=self.params.random_seed,
            runs=self.runs,
            anomalies=tuple(self.anomalies),
            covered=(self.covered[0], self.covered[1], self.covered[2]),
            absences_completed=self.baseline.absences_completed,
            overlays_ignored=self.baseline.overlays_ignored,
        )
        if reason is StopReason.COMPLETED and not self.anomalies:
            return SubstitutionResult(SolveStatus.OPTIMAL, proposals, stats)
        return SubstitutionResult(SolveStatus.FEASIBLE, proposals, stats)
