"""The worker loop (ADR 2026-11-04 §4), written against ports so it is unit-testable.

Per message: poll -> parse ->
  * poison (no school_id / job_id)         -> DLQ copy (best effort) -> commit
  * rejected (invalid data, unknown MAJOR) -> solve.completed MODEL_INVALID -> DLQ copy -> commit
  * expired (queue_deadline_at passed)     -> solve.completed EXPIRED (no started) -> commit
  * accepted -> solve.started -> flush + delivery check -> commit -> solve -> solve.completed
A stop requested before ``solve.started`` leaves the message uncommitted (redelivered later).
The DLQ is diagnostics only (events-kafka rules): a failed DLQ copy is logged, never retried.

The offset is committed *before* solving, so a solve that kills the process (OOM, kill -9) is
never redelivered; timetable's watchdog closes the job by run_deadline_at (spec §12).
Only transient produce/commit failures are retried (1/2/4 s); a solve is never retried.
"""

import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging import ERROR, INFO, WARNING
from typing import Final

from lad_solver import __version__
from lad_solver.adapters.kafka import jsonlog
from lad_solver.adapters.kafka.config import WorkerConfig
from lad_solver.adapters.kafka.mapping import (
    COMPLETED,
    MODEL_VERSION,
    STARTED,
    Accepted,
    Address,
    Completed,
    ErrorCode,
    Poison,
    Provenance,
    Rejected,
    address_of,
    completed_data,
    completed_message,
    dlq_message,
    params_for,
    parse_build_request,
    rejected_data,
    started_message,
)
from lad_solver.adapters.kafka.ports import (
    ConsumerPort,
    Engine,
    FatalKafkaError,
    InboundMessage,
    OutboundMessage,
    ProducerPort,
    TransientKafkaError,
)
from lad_solver.domain.problem import Params, Problem
from lad_solver.domain.result import ConflictRule, Result, SolveStatus
from lad_solver.model.progress import Progress
from lad_solver.model.stop import StopHandle

RETRY_DELAYS_S: Final = (1.0, 2.0, 4.0)
POLL_TIMEOUT_S: Final = 1.0
WAIT_SLICE_S: Final = 0.5
# Consecutive failed consumer polls after which the heartbeat file goes stale (unhealthy).
MAX_CONSUMER_ERRORS: Final = 30

EXIT_OK: Final = 0
EXIT_FAILURE: Final = 1


class WorkerFatalError(Exception):
    """Stop the process with exit code 1 (compose restarts it)."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Worker:
    consumer: ConsumerPort
    producer: ProducerPort
    engine: Engine
    config: WorkerConfig
    clock: Callable[[], datetime] = _utc_now
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    hard_exit: Callable[[int], None] = os._exit
    heartbeat: Callable[[], None] | None = None
    _stop_requested: bool = field(default=False, init=False)
    _stop_signal: str = field(default="SIGTERM", init=False)
    _stop_logged: bool = field(default=False, init=False)
    _consumer_errors: int = field(default=0, init=False)
    _current: Address | None = field(default=None, init=False)  # job being handled
    _last: Address | None = field(default=None, init=False)  # last job handled
    _solving: Future[Result] | None = field(default=None, init=False)
    _received_at: float = field(default=0.0, init=False)  # monotonic, when handle() began

    # ------------------------------------------------------------ lifecycle

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def request_stop(self, signal_name: str = "SIGTERM") -> None:
        """Signal-handler safe: only sets flags.

        No logging and no ``stop_search`` here: CpSolver.stop_search takes a non-reentrant lock,
        so a second signal arriving inside it would deadlock. The main loop (``_wait`` every
        ``WAIT_SLICE_S``) logs the stop and interrupts the search.
        """
        if not self._stop_requested:
            self._stop_signal = signal_name
        self._stop_requested = True

    def _stopping(self) -> bool:
        """Main-thread view of the stop flag; logs the stop once."""
        if self._stop_requested and not self._stop_logged:
            self._stop_logged = True
            jsonlog.log(INFO, "worker.stopping", signal=self._stop_signal)
        return self._stop_requested

    def run(self) -> int:
        code = EXIT_OK
        try:
            while not self._stopping():
                if self._consumer_errors < MAX_CONSUMER_ERRORS:
                    self._beat()
                message = self._poll()
                # A stop that arrived while polling: leave the message uncommitted.
                if message is not None and not self._stopping():
                    self.handle(message)
        except WorkerFatalError:
            code = EXIT_FAILURE
        except FatalKafkaError as exc:
            jsonlog.log(ERROR, "kafka.fatal", **_ids(self._current), error_class=type(exc).__name__)
            code = EXIT_FAILURE
        except Exception as exc:  # noqa: BLE001 -- logged without its text; exit 1 restarts us
            jsonlog.log(
                ERROR,
                "worker.crashed",
                **_ids(self._current),
                error_class=type(exc).__name__,
                where=jsonlog.where(exc),
            )
            code = EXIT_FAILURE
        finally:
            self._close()
            self._abandon_running_solve()
        return code

    def _abandon_running_solve(self) -> None:
        """A CP-SAT search still running in its thread would block a normal interpreter exit."""
        if self._solving is not None and not self._solving.done():
            jsonlog.log(ERROR, "engine.abandoned", **_ids(self._current))
            self.hard_exit(EXIT_FAILURE)

    def _poll(self) -> InboundMessage | None:
        try:
            message = self.consumer.poll(POLL_TIMEOUT_S)
        except TransientKafkaError as exc:
            self._consumer_errors += 1
            if self._consumer_errors == MAX_CONSUMER_ERRORS:
                jsonlog.log(
                    ERROR,
                    "consumer.unhealthy",
                    attempt=self._consumer_errors,
                    error_class=type(exc).__name__,
                    reason=_reason(exc),
                )
            return None
        self._consumer_errors = 0
        return message

    def _close(self) -> None:
        timeout_s = self._flush_timeout(self.config.completed_flush_timeout_s)
        try:
            self.consumer.close()
        except Exception as exc:  # noqa: BLE001 -- shutting down; log class only, keep closing
            jsonlog.log(WARNING, "consumer.close_failed", error_class=type(exc).__name__)
        try:
            undelivered = self.producer.close(timeout_s)
        except Exception as exc:  # noqa: BLE001 -- shutting down; log class only
            jsonlog.log(WARNING, "producer.close_failed", error_class=type(exc).__name__)
            return
        if undelivered > 0:
            jsonlog.log(
                ERROR,
                "producer.undelivered",
                **_ids(self._current or self._last),
                count=undelivered,
            )

    def _beat(self) -> None:
        if self.heartbeat is not None:
            self.heartbeat()

    def _flush_timeout(self, timeout_s: float) -> float:
        """Once stopping, flushes are capped so the stop fits compose's stop_grace_period."""
        if self._stop_requested:
            return min(timeout_s, self.config.stop_flush_timeout_s)
        return timeout_s

    # ------------------------------------------------------------ one message

    def handle(self, message: InboundMessage) -> None:
        self._received_at = self.monotonic()
        try:
            outcome = parse_build_request(message.value)
        except Exception as exc:  # noqa: BLE001 -- a parser bug must not crash-loop on one message
            outcome = self._parser_failed(message, exc)
        self._current = None if isinstance(outcome, Poison) else outcome.address
        match outcome:
            case Poison(error_class=error_class):
                self._on_poison(message, error_class)
            case Rejected():
                self._on_rejected(message, outcome)
            case Accepted():
                now = self.clock()
                if outcome.queue_deadline_at <= now:
                    self._on_expired(message, outcome, now)
                else:
                    self._on_accepted(message, outcome, now)
        self._last, self._current = self._current or self._last, None

    def _parser_failed(self, message: InboundMessage, exc: Exception) -> Poison | Rejected:
        address = address_of(message.value)
        error_class = f"parser_{type(exc).__name__}"
        jsonlog.log(
            ERROR,
            "parser.failed",
            **_ids(address),
            topic=message.topic,
            partition=message.partition,
            offset=message.offset,
            error_class=type(exc).__name__,
            where=jsonlog.where(exc),
        )
        if address is None:
            return Poison(error_class)
        return Rejected(address, ErrorCode.INTERNAL, error_class)

    def _on_poison(self, message: InboundMessage, error_class: str) -> None:
        jsonlog.log(
            WARNING,
            "message.poison",
            topic=message.topic,
            partition=message.partition,
            offset=message.offset,
            error_class=error_class,
            size_bytes=len(message.value or b""),
        )
        self._send_dlq(message, error_class, None)
        self._commit(message, None)
        jsonlog.log(
            INFO,
            "message.done",
            topic=message.topic,
            partition=message.partition,
            offset=message.offset,
            error_class=error_class,
            handle_ms=self._handle_ms(),
        )

    def _on_rejected(self, message: InboundMessage, outcome: Rejected) -> None:
        address = outcome.address
        completed = self._completed(address, rejected_data(address.job_id, outcome.code))
        self._send_completed(completed, address)
        self._send_dlq(message, outcome.error_class, address)
        self._commit(message, address)
        self._log_completed(address, completed, error_class=outcome.error_class)

    def _on_expired(self, message: InboundMessage, outcome: Accepted, now: datetime) -> None:
        address = outcome.address
        data = completed_data(address.job_id, Result(status=SolveStatus.EXPIRED), None)
        completed = self._completed(address, data)
        self._send_completed(completed, address)
        self._commit(message, address)
        self._log_completed(address, completed, queue_wait_ms=_ms_between(outcome, now))

    def _on_accepted(self, message: InboundMessage, outcome: Accepted, now: datetime) -> None:
        if self._stopping():  # nothing promised yet: leave it uncommitted for the next start
            return
        address = outcome.address
        config = self.config
        started = started_message(address, now, config.status_topic)
        self._send([started], config.started_flush_timeout_s, STARTED, address)
        self._commit(message, address)

        params = params_for(
            outcome.data,
            min_time_limit_s=config.min_time_limit_s,
            max_time_limit_s=config.max_time_limit_s,
            num_workers=config.num_workers,
            seed=config.seed,
            max_memory_mb=config.max_memory_mb,
            stall_min_s=config.stall_min_s,
            stall_ratio=config.stall_ratio,
        )
        jsonlog.log(
            INFO,
            "job.started",
            **_ids(address),
            queue_wait_ms=_ms_between(outcome, now),
            time_limit_s=params.time_limit_s,
            num_workers=params.num_workers,
            stall_min_s=params.stall_min_s,
        )
        began = self.monotonic()
        result, error_code = self._solve(address, outcome.problem, params)
        wall_time_ms = round((self.monotonic() - began) * 1000)
        if error_code is None:
            provenance = Provenance(wall_time_ms=wall_time_ms, input_sha256=outcome.input_sha256)
            data = completed_data(address.job_id, result, provenance)
            model_invalid_code = self._model_invalid_code(address, result)
            if model_invalid_code is not None:
                data["error"] = {"code": model_invalid_code.value}
        else:
            data = rejected_data(address.job_id, error_code)
        completed = self._completed(address, data)
        self._send_completed(completed, address)
        if error_code is not None:  # the engine failed: there is no search telemetry to report
            self._log_completed(address, completed, wall_time_ms=wall_time_ms)
            return
        stats = result.stats
        self._log_completed(
            address,
            completed,
            wall_time_ms=wall_time_ms,
            first_solution_ms=_ms(stats.first_solution_s),
            last_improvement_ms=_ms(stats.last_improvement_s),
            objective=stats.objective,
            best_bound=stats.best_bound,
            gap=_gap(stats.objective, stats.best_bound),
            stop_reason=stats.stop_reason.value,
            quality_relaxed=stats.quality_relaxed,
            phase_a_status=stats.phase_a_status,
            num_workers=stats.num_workers,
        )

    @staticmethod
    def _model_invalid_code(address: Address, result: Result) -> ErrorCode | None:
        """Every MODEL_INVALID carries error.code: invalid input, or an engine bug (internal)."""
        if result.status is not SolveStatus.MODEL_INVALID:
            return None
        if any(c.rule is ConflictRule.INVALID_INPUT for c in result.conflicts):
            return ErrorCode.INVALID_PAYLOAD
        jsonlog.log(
            ERROR,
            "engine.model_invalid",
            **_ids(address),
            phase_a_status=result.stats.phase_a_status,
        )
        return ErrorCode.INTERNAL

    # ------------------------------------------------------------ solve

    def _solve(
        self, address: Address, problem: Problem, params: Params
    ) -> tuple[Result, ErrorCode | None]:
        stop = StopHandle()
        progress = Progress()
        if self._stop_requested:  # the signal arrived before the handle existed
            stop.stop()
        future: Future[Result] = Future()

        def target() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(self.engine(problem, params, stop, progress))
            except BaseException as exc:  # noqa: BLE001 -- handed over to the main thread
                future.set_exception(exc)

        # A daemon thread instead of ThreadPoolExecutor: executor threads are joined at
        # interpreter exit, so a stuck CP-SAT search would block a fatal exit.
        solver_thread = threading.Thread(target=target, name="lad-solve", daemon=True)
        watchdog = threading.Timer(
            params.time_limit_s + self.config.watchdog_margin_s, self._watchdog, args=(address,)
        )
        watchdog.daemon = True
        self._solving = future
        solver_thread.start()
        watchdog.start()
        try:
            self._wait(future, stop, progress, address)
        finally:
            watchdog.cancel()
            if not future.done():  # leaving on an exception: do not leave the search running
                stop.stop()
        exc = future.exception()
        if exc is not None:
            jsonlog.log(
                ERROR,
                "engine.failed",
                **_ids(address),
                error_class=type(exc).__name__,
                where=jsonlog.where(exc),
            )
            return Result(status=SolveStatus.MODEL_INVALID), ErrorCode.INTERNAL
        return future.result(), None

    def _wait(
        self, future: Future[Result], stop: StopHandle, progress: Progress, address: Address
    ) -> None:
        stop_deadline: float | None = None
        stall_check = True
        while not future.done():
            wait([future], timeout=WAIT_SLICE_S)
            self.producer.poll(0)
            self._beat()
            if stall_check:
                stall_check = self._check_progress(progress, address)
            if self._stopping():
                stop.stop()
                if stop_deadline is None:
                    stop_deadline = self.monotonic() + self.config.stop_grace_s
                elif self.monotonic() > stop_deadline and not future.done():
                    jsonlog.log(ERROR, "engine.ignored_stop", **_ids(address))
                    raise WorkerFatalError

    @staticmethod
    def _check_progress(progress: Progress, address: Address) -> bool:
        """The no-improvement stop is advisory: a failure turns it off for this job only."""
        try:
            progress.check()
        except Exception as exc:  # noqa: BLE001 -- must not fail the job or skip SIGTERM handling
            jsonlog.log(
                ERROR,
                "progress.check_failed",
                **_ids(address),
                error_class=type(exc).__name__,
                where=jsonlog.where(exc),
            )
            return False
        return True

    def _watchdog(self, address: Address) -> None:
        try:
            jsonlog.log(ERROR, "engine.watchdog", **_ids(address))
        finally:
            self.hard_exit(EXIT_FAILURE)

    # ------------------------------------------------------------ delivery

    def _completed(self, address: Address, data: dict[str, object]) -> Completed:
        return completed_message(
            address,
            data,
            now=self.clock(),
            topic=self.config.status_topic,
            max_bytes=self.config.result_max_bytes,
        )

    def _retrying(self, action: Callable[[], None], what: str, address: Address | None) -> None:
        ids = _ids(address)
        for attempt, delay in enumerate((*RETRY_DELAYS_S, None), start=1):
            try:
                action()
            except TransientKafkaError as exc:
                # Once stopping there is no time budget for retries (compose stop_grace_period),
                # also when the stop arrives between two attempts.
                if delay is None or self._stop_requested:
                    jsonlog.log(
                        ERROR,
                        f"{what}.gave_up",
                        **ids,
                        attempt=attempt,
                        error_class=type(exc).__name__,
                        reason=_reason(exc),
                    )
                    raise WorkerFatalError from None
                jsonlog.log(
                    WARNING,
                    f"{what}.retry",
                    **ids,
                    attempt=attempt,
                    error_class=type(exc).__name__,
                    reason=_reason(exc),
                )
                self._beat()  # retries can take 7 s; stay healthy meanwhile
                self.sleep(delay)
            else:
                return

    def _send(
        self,
        messages: Sequence[OutboundMessage],
        timeout_s: float,
        what: str,
        address: Address | None,
    ) -> None:
        batch = list(messages)
        self._retrying(
            lambda: self.producer.send(batch, self._flush_timeout(timeout_s)), what, address
        )

    def _send_completed(self, completed: Completed, address: Address) -> None:
        self._send([completed.message], self.config.completed_flush_timeout_s, COMPLETED, address)

    def _send_dlq(self, message: InboundMessage, error_class: str, address: Address | None) -> None:
        """Best effort: the DLQ is diagnostics only; one attempt, then log and move on.

        Any failure, including one while building the copy, is logged so the offset is still
        committed: otherwise one bad message would crash-loop the worker.
        """
        try:
            dlq = dlq_message(message, topic=self.config.build_dlq_topic, error_class=error_class)
            self.producer.send([dlq], self._flush_timeout(self.config.started_flush_timeout_s))
        except FatalKafkaError:
            raise  # the client is unusable (error_cb during flush): exit 1, not "best effort"
        except Exception as exc:  # noqa: BLE001 -- best effort; class only, never the payload
            jsonlog.log(
                ERROR,
                "dlq.failed",
                **_ids(address),
                topic=message.topic,
                partition=message.partition,
                offset=message.offset,
                error_class=type(exc).__name__,
            )

    def _commit(self, message: InboundMessage, address: Address | None) -> None:
        self._retrying(lambda: self.consumer.commit(message), "commit", address)

    def _log_completed(self, address: Address, completed: Completed, **fields: object) -> None:
        jsonlog.log(
            INFO,
            "job.completed",
            **_ids(address),
            status=completed.status,
            error_code=completed.error_code.value if completed.error_code else None,
            size_bytes=len(completed.message.value or b""),
            solver_version=__version__,
            model_version=MODEL_VERSION,
            handle_ms=self._handle_ms(),
            **fields,
        )

    def _handle_ms(self) -> int:
        """Processing time of the current message: received -> answer sent (and committed)."""
        return round((self.monotonic() - self._received_at) * 1000)


def _ids(address: Address | None) -> dict[str, object]:
    if address is None:
        return {}
    return {
        "job_id": str(address.job_id),
        "school_id": str(address.school_id),
        "correlation_id": address.correlation_id,
    }


def _reason(exc: TransientKafkaError) -> str | None:
    """The librdkafka error name (or fixed literal) that kafka_io put in the exception."""
    return str(exc) or None


def _ms(seconds: float | None) -> int | None:
    return None if seconds is None else round(seconds * 1000)


def _gap(objective: int | None, bound: float | None) -> float | None:
    """(bound - objective) / bound of phase A; the CP-SAT bound is weak, so read it with care."""
    if objective is None or not bound:
        return None
    return round(max(0.0, (bound - objective) / bound), 4)


def _ms_between(outcome: Accepted, now: datetime) -> int | None:
    if outcome.requested_at is None or outcome.requested_at.tzinfo is None:
        return None
    return round((now - outcome.requested_at).total_seconds() * 1000)
