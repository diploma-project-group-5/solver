"""Small ports between the worker loop and Kafka, so the loop is unit-testable without a broker."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from lad_solver.domain.problem import Params, Problem
from lad_solver.domain.result import Result
from lad_solver.model.progress import Progress
from lad_solver.model.stop import StopHandle


class TransientKafkaError(Exception):
    """A produce/flush/commit failed in a way that may succeed on retry."""


class FatalKafkaError(Exception):
    """The client is unusable (e.g. fatal idempotent-producer error); the process must exit."""


@dataclass(frozen=True, slots=True)
class InboundMessage:
    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes | None
    handle: object = field(default=None, compare=False, repr=False)  # client message, for commit


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    topic: str
    key: bytes | None
    value: bytes | None
    headers: tuple[tuple[str, bytes], ...] = ()


class ConsumerPort(Protocol):
    def poll(self, timeout_s: float) -> InboundMessage | None:
        """A message, or None when idle; raises TransientKafkaError if the poll only saw errors."""
        ...

    def commit(self, message: InboundMessage) -> None:
        """Synchronous commit of ``message.offset + 1``; raises TransientKafkaError on failure."""
        ...

    def close(self) -> None: ...


class ProducerPort(Protocol):
    def send(self, messages: list[OutboundMessage], timeout_s: float) -> None:
        """Produce and flush; returns only when every message has a successful delivery report.

        Raises TransientKafkaError if anything is undelivered within ``timeout_s``. Calling it
        again with the same messages re-produces only those that got a delivery *error*; messages
        still awaiting a report are flushed again, not duplicated.
        """
        ...

    def poll(self, timeout_s: float) -> None:
        """Serve delivery and error callbacks (called while a solve runs)."""
        ...

    def close(self, timeout_s: float) -> int:
        """Flush and return how many messages are still undelivered."""
        ...


# progress: phase A telemetry + the no-improvement stop, ticked by the worker's wait loop.
type Engine = Callable[[Problem, Params, StopHandle, Progress], Result]
