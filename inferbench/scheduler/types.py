"""Shared types for the request scheduler (Person 4).

Nothing in this module imports onnxruntime or transformers, so the scheduler can
be developed and tested against a fake batch executor.

Priority convention: a HIGHER ``GenerationRequest.priority`` is served first.
Requests with equal priority are served first-in-first-out (the default, since
every request starts at priority 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RequestState(str, Enum):
    QUEUED = "queued"
    BATCHED = "batched"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in TERMINAL_STATES


TERMINAL_STATES = frozenset(
    {
        RequestState.COMPLETED,
        RequestState.CANCELLED,
        RequestState.TIMED_OUT,
        RequestState.FAILED,
    }
)


class SchedulerError(RuntimeError):
    """Base class for every error raised by the scheduler itself."""


class RequestRejectedError(SchedulerError):
    """The request was refused at submission time and never entered the queue."""

    reason = "rejected"


class QueueFullError(RequestRejectedError):
    reason = "queue_full"


class PromptTooLongError(RequestRejectedError):
    reason = "prompt_too_long"


class GenerationTooLongError(RequestRejectedError):
    reason = "generation_too_long"


class DuplicateRequestError(RequestRejectedError):
    reason = "duplicate_request_id"


class DeadlineExpiredError(RequestRejectedError):
    reason = "deadline_expired"


class SchedulerClosedError(RequestRejectedError):
    reason = "scheduler_closed"


class RequestCancelledError(SchedulerError):
    """The request was cancelled before it produced a result."""


class RequestTimeoutError(SchedulerError):
    """The request's timeout or deadline expired before it produced a result."""


@dataclass(frozen=True)
class SchedulerConfig:
    """Limits and policies for :class:`inferbench.scheduler.RequestScheduler`.

    max_queue_length: waiting requests allowed; more are rejected (overload).
    max_prompt_tokens: reject prompts longer than this (None = no limit).
    max_new_tokens: reject requests asking for more generated tokens (None = no limit).
    default_timeout_s: seconds a request may live (queue wait + execution) when it
        sets no ``deadline_s`` of its own (None = no timeout).
    max_batch_size / max_batch_tokens: most requests / prompt tokens handed to the
        batch executor in one call. ``max_batch_size=1`` gives single-request serving.
    batching_window_ms: how long the worker waits for more requests to arrive
        before dispatching a partly filled batch.
    aging_interval_s: every ``aging_interval_s`` of waiting adds one priority level,
        so low-priority requests cannot starve (None disables aging).
    """

    max_queue_length: int = 64
    max_prompt_tokens: int | None = None
    max_new_tokens: int | None = None
    default_timeout_s: float | None = None
    max_batch_size: int = 8
    max_batch_tokens: int = 8192
    batching_window_ms: float = 5.0
    aging_interval_s: float | None = 5.0

    def __post_init__(self) -> None:
        if self.max_queue_length <= 0:
            raise ValueError("max_queue_length must be positive")
        if self.max_prompt_tokens is not None and self.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        if self.max_new_tokens is not None and self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")
        if self.batching_window_ms < 0:
            raise ValueError("batching_window_ms must be non-negative")
        if self.aging_interval_s is not None and self.aging_interval_s <= 0:
            raise ValueError("aging_interval_s must be positive")