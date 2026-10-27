"""Shared contracts for optimization strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class OptimizationContext:
    """Facts used to decide whether a strategy is applicable.

    These fields are intentionally generic so the library can support
    different model formats, runtimes, and hardware backends.
    """

    model_format: str = "onnx"
    model_family: str | None = None
    runtime: str = "onnxruntime"
    execution_providers: tuple[str, ...] = ("CPUExecutionProvider",)
    logical_cpu_count: int = 1
    available_memory_mb: float | None = None
    input_shapes: dict[str, tuple[int | str | None, ...]] = field(default_factory=dict)
    input_dtypes: dict[str, str] = field(default_factory=dict)
    has_dynamic_shapes: bool = False
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_int8: bool = False
    supports_int4: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyCandidate:
    """One proposed configuration or transformation."""

    strategy: str
    config: dict[str, Any]
    description: str
    requires_artifact_transform: bool = False
    tags: tuple[str, ...] = ()


@dataclass
class StrategyResult:
    """Outcome after an optimization candidate has been evaluated."""

    candidate: StrategyCandidate
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    reason: str | None = None
    artifact_path: str | None = None


@runtime_checkable
class OptimizationStrategy(Protocol):
    """Contract every optimization strategy must implement."""

    name: str
    category: str
    description: str

    def is_applicable(self, context: OptimizationContext) -> tuple[bool, str | None]:
        """Return (applicable, reason_if_not). Must not execute the model."""
        ...

    def generate_candidates(
        self, context: OptimizationContext
    ) -> list[StrategyCandidate]:
        """Generate candidate settings for this context."""
        ...
