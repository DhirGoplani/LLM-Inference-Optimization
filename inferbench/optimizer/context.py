"""Optimization context and candidate models for InferBench AutoTune."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class OptimizationContext:
    """Facts used to decide whether an optimization strategy is applicable."""

    model_format: str = "onnx"
    model_family: str | None = None
    runtime: str = "onnxruntime"
    execution_providers: tuple[str, ...] = ("CPUExecutionProvider",)
    logical_cpu_count: int = field(default_factory=lambda: max(1, os.cpu_count() or 1))
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


@runtime_checkable
class OptimizationStrategy(Protocol):
    """Protocol for optimization strategies."""

    name: str
    category: str
    description: str

    def is_applicable(self, context: OptimizationContext) -> tuple[bool, str | None]:
        """Return (applicable, reason_if_not)."""
        ...

    def generate_candidates(
        self, context: OptimizationContext
    ) -> list[StrategyCandidate]:
        """Generate candidate settings for this context."""
        ...
