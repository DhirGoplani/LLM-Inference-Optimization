"""InferBench's cross-platform optimization strategy library."""

from .base import (
    OptimizationContext,
    OptimizationStrategy,
    StrategyCandidate,
    StrategyResult,
)
from .registry import StrategyRegistry, default_registry

__all__ = [
    "OptimizationContext",
    "OptimizationStrategy",
    "StrategyCandidate",
    "StrategyResult",
    "StrategyRegistry",
    "default_registry",
]
