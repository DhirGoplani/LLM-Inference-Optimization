"""InferBench's cross-platform optimization strategy library."""

from .base import (
    OptimizationContext,
    OptimizationStrategy,
    StrategyCandidate,
    StrategyResult,
)
from .registry import StrategyRegistry, default_registry
from .benchmarking import benchmark_candidates, config_for_candidate, select_best

__all__ = [
    "OptimizationContext",
    "OptimizationStrategy",
    "StrategyCandidate",
    "StrategyResult",
    "StrategyRegistry",
    "default_registry",
    "benchmark_candidates",
    "config_for_candidate",
    "select_best",
]
