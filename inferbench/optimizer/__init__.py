"""InferBench Optimizer module for AutoTune candidate generation and evaluation."""

from inferbench.optimizer.adapter import (
    BASELINE_NUM_THREADS,
    benchmark_candidates,
    candidate_to_benchmark_config,
    run_autotune,
    select_best,
)
from inferbench.optimizer.context import OptimizationContext, StrategyCandidate
from inferbench.optimizer.registry import StrategyRegistry, get_tuning_registry
from inferbench.optimizer.strategies import (
    CPUThreadTuningStrategy,
    GraphOptimizationStrategy,
    QuantizationStrategy,
)

__all__ = [
    "BASELINE_NUM_THREADS",
    "OptimizationContext",
    "StrategyCandidate",
    "GraphOptimizationStrategy",
    "CPUThreadTuningStrategy",
    "QuantizationStrategy",
    "StrategyRegistry",
    "get_tuning_registry",
    "candidate_to_benchmark_config",
    "benchmark_candidates",
    "select_best",
    "run_autotune",
]