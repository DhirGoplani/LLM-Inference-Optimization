"""Registry and discovery for optimization strategies."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .base import OptimizationContext, OptimizationStrategy, StrategyCandidate
from .strategies.runtime import (
    GraphOptimizationStrategy,
    CPUThreadTuningStrategy,
    ExecutionModeStrategy,
)
from .strategies.batching import BatchSizeStrategy
from .strategies.precision import PrecisionCapabilityStrategy
from .strategies.memory import MemoryStrategy
from .strategies.compilation import CompilationStrategy


class StrategyRegistry:
    """Registers strategies and returns applicable candidates."""

    def __init__(self) -> None:
        self._strategies: dict[str, OptimizationStrategy] = {}

    def register(self, strategy: OptimizationStrategy, *, replace: bool = False) -> None:
        if not strategy.name or not isinstance(strategy.name, str):
            raise ValueError("Strategy must define a non-empty name.")
        if strategy.name in self._strategies and not replace:
            raise ValueError(f"Strategy already registered: {strategy.name}")
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> OptimizationStrategy:
        return self._strategies[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._strategies))

    def describe(self) -> list[dict[str, str]]:
        return [
            {
                "name": s.name,
                "category": s.category,
                "description": s.description,
            }
            for s in (self._strategies[n] for n in self.names())
        ]

    def generate_candidates(
        self,
        context: OptimizationContext,
        strategy_names: Iterable[str] | None = None,
    ) -> tuple[list[StrategyCandidate], dict[str, str]]:
        names = list(strategy_names) if strategy_names is not None else list(self.names())
        candidates: list[StrategyCandidate] = []
        skipped: dict[str, str] = {}
        seen: set[tuple[str, str]] = set()

        for name in names:
            strategy = self.get(name)
            applicable, reason = strategy.is_applicable(context)
            if not applicable:
                skipped[name] = reason or "Not applicable to this context."
                continue

            for candidate in strategy.generate_candidates(context):
                key = (candidate.strategy, repr(sorted(candidate.config.items())))
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)

        return candidates, skipped


def default_registry() -> StrategyRegistry:
    """Create the initial strategy library.

    Strategies generate candidates only. A separate benchmark runner must
    execute and validate them before any candidate can be recommended.
    """
    registry = StrategyRegistry()
    for strategy in (
        GraphOptimizationStrategy(),
        CPUThreadTuningStrategy(),
        ExecutionModeStrategy(),
        BatchSizeStrategy(),
        PrecisionCapabilityStrategy(),
        MemoryStrategy(),
        CompilationStrategy(),
    ):
        registry.register(strategy)
    return registry
