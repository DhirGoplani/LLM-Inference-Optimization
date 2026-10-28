"""Strategy registry for registering and discovering optimization strategies."""

from __future__ import annotations

from typing import Iterable

from inferbench.optimizer.context import OptimizationContext, OptimizationStrategy, StrategyCandidate
from inferbench.optimizer.strategies import CPUThreadTuningStrategy, GraphOptimizationStrategy


class StrategyRegistry:
    """Registers strategies and generates applicable candidates."""

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
            if name not in self._strategies:
                skipped[name] = f"Strategy '{name}' is not registered."
                continue
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


def get_tuning_registry() -> StrategyRegistry:
    """Create a strategy registry with executable ONNX Runtime strategies."""
    registry = StrategyRegistry()
    registry.register(GraphOptimizationStrategy())
    registry.register(CPUThreadTuningStrategy())
    return registry
