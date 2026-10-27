"""Workload and batch-size candidates."""

from ..base import OptimizationContext, StrategyCandidate


class BatchSizeStrategy:
    name = "batch_size"
    category = "workload"
    description = "Explore user-approved batch sizes for fixed-shape or batchable models."

    def __init__(self, batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)):
        self.batch_sizes = batch_sizes

    def is_applicable(self, context: OptimizationContext):
        if context.metadata.get("batching_supported") is False:
            return False, "Model or serving adapter reports batching is unsupported."
        return True, None

    def generate_candidates(self, context: OptimizationContext):
        max_batch = context.metadata.get("max_batch_size")
        sizes = [
            n for n in self.batch_sizes
            if n >= 1 and (max_batch is None or n <= max_batch)
        ]
        return [
            StrategyCandidate(
                strategy=self.name,
                config={"batch_size": n},
                description=f"Benchmark batch size {n}.",
                tags=("workload",),
            )
            for n in sizes
        ]
