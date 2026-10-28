"""Memory-related optimization candidates."""

from ..base import OptimizationContext, StrategyCandidate


class MemoryStrategy:
    name = "memory_optimization"
    category = "memory"
    description = "Propose runtime memory-pattern and memory-arena experiments."

    def is_applicable(self, context: OptimizationContext):
        if context.runtime.lower() != "onnxruntime" or context.model_format.lower() != "onnx":
            return False, "Initial memory candidates target ONNX Runtime."
        return True, None

    def generate_candidates(self, context: OptimizationContext):
        # These are runtime session options, not guaranteed improvements.
        return [
            StrategyCandidate(
                strategy=self.name,
                config={"enable_mem_pattern": enabled},
                description=f"Set ONNX Runtime memory pattern to {enabled}.",
                tags=("memory", "runtime"),
            )
            for enabled in (True, False)
        ]
