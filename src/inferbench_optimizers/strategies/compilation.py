"""Compilation strategy placeholder with explicit support gating."""

from ..base import OptimizationContext, StrategyCandidate


class CompilationStrategy:
    name = "compilation"
    category = "compiler"
    description = "Reserve a common strategy slot for runtime-specific compilation."

    def is_applicable(self, context: OptimizationContext):
        # ONNX Runtime graph optimization is handled separately. We do not
        # claim arbitrary ONNX compilation is available through every runtime.
        if context.metadata.get("compiler_adapter") is None:
            return False, "No compatible compiler adapter was registered for this environment."
        return True, None

    def generate_candidates(self, context: OptimizationContext):
        adapter = context.metadata["compiler_adapter"]
        options = adapter.get("candidate_options", [])
        return [
            StrategyCandidate(
                strategy=self.name,
                config={"compiler": adapter.get("name"), **dict(option)},
                description="Compiler candidate; must be executed by the registered adapter.",
                requires_artifact_transform=True,
                tags=("compiler",),
            )
            for option in options
        ]
