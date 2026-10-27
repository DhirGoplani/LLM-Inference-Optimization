"""Precision capability candidates.

This strategy only proposes precision modes when the context explicitly says
the runtime/hardware supports them. Actual conversion or quantization is a
separate transformation and must be implemented by a runtime-specific adapter.
"""

from ..base import OptimizationContext, StrategyCandidate


class PrecisionCapabilityStrategy:
    name = "precision_mode"
    category = "numerical"
    description = "Propose supported precision modes for adapter evaluation."

    def is_applicable(self, context: OptimizationContext):
        return True, None

    def generate_candidates(self, context: OptimizationContext):
        modes = []
        if context.supports_fp16:
            modes.append("fp16")
        if context.supports_bf16:
            modes.append("bf16")
        if context.supports_int8:
            modes.append("int8")
        if context.supports_int4:
            modes.append("int4")

        return [
            StrategyCandidate(
                strategy=self.name,
                config={"precision": mode},
                description=(
                    f"Evaluate {mode} using a compatible runtime transformation; "
                    "candidate generation does not itself convert the model."
                ),
                requires_artifact_transform=(mode in {"int8", "int4"}),
                tags=("precision", "quality-validation-required"),
            )
            for mode in modes
        ]
