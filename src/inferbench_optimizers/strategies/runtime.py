"""Runtime and CPU execution configuration strategies."""

from __future__ import annotations

import os

from ..base import OptimizationContext, StrategyCandidate


class GraphOptimizationStrategy:
    name = "graph_optimization"
    category = "graph"
    description = "Compare ONNX Runtime graph optimization levels."

    LEVELS = (
        "ORT_DISABLE_ALL",
        "ORT_ENABLE_BASIC",
        "ORT_ENABLE_EXTENDED",
        "ORT_ENABLE_ALL",
    )

    def is_applicable(self, context: OptimizationContext):
        if context.runtime.lower() != "onnxruntime" or context.model_format.lower() != "onnx":
            return False, "Currently supported for ONNX models running on ONNX Runtime."
        return True, None

    def generate_candidates(self, context: OptimizationContext):
        return [
            StrategyCandidate(
                strategy=self.name,
                config={"graph_optimization": level},
                description=f"Use graph optimization level {level}.",
                tags=("runtime", "graph"),
            )
            for level in self.LEVELS
        ]


class CPUThreadTuningStrategy:
    name = "cpu_thread_tuning"
    category = "runtime"
    description = "Try conservative ONNX Runtime CPU thread counts."

    def __init__(self, thread_counts: tuple[int, ...] | None = None):
        self.thread_counts = thread_counts

    def is_applicable(self, context: OptimizationContext):
        if not any("CPUExecutionProvider" in p for p in context.execution_providers):
            return False, "CPUExecutionProvider is not available."
        if context.logical_cpu_count < 1:
            return False, "Logical CPU count is invalid."
        return True, None

    def generate_candidates(self, context: OptimizationContext):
        core_count = max(1, context.logical_cpu_count or os.cpu_count() or 1)
        counts = self.thread_counts or (1, 2, 4, 8, core_count)
        counts = sorted({n for n in counts if isinstance(n, int) and 1 <= n <= core_count})
        return [
            StrategyCandidate(
                strategy=self.name,
                config={"intra_op_threads": count, "inter_op_threads": 1},
                description=f"Use {count} intra-op CPU thread(s).",
                tags=("cpu", "runtime"),
            )
            for count in counts
        ]


class ExecutionModeStrategy:
    name = "execution_mode"
    category = "runtime"
    description = "Compare sequential and parallel graph execution modes."

    def is_applicable(self, context: OptimizationContext):
        if context.runtime.lower() != "onnxruntime" or context.model_format.lower() != "onnx":
            return False, "Currently supported for ONNX Runtime with ONNX models."
        return True, None

    def generate_candidates(self, context: OptimizationContext):
        return [
            StrategyCandidate(
                strategy=self.name,
                config={"execution_mode": mode},
                description=f"Use {mode} execution.",
                tags=("runtime", "parallelism"),
            )
            for mode in ("ORT_SEQUENTIAL", "ORT_PARALLEL")
        ]
