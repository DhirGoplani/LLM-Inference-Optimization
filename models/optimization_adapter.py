
import os

from models.schema import ModelSpec
from inferbench.optimizer.context import OptimizationContext


def model_spec_to_optimization_context(
    spec: ModelSpec,
) -> OptimizationContext:
    input_shapes = {
        name: tuple(shape)
        for name, shape in spec.input_shapes.items()
    }

    has_dynamic_shapes = any(
        not isinstance(dimension, int) or dimension <= 0
        for shape in input_shapes.values()
        for dimension in shape
    )

    return OptimizationContext(
        model_format="onnx",
        runtime="onnxruntime",
        execution_providers=tuple(spec.compatible_providers),
        logical_cpu_count=max(1, os.cpu_count() or 1),
        input_shapes=input_shapes,
        input_dtypes=dict(spec.input_dtypes),
        has_dynamic_shapes=has_dynamic_shapes,
        metadata={
            "model_path": spec.model_path,
            "input_names": list(spec.input_names),
            "output_names": list(spec.output_names),
            "output_shapes": dict(spec.output_shapes),
            "output_dtypes": dict(spec.output_dtypes),
            "available_providers": list(spec.available_providers),
            "session_creation_providers": list(
                spec.compatible_providers
            ),
            "observed_execution_providers": list(
                spec.observed_execution_providers
            ),
            "provider_compatibility_verified": False,
        },
    )
