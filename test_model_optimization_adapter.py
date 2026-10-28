
from models.loader import load_model
from models.optimization_adapter import model_spec_to_optimization_context


def test_model_spec_to_optimization_context():
    spec = load_model("models/mobilenet_v2.onnx")
    context = model_spec_to_optimization_context(spec)

    assert context.model_format == "onnx"
    assert context.runtime == "onnxruntime"
    assert context.input_shapes["input"] == (1, 3, 224, 224)
    assert context.input_dtypes["input"] == "tensor(float)"
    assert context.has_dynamic_shapes is False
    assert context.logical_cpu_count >= 1
    assert "CPUExecutionProvider" in context.execution_providers
    assert context.metadata["session_creation_providers"] == (
        spec.compatible_providers
    )
    assert context.metadata["observed_execution_providers"] == (
        spec.observed_execution_providers
    )
    assert context.metadata["provider_compatibility_verified"] is False


def test_dynamic_shape_detection():
    from dataclasses import replace

    spec = load_model("models/mobilenet_v2.onnx")
    spec.input_shapes["input"] = ["batch_size", 3, 224, 224]

    context = model_spec_to_optimization_context(spec)

    assert context.has_dynamic_shapes is True
