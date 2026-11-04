
from models.loader import load_model


def test_load_mobilenet():
    spec = load_model("models/mobilenet_v2.onnx")

    assert spec.input_names
    assert spec.output_names
    assert spec.input_shapes
    assert spec.output_shapes
    assert spec.input_dtypes
    assert spec.output_dtypes
    assert "CPUExecutionProvider" in spec.available_providers
    assert "CPUExecutionProvider" in spec.compatible_providers

    print(spec)


if __name__ == "__main__":
    test_load_mobilenet()
