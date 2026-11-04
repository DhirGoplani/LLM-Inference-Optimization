
import pytest

from models.provider_diagnostics import diagnose_execution_providers


def test_provider_diagnostic_mobilenet():
    result = diagnose_execution_providers(
        "models/mobilenet_v2.onnx"
    )

    assert result["available_providers"]
    assert result["results"]

    for item in result["results"]:
        assert "session_created" in item
        assert "inference_succeeded" in item
        assert "observed_node_providers" in item


def test_provider_diagnostic_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        diagnose_execution_providers(
            str(tmp_path / "missing.onnx")
        )


def test_provider_diagnostic_rejects_non_onnx(tmp_path):
    path = tmp_path / "model.txt"
    path.write_text("not an ONNX model")

    with pytest.raises(ValueError, match="Expected an ONNX"):
        diagnose_execution_providers(str(path))
