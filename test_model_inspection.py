
import pytest

from models.inspection import inspect_model_configuration


def test_inspect_mobilenet():
    result = inspect_model_configuration("models/mobilenet_v2.onnx")

    assert result["model_path"].endswith("mobilenet_v2.onnx")
    assert result["opsets"]
    assert result["operators"]
    assert result["runtime_status"] == "session_created"
    assert result["inputs"]
    assert result["outputs"]


def test_missing_model_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_model_configuration(
            str(tmp_path / "missing.onnx")
        )


def test_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        inspect_model_configuration(str(tmp_path))


def test_non_onnx_file_is_rejected(tmp_path):
    path = tmp_path / "model.txt"
    path.write_text("not an ONNX model")

    with pytest.raises(ValueError, match="Expected an ONNX"):
        inspect_model_configuration(str(path))

