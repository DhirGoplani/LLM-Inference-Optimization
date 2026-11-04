
import pytest

from models.loader import load_model


def test_missing_model_file():
    with pytest.raises(FileNotFoundError):
        load_model("models/does_not_exist.onnx")


def test_directory_is_rejected():
    with pytest.raises(ValueError):
        load_model("models")


def test_non_onnx_file_is_rejected():
    with pytest.raises(ValueError):
        load_model("README.md")
