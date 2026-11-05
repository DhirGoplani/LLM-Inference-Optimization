
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import onnxruntime as ort

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
        assert "provider_execution_status" in item

    cpu_result = next(
        item
        for item in result["results"]
        if item["provider"] == "CPUExecutionProvider"
    )
    assert cpu_result["session_created"]
    assert cpu_result["inference_succeeded"]
    assert cpu_result["outputs"]


def test_provider_diagnostic_qwen_decoder_outputs(monkeypatch):
    model_path = Path("models/qwen2.5_0.5b/model.onnx")
    if not model_path.is_file():
        pytest.skip("Qwen decoder model is not available")

    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    result = diagnose_execution_providers(str(model_path))
    diagnostic = result["results"][0]

    assert diagnostic["session_created"]
    assert diagnostic["inference_succeeded"]
    assert diagnostic["input_shapes"]["past_key_values.0.key"] == [
        1,
        2,
        0,
        64,
    ]
    assert diagnostic["input_shapes"]["attention_mask"] == [1, 2]
    assert diagnostic["input_dtypes"]["past_key_values.0.key"] == "float16"

    outputs = {item["name"]: item for item in diagnostic["outputs"]}
    assert outputs["logits"]["shape"] == [1, 2, 151936]
    assert outputs["logits"]["dtype"] == "float16"
    assert outputs["present.0.key"]["shape"] == [1, 2, 2, 64]
    assert outputs["present.0.value"]["shape"] == [1, 2, 2, 64]


def test_provider_diagnostic_reports_inference_error(monkeypatch, tmp_path):
    class FakeSession:
        def __init__(self, _model_path, sess_options, providers):
            self.profile_prefix = sess_options.profile_file_prefix
            self.providers = providers

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="input",
                    shape=[1, 3],
                    type="tensor(float)",
                )
            ]

        def get_outputs(self):
            return []

        def get_providers(self):
            return self.providers

        def run(self, _output_names, _feed):
            raise RuntimeError("detailed kernel failure")

        def end_profiling(self):
            profile_path = Path(f"{self.profile_prefix}.json")
            profile_path.write_text("[]")
            return str(profile_path)

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"mock model")
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["TestExecutionProvider"],
    )
    monkeypatch.setattr(ort, "SessionOptions", lambda: SimpleNamespace())
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)

    diagnostic = diagnose_execution_providers(str(model_path))["results"][0]

    assert diagnostic["session_created"]
    assert not diagnostic["inference_succeeded"]
    assert diagnostic["provider_execution_status"] == "inference_failed"
    assert diagnostic["error"] == "RuntimeError: detailed kernel failure"


def test_provider_diagnostic_reports_cpu_fallback(monkeypatch, tmp_path):
    class FakeSession:
        def __init__(self, _model_path, sess_options, providers):
            self.profile_prefix = sess_options.profile_file_prefix
            self.providers = [*providers, "CPUExecutionProvider"]

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="input",
                    shape=[1, 3],
                    type="tensor(float)",
                )
            ]

        def get_outputs(self):
            return [
                SimpleNamespace(
                    name="output",
                    shape=[1, 3],
                    type="tensor(float)",
                )
            ]

        def get_providers(self):
            return self.providers

        def run(self, _output_names, feed):
            assert feed["input"].shape == (1, 3)
            return [np.zeros((1, 3), dtype=np.float32)]

        def end_profiling(self):
            profile_path = Path(f"{self.profile_prefix}.json")
            profile_path.write_text(json.dumps([
                {
                    "cat": "Node",
                    "args": {"provider": "CPUExecutionProvider"},
                }
            ]))
            return str(profile_path)

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"mock model")
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["TestExecutionProvider"],
    )
    monkeypatch.setattr(ort, "SessionOptions", lambda: SimpleNamespace())
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)

    diagnostic = diagnose_execution_providers(str(model_path))["results"][0]

    assert diagnostic["session_created"]
    assert diagnostic["inference_succeeded"]
    assert diagnostic["actual_node_providers"] == ["CPUExecutionProvider"]
    assert diagnostic["provider_execution_status"] == "fallback"
    assert diagnostic["provider_fallback"] is True


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
