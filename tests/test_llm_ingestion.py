from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.llm_inputs import create_prefill_feed, is_kv_cache_input
from models.loader import load_llm_model
from models.metadata import (
    calculate_approx_parameter_count,
    detect_onnx_precision,
    get_qwen2_5_0_5b_metadata,
    load_llm_metadata,
    resolve_canonical_model_id,
)
from models.schema import ModelSpec
from models.tokenizer import decode_tokens, encode_prompt, load_tokenizer


class MockInput:
    """Mock session input with name, shape, and type descriptors."""
    def __init__(self, name: str, shape: list, inp_type: str):
        self.name = name
        self.shape = shape
        self.type = inp_type


def create_mock_onnx_model(target_path: Path, precisions: list[str]) -> None:
    """Helper to create minimal valid ONNX model file with specific initializer data types."""
    import onnx
    from onnx import helper, TensorProto

    type_map = {
        "fp16": TensorProto.FLOAT16,
        "fp32": TensorProto.FLOAT,
        "int64": TensorProto.INT64,
    }
    initializers = [
        helper.make_tensor(f"weight_{idx}", type_map[prec], [2], [1.0, 2.0] if prec != "int64" else [1, 2])
        for idx, prec in enumerate(precisions)
    ]
    graph = helper.make_graph([], "mock_graph", [], [], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(target_path))


# -------------------------------------------------------------------------
# 1. Schema & Backward Compatibility Tests
# -------------------------------------------------------------------------

def test_model_spec_metadata_backward_compatibility():
    """Verify ModelSpec initializes with or without metadata, maintaining compatibility."""
    spec_default = ModelSpec(
        model_path="test.onnx",
        input_names=["input"],
        output_names=["output"],
        input_shapes={"input": [1, 3]},
        input_dtypes={"input": "tensor(float)"},
        output_shapes={"output": [1, 3]},
        output_dtypes={"output": "tensor(float)"},
    )
    assert spec_default.metadata == {}
    assert spec_default.available_providers == []

    spec_meta = ModelSpec(
        model_path="test.onnx",
        input_names=["input"],
        output_names=["output"],
        input_shapes={"input": [1, 3]},
        input_dtypes={"input": "tensor(float)"},
        output_shapes={"output": [1, 3]},
        output_dtypes={"output": "tensor(float)"},
        metadata={"architecture": "TestModel", "vocab_size": 100},
    )
    assert spec_meta.metadata["architecture"] == "TestModel"

    d = asdict(spec_meta)
    assert d["metadata"]["vocab_size"] == 100


# -------------------------------------------------------------------------
# 2. Parameter Count Calculation Tests
# -------------------------------------------------------------------------

def test_parameter_count_qwen_verified():
    """Verify that configuration-derived parameter calculation consistently yields 494,032,768 for Qwen2.5-0.5B."""
    qwen_cfg = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    calc_params = calculate_approx_parameter_count(qwen_cfg)
    assert calc_params == 494_032_768


def test_parameter_count_invalid_and_missing_configs():
    """Verify calculate_approx_parameter_count returns None for missing or invalid configs."""
    base_valid = {
        "hidden_size": 896,
        "vocab_size": 151936,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
    }

    # Non-dict inputs
    assert calculate_approx_parameter_count(None) is None  # type: ignore
    assert calculate_approx_parameter_count([]) is None  # type: ignore
    assert calculate_approx_parameter_count("invalid") is None  # type: ignore

    # Missing each required field
    for field in ("hidden_size", "vocab_size", "num_hidden_layers", "intermediate_size", "num_attention_heads"):
        cfg = base_valid.copy()
        del cfg[field]
        assert calculate_approx_parameter_count(cfg) is None

    # Invalid field types / values
    assert calculate_approx_parameter_count({**base_valid, "hidden_size": -896}) is None
    assert calculate_approx_parameter_count({**base_valid, "hidden_size": 0}) is None
    assert calculate_approx_parameter_count({**base_valid, "hidden_size": "896"}) is None
    assert calculate_approx_parameter_count({**base_valid, "hidden_size": True}) is None  # bool != valid int

    # Hidden size not divisible by attention heads
    assert calculate_approx_parameter_count({**base_valid, "hidden_size": 895, "num_attention_heads": 14}) is None

    # Invalid KV heads for GQA
    assert calculate_approx_parameter_count({**base_valid, "num_attention_heads": 14, "num_key_value_heads": 16}) is None
    assert calculate_approx_parameter_count({**base_valid, "num_attention_heads": 14, "num_key_value_heads": 5}) is None


def test_parameter_count_tied_vs_untied_embeddings():
    """Verify tied vs untied embeddings account for LM head (vocab_size * hidden_size)."""
    cfg_tied = {
        "model_type": "llama",
        "hidden_size": 2048,
        "vocab_size": 128256,
        "num_hidden_layers": 16,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "tie_word_embeddings": True,
    }
    cfg_untied = {**cfg_tied, "tie_word_embeddings": False}

    count_tied = calculate_approx_parameter_count(cfg_tied)
    count_untied = calculate_approx_parameter_count(cfg_untied)

    assert count_tied is not None
    assert count_untied is not None
    assert count_untied - count_tied == 128256 * 2048


def test_metadata_insufficient_config_returns_unknown_basis(tmp_path):
    """Verify load_llm_metadata reports unknown basis when parameter count cannot be calculated."""
    # Incomplete config lacking intermediate_size and num_attention_heads
    cfg = {
        "model_type": "custom",
        "hidden_size": 512,
        "vocab_size": 1000,
        "num_hidden_layers": 4,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    meta = load_llm_metadata(tmp_path)
    assert meta["parameter_count_approx"] is None
    assert "unknown" in meta["parameter_count_basis"]


# -------------------------------------------------------------------------
# 3. Precision Detection Tests
# -------------------------------------------------------------------------

def test_detect_onnx_precision_all_fp16(tmp_path):
    """Verify detect_onnx_precision returns fp16 when all initializers are FLOAT16."""
    create_mock_onnx_model(tmp_path / "model.onnx", ["fp16", "fp16"])
    precision, source = detect_onnx_precision(tmp_path)
    assert precision == "fp16"
    assert "consistent float16" in source


def test_detect_onnx_precision_all_fp32(tmp_path):
    """Verify detect_onnx_precision returns fp32 when all initializers are FLOAT."""
    create_mock_onnx_model(tmp_path / "model.onnx", ["fp32", "fp32"])
    precision, source = detect_onnx_precision(tmp_path)
    assert precision == "fp32"
    assert "consistent float32" in source


def test_detect_onnx_precision_mixed(tmp_path):
    """Verify detect_onnx_precision returns mixed when multiple float precisions exist."""
    create_mock_onnx_model(tmp_path / "model.onnx", ["fp16", "fp32"])
    precision, source = detect_onnx_precision(tmp_path)
    assert precision == "mixed"
    assert "mixed precision" in source


def test_detect_onnx_precision_missing_file(tmp_path):
    """Verify detect_onnx_precision returns unknown when no ONNX file exists."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    precision, source = detect_onnx_precision(empty_dir)
    assert precision == "unknown"
    assert source == "no_onnx_file_found"


def test_detect_onnx_precision_inspection_failure(tmp_path):
    """Verify detect_onnx_precision warns with path and exception and returns unknown on corrupted ONNX."""
    bad_model = tmp_path / "model.onnx"
    bad_model.write_text("corrupted non-onnx content", encoding="utf-8")

    with pytest.warns(UserWarning, match="Failed to inspect precision from"):
        precision, source = detect_onnx_precision(tmp_path)
        assert precision == "unknown"
        assert "inspection_error" in source


def test_detect_onnx_precision_no_float_initializers(tmp_path):
    """Verify detect_onnx_precision returns unknown when graph has only non-float initializers."""
    create_mock_onnx_model(tmp_path / "model.onnx", ["int64"])
    precision, source = detect_onnx_precision(tmp_path)
    assert precision == "unknown"
    assert "no float initializers found" in source


# -------------------------------------------------------------------------
# 4. Canonical Model ID Resolution Tests
# -------------------------------------------------------------------------

def test_resolve_canonical_model_id_supported_models(tmp_path):
    """Verify resolution for all four supported canonical models and valid HF IDs."""
    # 1. Qwen 2.5 0.5B
    cfg_qwen_05 = {
        "_name_or_path": "models/qwen2.5_0.5b",  # local path, should use signature
        "model_type": "qwen2",
        "num_hidden_layers": 24,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "vocab_size": 151936,
    }
    assert resolve_canonical_model_id(cfg_qwen_05, tmp_path) == "Qwen/Qwen2.5-0.5B-Instruct"

    # 2. Qwen 2.5 1.5B
    cfg_qwen_15 = {
        "model_type": "qwen2",
        "num_hidden_layers": 28,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "vocab_size": 151936,
    }
    assert resolve_canonical_model_id(cfg_qwen_15, tmp_path) == "Qwen/Qwen2.5-1.5B-Instruct"

    # 3. Llama 3.2 1B
    cfg_llama = {
        "model_type": "llama",
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 128256,
    }
    assert resolve_canonical_model_id(cfg_llama, tmp_path) == "meta-llama/Llama-3.2-1B-Instruct"

    # 4. Gemma 2 2B
    cfg_gemma = {
        "model_type": "gemma2",
        "num_hidden_layers": 26,
        "hidden_size": 2304,
        "intermediate_size": 9216,
        "vocab_size": 256000,
    }
    assert resolve_canonical_model_id(cfg_gemma, tmp_path) == "google/gemma-2-2b-it"

    # Explicit canonical HF ID in _name_or_path
    cfg_explicit = {"_name_or_path": "custom-org/custom-model-id", "model_type": "custom"}
    assert resolve_canonical_model_id(cfg_explicit, tmp_path) == "custom-org/custom-model-id"


def test_resolve_canonical_model_id_unknown_fallback(tmp_path):
    """Verify unknown models fall back safely to raw ID or directory name without misidentification."""
    # Partial match: has hidden_size 896 and layers 24, but different intermediate_size, vocab_size, and model_type
    cfg_partial = {
        "_name_or_path": "./local/custom_path",
        "model_type": "custom_arch",
        "num_hidden_layers": 24,
        "hidden_size": 896,
        "intermediate_size": 2048,
        "vocab_size": 32000,
    }
    res = resolve_canonical_model_id(cfg_partial, tmp_path)
    assert res == "./local/custom_path"

    # Unknown model with no _name_or_path
    cfg_unknown = {
        "model_type": "unknown_transformer",
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "vocab_size": 50257,
    }
    assert resolve_canonical_model_id(cfg_unknown, tmp_path) == tmp_path.name


# -------------------------------------------------------------------------
# 5. Fallback Metadata Tests
# -------------------------------------------------------------------------

def test_fallback_metadata_clearly_distinguishable(tmp_path):
    """Verify static reference fallback metadata is distinguishable from dynamic metadata."""
    empty_dir = tmp_path / "nonexistent_model_dir"
    fallback_meta = get_qwen2_5_0_5b_metadata(empty_dir)

    assert fallback_meta["precision_source"] == "static_reference"
    assert "static_reference" in fallback_meta["parameter_count_basis"]
    assert fallback_meta["parameter_count_approx"] == 494_032_768
    assert fallback_meta["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"

    # Now verify dynamically loaded metadata has dynamic bases
    valid_cfg = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    model_dir = tmp_path / "valid_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(valid_cfg), encoding="utf-8")
    create_mock_onnx_model(model_dir / "model.onnx", ["fp16"])

    dynamic_meta = load_llm_metadata(model_dir)
    assert dynamic_meta["precision_source"] != "static_reference"
    assert "onnx_graph" in dynamic_meta["precision_source"]
    assert "configuration-derived" in dynamic_meta["parameter_count_basis"]
    assert "static_reference" not in dynamic_meta["parameter_count_basis"]


# -------------------------------------------------------------------------
# 6. Metadata Loading Integration Tests
# -------------------------------------------------------------------------

def test_load_llm_metadata_canonical_id_and_parsing(tmp_path):
    """Verify canonical Hugging Face model ID resolution and config extraction."""
    cfg = {
        "_name_or_path": "models/local_qwen_dir",
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    meta = load_llm_metadata(tmp_path)
    assert meta["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert meta["head_dim"] == 64
    assert meta["parameter_count_approx"] == 494_032_768


def test_metadata_corrupted_generation_config_warning(tmp_path):
    """Verify that a malformed generation_config.json triggers UserWarning instead of silent pass."""
    cfg = {
        "model_type": "qwen2",
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "vocab_size": 151936,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "generation_config.json").write_text("{corrupted json", encoding="utf-8")

    with pytest.warns(UserWarning, match="Could not parse generation_config.json"):
        meta = load_llm_metadata(tmp_path)
        assert meta["hidden_size"] == 896


# -------------------------------------------------------------------------
# 7. Model Loader Safe Path Validation Tests
# -------------------------------------------------------------------------

def test_load_llm_model_safe_path_validation(tmp_path):
    """Verify model_file_name path traversal and extension checks in load_llm_model."""
    # 1. Non-onnx extension rejected
    with pytest.raises(ValueError, match="must end with .onnx"):
        load_llm_model(tmp_path, model_file_name="model.bin")

    # 2. Path traversal rejected
    with pytest.raises(ValueError, match="attempts path traversal outside"):
        load_llm_model(tmp_path, model_file_name="../secret.onnx")

    # 3. Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Specified model file"):
        load_llm_model(tmp_path, model_file_name="nonexistent.onnx")

    # 4. Multiple models without explicit name raises ValueError
    (tmp_path / "model1.onnx").write_text("fake")
    (tmp_path / "model2.onnx").write_text("fake")
    with pytest.raises(ValueError, match="Multiple ONNX models found"):
        load_llm_model(tmp_path)


# -------------------------------------------------------------------------
# 8. LLM Input Feed Construction & Layout Validation Tests
# -------------------------------------------------------------------------

def test_is_kv_cache_input_pattern_recognition():
    """Verify is_kv_cache_input strictly recognizes standard past KV-cache names and rejects non-standard ones."""
    # Standard matching names
    assert is_kv_cache_input("past_key_values.0.key")
    assert is_kv_cache_input("past_key_values.23.value")
    assert is_kv_cache_input("past_0_key")
    assert is_kv_cache_input("past_12_value")
    assert is_kv_cache_input("past_key_0")
    assert is_kv_cache_input("past_value_15")

    # Non-matching names rejected
    assert not is_kv_cache_input("past_key_values")
    assert not is_kv_cache_input("past_key_values.custom")
    assert not is_kv_cache_input("past_custom_tensor")
    assert not is_kv_cache_input("input_ids")
    assert not is_kv_cache_input("present.0.key")


def test_create_prefill_feed_valid_dynamic_derivation():
    """Verify create_prefill_feed derives KV-cache dimensions dynamically."""
    mock_inputs = [
        MockInput("input_ids", ["batch_size", "sequence_length"], "tensor(int64)"),
        MockInput("attention_mask", ["batch_size", "past_seq + sequence_length"], "tensor(int64)"),
        MockInput("position_ids", ["batch_size", "sequence_length"], "tensor(int64)"),
        MockInput("past_key_values.0.key", ["batch_size", 4, "past_sequence_length", 32], "tensor(float16)"),
        MockInput("past_key_values.0.value", ["batch_size", 4, "past_sequence_length", 32], "tensor(float16)"),
        MockInput("use_cache_branch", [1], "tensor(bool)"),
    ]

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = mock_inputs

    input_ids = np.array([[101, 102, 103]], dtype=np.int64)
    feed = create_prefill_feed(mock_session, input_ids)

    assert feed["input_ids"].shape == (1, 3)
    assert feed["attention_mask"].shape == (1, 3)
    assert np.array_equal(feed["position_ids"], np.array([[0, 1, 2]]))
    assert feed["use_cache_branch"].tolist() == [False]

    # Derived dynamically: batch=1, heads=4, past_seq=0, head_dim=32
    past_k = feed["past_key_values.0.key"]
    assert past_k.shape == (1, 4, 0, 32)
    assert past_k.dtype == np.float16


def test_create_prefill_feed_unsupported_kv_cache_layouts():
    """Verify create_prefill_feed rejects invalid KV-cache ranks and dynamic head dimensions."""
    mock_session = MagicMock()

    # 1. Rank 3 KV-cache tensor rejected
    mock_session.get_inputs.return_value = [
        MockInput("input_ids", ["batch", "seq"], "tensor(int64)"),
        MockInput("past_key_values.0.key", ["batch", "seq", 64], "tensor(float16)"),
    ]
    with pytest.raises(ValueError, match="Unsupported KV-cache tensor rank 3"):
        create_prefill_feed(mock_session, np.array([[1, 2]]))

    # 2. Dynamic head dimension rejected
    mock_session.get_inputs.return_value = [
        MockInput("input_ids", ["batch", "seq"], "tensor(int64)"),
        MockInput("past_key_values.0.key", ["batch", 2, "past_seq", "dynamic_head_dim"], "tensor(float16)"),
    ]
    with pytest.raises(ValueError, match="Unsupported or dynamic head dimension"):
        create_prefill_feed(mock_session, np.array([[1, 2]]))

    # 3. Positive fixed past-sequence dimension rejected
    mock_session.get_inputs.return_value = [
        MockInput("input_ids", ["batch", "seq"], "tensor(int64)"),
        MockInput("past_key_values.0.key", ["batch", 2, 10, 64], "tensor(float16)"),
    ]
    with pytest.raises(ValueError, match="positive fixed past-sequence dimension"):
        create_prefill_feed(mock_session, np.array([[1, 2]]))


def test_create_prefill_feed_batch_dimension_and_empty_validation():
    """Verify create_prefill_feed rejects empty batches, empty sequences, and static batch mismatches."""
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [
        MockInput("input_ids", [2, "sequence_length"], "tensor(int64)"),
    ]

    # 1. Empty batch dimension (0, 3)
    with pytest.raises(ValueError, match="positive batch and sequence dimensions"):
        create_prefill_feed(mock_session, np.zeros((0, 3), dtype=np.int64))

    # 2. Empty sequence dimension (1, 0)
    with pytest.raises(ValueError, match="positive batch and sequence dimensions"):
        create_prefill_feed(mock_session, np.zeros((1, 0), dtype=np.int64))

    # 3. Static batch mismatch: model expects static batch 2, provided batch 1
    with pytest.raises(ValueError, match="has static batch dimension 2, which does not match input batch size 1"):
        create_prefill_feed(mock_session, np.array([[101, 102]], dtype=np.int64))


def test_create_prefill_feed_control_input_validation():
    """Verify control input derives shape/dtype and rejects invalid types or shapes."""
    mock_session = MagicMock()

    # 1. Non-boolean control input rejected
    mock_session.get_inputs.return_value = [
        MockInput("input_ids", ["batch", "seq"], "tensor(int64)"),
        MockInput("use_cache_branch", [1], "tensor(int64)"),
    ]
    with pytest.raises(TypeError, match="Expected boolean tensor type"):
        create_prefill_feed(mock_session, np.array([[1, 2]]))

    # 2. Unsupported shape (e.g. 2D) for control input rejected
    mock_session.get_inputs.return_value = [
        MockInput("input_ids", ["batch", "seq"], "tensor(int64)"),
        MockInput("use_cache_branch", [2, 2], "tensor(bool)"),
    ]
    with pytest.raises(ValueError, match="Unsupported shape"):
        create_prefill_feed(mock_session, np.array([[1, 2]]))

    # 3. Scalar control input supported
    mock_session.get_inputs.return_value = [
        MockInput("input_ids", ["batch", "seq"], "tensor(int64)"),
        MockInput("use_cache_branch", [], "tensor(bool)"),
    ]
    feed = create_prefill_feed(mock_session, np.array([[1, 2]]))
    assert feed["use_cache_branch"].ndim == 0
    assert bool(feed["use_cache_branch"]) is False


def test_create_prefill_feed_invalid_attention_mask_shapes():
    """Verify validation for attention_mask rank, batch, and sequence length mismatch."""
    mock_inputs = [
        MockInput("input_ids", ["batch_size", "sequence_length"], "tensor(int64)"),
        MockInput("attention_mask", ["batch_size", "sequence_length"], "tensor(int64)"),
    ]
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = mock_inputs

    input_ids = np.array([[10, 20, 30]], dtype=np.int64)  # shape (1, 3)

    # 1. Rank mismatch
    with pytest.raises(ValueError, match="attention_mask must be 2D"):
        create_prefill_feed(mock_session, input_ids, attention_mask=np.array([1, 1, 1]))

    # 2. Batch mismatch
    with pytest.raises(ValueError, match="batch dimension .* does not match"):
        create_prefill_feed(mock_session, input_ids, attention_mask=np.ones((2, 3), dtype=np.int64))

    # 3. Sequence length mismatch
    with pytest.raises(ValueError, match="sequence length .* does not match"):
        create_prefill_feed(mock_session, input_ids, attention_mask=np.ones((1, 5), dtype=np.int64))


def test_create_prefill_feed_unsupported_dtype():
    """Verify descriptive TypeError when encountering an unsupported tensor type."""
    mock_inputs = [
        MockInput("input_ids", ["batch_size", "sequence_length"], "tensor(unsupported_type)"),
    ]
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = mock_inputs

    with pytest.raises(TypeError, match="Unsupported ONNX tensor type 'tensor\\(unsupported_type\\)'"):
        create_prefill_feed(mock_session, np.array([[1, 2]], dtype=np.int64))


def test_create_prefill_feed_unrecognized_input_name():
    """Verify descriptive ValueError when encountering an unknown required input name."""
    mock_inputs = [
        MockInput("input_ids", ["batch_size", "sequence_length"], "tensor(int64)"),
        MockInput("unknown_feature_tensor", [1, 10], "tensor(float)"),
    ]
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = mock_inputs

    with pytest.raises(ValueError, match="Unsupported or unrecognized required model input 'unknown_feature_tensor'"):
        create_prefill_feed(mock_session, np.array([[1, 2]], dtype=np.int64))


# -------------------------------------------------------------------------
# 9. Tokenizer Output Validation & Decode Tests
# -------------------------------------------------------------------------

def test_tokenizer_encode_prompt_validation():
    """Verify encode_prompt enforces non-empty string prompts and valid outputs."""
    mock_tok = MagicMock()

    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        encode_prompt(mock_tok, "")

    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        encode_prompt(mock_tok, "   ")

    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        encode_prompt(mock_tok, 12345)  # type: ignore


def test_tokenizer_encode_prompt_output_structure_validation():
    """Verify encode_prompt validates keys, shapes, and non-empty arrays."""
    mock_tok = MagicMock()

    # 1. Missing required key
    mock_tok.return_value = {"input_ids": np.array([[1, 2]])}
    with pytest.raises(ValueError, match="missing required key 'attention_mask'"):
        encode_prompt(mock_tok, "valid prompt")

    # 2. Empty array
    mock_tok.return_value = {
        "input_ids": np.array([[]]),
        "attention_mask": np.array([[]]),
    }
    with pytest.raises(ValueError, match="empty input_ids array"):
        encode_prompt(mock_tok, "valid prompt")

    # 3. 1D array instead of 2D
    mock_tok.return_value = {
        "input_ids": np.array([1, 2]),
        "attention_mask": np.array([1, 1]),
    }
    with pytest.raises(ValueError, match="input_ids must be 2D"):
        encode_prompt(mock_tok, "valid prompt")

    # 4. Shape mismatch between input_ids and attention_mask
    mock_tok.return_value = {
        "input_ids": np.array([[1, 2]]),
        "attention_mask": np.array([[1, 1, 1]]),
    }
    with pytest.raises(ValueError, match="does not match attention_mask shape"):
        encode_prompt(mock_tok, "valid prompt")

    # 5. Floating-point input_ids rejected
    mock_tok.return_value = {
        "input_ids": np.array([[1.5, 2.5]]),
        "attention_mask": np.array([[1, 1]]),
    }
    with pytest.raises(TypeError, match="invalid input_ids dtype"):
        encode_prompt(mock_tok, "valid prompt")

    # 6. Boolean input_ids rejected
    mock_tok.return_value = {
        "input_ids": np.array([[True, False]]),
        "attention_mask": np.array([[1, 1]]),
    }
    with pytest.raises(TypeError, match="invalid input_ids dtype"):
        encode_prompt(mock_tok, "valid prompt")

    # 7. Non-binary attention mask rejected
    mock_tok.return_value = {
        "input_ids": np.array([[10, 20]]),
        "attention_mask": np.array([[1, 2]]),
    }
    with pytest.raises(ValueError, match="attention_mask values must be binary"):
        encode_prompt(mock_tok, "valid prompt")


def test_tokenizer_decode_tokens_validation():
    """Verify decode_tokens rejects None, invalid types, floats, and negative token IDs."""
    mock_tok = MagicMock()
    mock_tok.decode.return_value = "decoded text"

    # 1. None rejected
    with pytest.raises(ValueError, match="token_ids cannot be None"):
        decode_tokens(mock_tok, None)  # type: ignore

    # 2. Non-sequence rejected
    with pytest.raises(TypeError, match="must be a Sequence or numpy array"):
        decode_tokens(mock_tok, 12345)  # type: ignore

    # 3. Floating-point numpy array rejected
    with pytest.raises(TypeError, match="token_ids must be integers"):
        decode_tokens(mock_tok, np.array([1.5, 2.5]))

    # 4. Boolean numpy array rejected
    with pytest.raises(TypeError, match="token_ids must be integers"):
        decode_tokens(mock_tok, np.array([True, False]))

    # 5. Floating-point element in list rejected
    with pytest.raises(TypeError, match="Token IDs must be integers"):
        decode_tokens(mock_tok, [10, 20.5])

    # 6. Boolean element in list rejected
    with pytest.raises(TypeError, match="Token IDs must be integers"):
        decode_tokens(mock_tok, [10, True])

    # 7. Negative token IDs rejected
    with pytest.raises(ValueError, match="must be non-negative integers"):
        decode_tokens(mock_tok, [100, -1, 200])

    # 8. Valid inputs decode properly
    assert decode_tokens(mock_tok, [10, 20, 30]) == "decoded text"
    assert decode_tokens(mock_tok, np.array([[10, 20], [30, 40]])) == "decoded text"


def test_is_kv_cache_output_pattern_recognition():
    """Verify is_kv_cache_output strictly matches layer/key/value naming conventions."""
    from scripts.validate_qwen_onnx import is_kv_cache_output
    assert is_kv_cache_output("present.0.key")
    assert is_kv_cache_output("present.23.value")
    assert is_kv_cache_output("present_0_key")
    assert is_kv_cache_output("present_15_value")
    assert is_kv_cache_output("present_key_0")
    assert is_kv_cache_output("present_value_23")

    assert not is_kv_cache_output("present")
    assert not is_kv_cache_output("present.0")
    assert not is_kv_cache_output("present_custom_output")
    assert not is_kv_cache_output("logits")


def test_tokenizer_qwen_compatibility_fallback():
    """Verify load_tokenizer only catches fix_mistral_regex TypeErrors and re-raises genuine errors."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        # 1. TypeError mentioning fix_mistral_regex triggers fallback retry
        mock_from_pretrained.side_effect = [
            TypeError("unexpected keyword argument 'fix_mistral_regex'"),
            "fallback_successful_tokenizer",
        ]
        result = load_tokenizer("qwen_model_path", fix_mistral_regex=True)
        assert result == "fallback_successful_tokenizer"
        assert mock_from_pretrained.call_count == 2

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        # 2. Genuine TypeError unrelated to fix_mistral_regex is re-raises immediately
        mock_from_pretrained.side_effect = TypeError("Genuine syntax error in tokenizer config")
        with pytest.raises(TypeError, match="Genuine syntax error"):
            load_tokenizer("qwen_model_path", fix_mistral_regex=True)


# -------------------------------------------------------------------------
# 10. Live Model Smoke Test (Conditional)
# -------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("models/qwen2.5_0.5b/model.onnx").is_file(),
    reason="Local Qwen2.5-0.5B ONNX model file not present; skipping live prefill test",
)
def test_qwen2_5_live_prefill_smoke():
    """Live smoke test executing single-pass prefill on the local Qwen ONNX model."""
    import onnxruntime as ort

    model_dir = Path("models/qwen2.5_0.5b")
    spec = load_llm_model(model_dir, model_file_name="model.onnx")
    assert spec.metadata.get("model_id") == "Qwen/Qwen2.5-0.5B-Instruct"
    assert spec.metadata.get("parameter_count_approx") == 494_032_768
    assert spec.metadata.get("precision") in ("fp16", "mixed")

    # Assert exact model input and output counts for exported Qwen graph
    assert len(spec.input_names) == 51
    assert len(spec.output_names) == 49

    kv_inputs = [name for name in spec.input_names if is_kv_cache_input(name)]
    assert len(kv_inputs) == 48

    tokenizer = load_tokenizer(model_dir)
    encoded = encode_prompt(tokenizer, "Test prompt")
    session = ort.InferenceSession(spec.model_path, providers=["CPUExecutionProvider"])
    feed = create_prefill_feed(session, encoded["input_ids"], encoded["attention_mask"])

    outputs = session.run(None, feed)
    output_dict = dict(zip([o.name for o in session.get_outputs()], outputs))

    logits = output_dict["logits"]
    assert np.isfinite(logits).all()
    assert logits.shape == (1, encoded["input_ids"].shape[1], 151936)

    # Verify all 48 KV-cache outputs exist, are finite float16, and match shape (1, 2, seq_len, 64)
    kv_outputs = [k for k in output_dict if k.startswith("present.") or "present" in k]
    assert len(kv_outputs) == 48
    for kv in kv_outputs:
        arr = output_dict[kv]
        assert arr.dtype == np.float16
        assert np.isfinite(arr).all()
        assert arr.shape == (1, 2, encoded["input_ids"].shape[1], 64)
