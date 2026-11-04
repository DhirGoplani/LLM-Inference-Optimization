
import numpy as np
import pytest
import onnxruntime as ort

from models.loader import load_model
from models.sample_inputs import (
    generate_diagnostic_inputs,
    load_sample_inputs,
)


@pytest.fixture
def model_spec():
    return load_model("models/mobilenet_v2.onnx")


def test_load_valid_npz(tmp_path, model_spec):
    sample_path = tmp_path / "sample.npz"
    arrays = {}

    for name in model_spec.input_names:
        shape = [
            dim if isinstance(dim, int) and dim > 0 else 1
            for dim in model_spec.input_shapes[name]
        ]

        dtype_map = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(float16)": np.float16,
            "tensor(int8)": np.int8,
            "tensor(int16)": np.int16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(uint8)": np.uint8,
            "tensor(uint16)": np.uint16,
            "tensor(uint32)": np.uint32,
            "tensor(uint64)": np.uint64,
            "tensor(bool)": np.bool_,
        }

        arrays[name] = np.zeros(
            shape,
            dtype=dtype_map[model_spec.input_dtypes[name]],
        )

    np.savez(sample_path, **arrays)

    result = load_sample_inputs(str(sample_path), model_spec)

    assert set(result) == set(model_spec.input_names)

    for name in model_spec.input_names:
        assert result[name].dtype == arrays[name].dtype
        assert result[name].shape == arrays[name].shape


def test_missing_input_is_rejected(tmp_path, model_spec):
    sample_path = tmp_path / "missing.npz"
    np.savez(sample_path)

    with pytest.raises(ValueError, match="Missing model inputs"):
        load_sample_inputs(str(sample_path), model_spec)


def test_non_npz_file_is_rejected(tmp_path, model_spec):
    sample_path = tmp_path / "sample.txt"
    sample_path.write_text("not an npz file")

    with pytest.raises(ValueError, match="Expected a .npz"):
        load_sample_inputs(str(sample_path), model_spec)


def test_missing_sample_file_is_rejected(tmp_path, model_spec):
    with pytest.raises(FileNotFoundError):
        load_sample_inputs(
            str(tmp_path / "missing.npz"),
            model_spec,
        )

def test_unexpected_input_is_rejected(tmp_path, model_spec):
    sample_path = tmp_path / "unexpected.npz"

    arrays = {}
    name = model_spec.input_names[0]
    shape = [
        dim if isinstance(dim, int) and dim > 0 else 1
        for dim in model_spec.input_shapes[name]
    ]

    dtype_map = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(float16)": np.float16,
        "tensor(int8)": np.int8,
        "tensor(int16)": np.int16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(uint16)": np.uint16,
        "tensor(uint32)": np.uint32,
        "tensor(uint64)": np.uint64,
        "tensor(bool)": np.bool_,
    }

    for input_name in model_spec.input_names:
        input_shape = [
            dim if isinstance(dim, int) and dim > 0 else 1
            for dim in model_spec.input_shapes[input_name]
        ]
        arrays[input_name] = np.zeros(
            input_shape,
            dtype=dtype_map[model_spec.input_dtypes[input_name]],
        )

    arrays["unexpected_input"] = np.zeros((1,), dtype=np.float32)
    np.savez(sample_path, **arrays)

    with pytest.raises(ValueError, match="Unexpected inputs"):
        load_sample_inputs(str(sample_path), model_spec)


def test_wrong_dtype_is_rejected(tmp_path, model_spec):
    sample_path = tmp_path / "wrong_dtype.npz"

    name = model_spec.input_names[0]
    shape = [
        dim if isinstance(dim, int) and dim > 0 else 1
        for dim in model_spec.input_shapes[name]
    ]

    arrays = {}
    for input_name in model_spec.input_names:
        input_shape = [
            dim if isinstance(dim, int) and dim > 0 else 1
            for dim in model_spec.input_shapes[input_name]
        ]
        arrays[input_name] = np.zeros(
            input_shape,
            dtype=np.float32,
        )

    expected_dtype = model_spec.input_dtypes[name]
    if expected_dtype == "tensor(float)":
        arrays[name] = arrays[name].astype(np.int32)

    np.savez(sample_path, **arrays)

    if expected_dtype == "tensor(float)":
        with pytest.raises(ValueError, match="has dtype"):
            load_sample_inputs(str(sample_path), model_spec)
            

def test_wrong_fixed_dimension_is_rejected(tmp_path, model_spec):
    name = model_spec.input_names[0]
    expected_shape = model_spec.input_shapes[name]

    fixed_axis = next(
        (
            index
            for index, dimension in enumerate(expected_shape)
            if isinstance(dimension, int) and dimension > 0
        ),
        None,
    )

    if fixed_axis is None:
        pytest.skip("Model has no fixed input dimensions to test")

    shape = [
        dim if isinstance(dim, int) and dim > 0 else 1
        for dim in expected_shape
    ]
    shape[fixed_axis] += 1

    dtype_map = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(float16)": np.float16,
        "tensor(int8)": np.int8,
        "tensor(int16)": np.int16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(uint16)": np.uint16,
        "tensor(uint32)": np.uint32,
        "tensor(uint64)": np.uint64,
        "tensor(bool)": np.bool_,
    }

    arrays = {}
    for input_name in model_spec.input_names:
        input_shape = [
            dim if isinstance(dim, int) and dim > 0 else 1
            for dim in model_spec.input_shapes[input_name]
        ]
        arrays[input_name] = np.zeros(
            input_shape,
            dtype=dtype_map[model_spec.input_dtypes[input_name]],
        )

    arrays[name] = np.zeros(
        shape,
        dtype=dtype_map[model_spec.input_dtypes[name]],
    )

    sample_path = tmp_path / "wrong_shape.npz"
    np.savez(sample_path, **arrays)

    with pytest.raises(ValueError, match="expected"):
        load_sample_inputs(str(sample_path), model_spec)


def test_dynamic_dimension_accepts_different_sizes(tmp_path):
    from models.schema import ModelSpec

    spec = ModelSpec(
        model_path="dynamic.onnx",
        input_names=["input"],
        output_names=["output"],
        input_shapes={"input": ["batch", 3]},
        input_dtypes={"input": "tensor(float)"},
        output_shapes={"output": ["batch", 3]},
        output_dtypes={"output": "tensor(float)"},
    )

    sample_path = tmp_path / "dynamic.npz"
    np.savez(
        sample_path,
        input=np.zeros((4, 3), dtype=np.float32),
    )

    result = load_sample_inputs(str(sample_path), spec)

    assert result["input"].shape == (4, 3)


def test_generate_diagnostic_inputs_for_mobilenet():
    session = ort.InferenceSession(
        "models/mobilenet_v2.onnx",
        providers=["CPUExecutionProvider"],
    )

    feed = generate_diagnostic_inputs(session.get_inputs())

    assert set(feed) == {item.name for item in session.get_inputs()}
    for item in session.get_inputs():
        assert feed[item.name].shape == tuple(item.shape)
        assert str(feed[item.name].dtype) == {
            "tensor(float)": "float32",
            "tensor(float16)": "float16",
        }.get(item.type, str(feed[item.name].dtype))


def test_generate_qwen_decoder_diagnostic_inputs():
    from types import SimpleNamespace

    inputs = [
        SimpleNamespace(
            name="input_ids",
            shape=["batch_size", "sequence_length"],
            type="tensor(int64)",
        ),
        SimpleNamespace(
            name="attention_mask",
            shape=[
                "batch_size",
                "past_sequence_length + sequence_length",
            ],
            type="tensor(int64)",
        ),
        SimpleNamespace(
            name="position_ids",
            shape=["batch_size", "sequence_length"],
            type="tensor(int64)",
        ),
    ]
    for layer in range(24):
        for kind in ("key", "value"):
            inputs.append(
                SimpleNamespace(
                    name=f"past_key_values.{layer}.{kind}",
                    shape=["batch_size", 2, "past_sequence_length", 64],
                    type="tensor(float16)",
                )
            )

    feed = generate_diagnostic_inputs(inputs)

    assert feed["input_ids"].shape == (1, 2)
    assert feed["input_ids"].dtype == np.int64
    assert np.all(feed["input_ids"] == 0)
    assert feed["attention_mask"].shape == (1, 2)
    assert np.array_equal(feed["attention_mask"], [[1, 1]])
    assert feed["position_ids"].shape == (1, 2)
    assert np.array_equal(feed["position_ids"], [[0, 1]])
    assert feed["past_key_values.0.key"].shape == (1, 2, 0, 64)
    assert feed["past_key_values.0.key"].dtype == np.float16
    assert feed["past_key_values.23.value"].shape == (1, 2, 0, 64)
    assert feed["past_key_values.23.value"].dtype == np.float16


def test_generate_diagnostic_inputs_rejects_fixed_nonempty_cache():
    from types import SimpleNamespace

    inputs = [
        SimpleNamespace(
            name="input_ids",
            shape=["batch_size", "sequence_length"],
            type="tensor(int64)",
        ),
        SimpleNamespace(
            name="attention_mask",
            shape=[
                "batch_size",
                "past_sequence_length + sequence_length",
            ],
            type="tensor(int64)",
        ),
        SimpleNamespace(
            name="past_key_values.0.key",
            shape=["batch_size", 2, 8, 64],
            type="tensor(float16)",
        ),
    ]

    with pytest.raises(ValueError, match="empty first-step KV cache"):
        generate_diagnostic_inputs(inputs)

