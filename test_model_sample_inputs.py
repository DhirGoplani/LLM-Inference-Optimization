
import numpy as np
import pytest

from models.loader import load_model
from models.sample_inputs import load_sample_inputs


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



