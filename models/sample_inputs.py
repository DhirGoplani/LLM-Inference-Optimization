
from pathlib import Path
from typing import Any

import numpy as np

from models.schema import ModelSpec


def load_sample_inputs(
    sample_path: str,
    spec: ModelSpec,
) -> dict[str, np.ndarray]:
    path = Path(sample_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Sample input file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Sample input path is not a file: {path}"
        )

    if path.suffix.lower() != ".npz":
        raise ValueError(
            f"Expected a .npz sample input file: {path}"
        )

    try:
        with np.load(path, allow_pickle=False) as data:
            samples = {
                name: data[name].copy()
                for name in data.files
            }
    except Exception as exc:
        raise ValueError(
            f"Could not read sample input file: {exc}"
        ) from exc

    expected_names = set(spec.input_names)
    actual_names = set(samples)

    missing = expected_names - actual_names
    unexpected = actual_names - expected_names

    if missing:
        raise ValueError(
            f"Missing model inputs in .npz file: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected inputs in .npz file: {sorted(unexpected)}"
        )

    for name in spec.input_names:
        array = samples[name]
        expected_shape = spec.input_shapes[name]
        expected_dtype = spec.input_dtypes[name]

        if array.dtype.hasobject:
            raise ValueError(
                f"Object arrays are not supported for input '{name}'."
            )

        if len(array.shape) != len(expected_shape):
            raise ValueError(
                f"Input '{name}' has rank {array.ndim}; "
                f"expected rank {len(expected_shape)}."
            )

        for axis, (actual_dim, expected_dim) in enumerate(
            zip(array.shape, expected_shape)
        ):
            if (
                isinstance(expected_dim, int)
                and expected_dim > 0
                and actual_dim != expected_dim
            ):
                raise ValueError(
                    f"Input '{name}' dimension {axis} is "
                    f"{actual_dim}; expected {expected_dim}."
                )

        dtype_map = {
            "tensor(float)": np.dtype(np.float32),
            "tensor(double)": np.dtype(np.float64),
            "tensor(float16)": np.dtype(np.float16),
            "tensor(int8)": np.dtype(np.int8),
            "tensor(int16)": np.dtype(np.int16),
            "tensor(int32)": np.dtype(np.int32),
            "tensor(int64)": np.dtype(np.int64),
            "tensor(uint8)": np.dtype(np.uint8),
            "tensor(uint16)": np.dtype(np.uint16),
            "tensor(uint32)": np.dtype(np.uint32),
            "tensor(uint64)": np.dtype(np.uint64),
            "tensor(bool)": np.dtype(np.bool_),
        }

        expected_numpy_dtype = dtype_map.get(expected_dtype)

        if expected_numpy_dtype is None:
            raise ValueError(
                f"Unsupported ONNX input type '{expected_dtype}' "
                f"for input '{name}'."
            )

        if array.dtype != expected_numpy_dtype:
            raise ValueError(
                f"Input '{name}' has dtype {array.dtype}; "
                f"expected {expected_numpy_dtype} "
                f"for ONNX type '{expected_dtype}'."
            )

    return samples
