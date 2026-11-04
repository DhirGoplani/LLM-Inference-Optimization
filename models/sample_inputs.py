
import ast
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from models.llm_inputs import is_kv_cache_input
from models.schema import ModelSpec

ONNX_TO_NUMPY_DTYPES: dict[str, type] = {
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


def _dimension_value(
    dimension: Any,
    *,
    batch_size: int,
    sequence_length: int,
    past_length: int,
) -> int:
    if isinstance(dimension, int):
        if dimension < 0:
            raise ValueError(f"Invalid negative ONNX dimension: {dimension}")
        return dimension

    bindings = {
        "batch": batch_size,
        "batch_size": batch_size,
        "sequence": sequence_length,
        "sequence_length": sequence_length,
        "seq": sequence_length,
        "seq_len": sequence_length,
        "past_sequence_length": past_length,
        "past_seq_len": past_length,
    }

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Name):
            name = node.id.lower()
            if name in bindings:
                return bindings[name]
            if "past" in name and ("seq" in name or "sequence" in name):
                return past_length
            if "batch" in name:
                return batch_size
            if "seq" in name or "sequence" in name:
                return sequence_length
            raise ValueError(f"Unknown symbolic ONNX dimension: {node.id}")
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv) and right:
                return left // right
        raise ValueError(f"Unsupported symbolic ONNX dimension: {dimension!r}")

    if not isinstance(dimension, str):
        return 1

    try:
        value = evaluate(ast.parse(dimension, mode="eval").body)
        if value >= 0:
            return value
    except (SyntaxError, ValueError):
        pass

    return 1


def _fixed_dimension(dimension: Any) -> int | None:
    return dimension if isinstance(dimension, int) and dimension > 0 else None


def generate_diagnostic_inputs(
    model_inputs: Iterable[Any],
) -> dict[str, np.ndarray]:
    """Create a small, signature-compatible first-step input feed for an ONNX session."""
    inputs = list(model_inputs)
    if not inputs:
        raise ValueError("Cannot generate diagnostic inputs for a model with no inputs.")

    cache_inputs = [
        item
        for item in inputs
        if is_kv_cache_input(item.name)
    ]

    fixed_past_lengths = {
        length
        for item in cache_inputs
        if len(item.shape) == 4
        if (length := _fixed_dimension(item.shape[2])) is not None
    }
    if fixed_past_lengths:
        raise ValueError(
            "Cannot create an empty first-step KV cache because the model "
            "requires fixed positive past-sequence dimensions: "
            f"{sorted(fixed_past_lengths)}."
        )
    past_length = 0

    batch_candidates = {
        size
        for item in inputs
        if item.name in ("input_ids", "attention_mask", "position_ids")
        or is_kv_cache_input(item.name)
        if item.shape
        if (size := _fixed_dimension(item.shape[0])) is not None
    }
    if len(batch_candidates) > 1:
        raise ValueError(
            f"Model inputs declare incompatible fixed batch dimensions: "
            f"{sorted(batch_candidates)}."
        )
    batch_size = next(iter(batch_candidates), 1)

    sequence_candidates = {
        size
        for item in inputs
        if item.name in ("input_ids", "position_ids")
        if len(item.shape) > 1
        if (size := _fixed_dimension(item.shape[1])) is not None
    }
    if len(sequence_candidates) > 1:
        raise ValueError(
            "Token inputs declare incompatible fixed sequence dimensions: "
            f"{sorted(sequence_candidates)}."
        )
    sequence_length = next(iter(sequence_candidates), 2)

    fixed_mask_lengths = {
        size
        for item in inputs
        if item.name == "attention_mask"
        if len(item.shape) > 1
        if (size := _fixed_dimension(item.shape[1])) is not None
    }
    for mask_length in fixed_mask_lengths:
        mask_sequence_length = mask_length - past_length
        if mask_sequence_length <= 0:
            raise ValueError(
                f"Fixed attention-mask length {mask_length} is not compatible "
                f"with past length {past_length}."
            )
        if sequence_candidates and mask_sequence_length != sequence_length:
            raise ValueError(
                "Fixed attention-mask and token sequence dimensions are "
                "incompatible."
            )
        if not sequence_candidates:
            sequence_length = mask_sequence_length

    result: dict[str, np.ndarray] = {}

    for item in inputs:
        dtype = ONNX_TO_NUMPY_DTYPES.get(item.type)
        if dtype is None:
            raise ValueError(
                f"Unsupported diagnostic input type {item.type!r} "
                f"for {item.name!r}"
            )

        shape = [
            _dimension_value(
                dimension,
                batch_size=batch_size,
                sequence_length=sequence_length,
                past_length=past_length,
            )
            for dimension in item.shape
        ]

        if item.name in ("input_ids", "position_ids"):
            if len(shape) != 2:
                raise ValueError(
                    f"Expected 2D token input {item.name!r}, got shape {item.shape}."
                )
            shape = [batch_size, sequence_length]
        elif item.name == "attention_mask":
            if len(shape) != 2:
                raise ValueError(
                    f"Expected 2D attention mask, got shape {item.shape}."
                )
            shape = [batch_size, past_length + sequence_length]
        elif is_kv_cache_input(item.name):
            if len(shape) != 4:
                raise ValueError(
                    f"Expected a 4D KV-cache input for {item.name!r}, "
                    f"got shape {item.shape}."
                )
            shape[0] = batch_size
            shape[2] = past_length

        if item.name == "position_ids":
            values = np.arange(
                past_length,
                past_length + sequence_length,
                dtype=dtype,
            )
            result[item.name] = np.broadcast_to(
                values.reshape(1, sequence_length),
                (batch_size, sequence_length),
            ).copy()
        elif item.name == "attention_mask":
            result[item.name] = np.ones(shape, dtype=dtype)
        else:
            result[item.name] = np.zeros(shape, dtype=dtype)

    return result


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

        expected_numpy_dtype = ONNX_TO_NUMPY_DTYPES.get(expected_dtype)

        if expected_numpy_dtype is None:
            raise ValueError(
                f"Unsupported ONNX input type '{expected_dtype}' "
                f"for input '{name}'."
            )
        expected_numpy_dtype = np.dtype(expected_numpy_dtype)

        if array.dtype != expected_numpy_dtype:
            raise ValueError(
                f"Input '{name}' has dtype {array.dtype}; "
                f"expected {expected_numpy_dtype} "
                f"for ONNX type '{expected_dtype}'."
            )

    return samples
