from __future__ import annotations

import re
from typing import Any

import numpy as np

ONNX_TO_NUMPY_TYPE: dict[str, type] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}

# Strict recognition pattern for past KV-cache tensor names:
# Matches:
# - Hugging Face Optimum format: past_key_values.<layer>.key / past_key_values.<layer>.value
# - ONNX Runtime format: past_<layer>_key / past_<layer>_value or past_key_<layer> / past_value_<layer>
_KV_CACHE_INPUT_PATTERN = re.compile(
    r"^(?:past_key_values\.\d+\.(?:key|value)|past_\d+_(?:key|value)|past_(?:key|value)_\d+)$"
)


def is_kv_cache_input(name: str) -> bool:
    """Explicitly identify whether an ONNX input represents a past KV-cache tensor.

    Adheres strictly to Hugging Face Optimum and ONNX Runtime decoder-model naming conventions:
    - 'past_key_values.{layer}.key' / 'past_key_values.{layer}.value'
    - 'past_{layer}_key' / 'past_{layer}_value'
    - 'past_key_{layer}' / 'past_value_{layer}'
    """
    return bool(_KV_CACHE_INPUT_PATTERN.match(name))


def create_prefill_feed(
    session: Any,
    input_ids: np.ndarray,
    attention_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Construct an initial prefill feed dictionary for a causal LM ONNX session.

    Scope:
    - This helper specifically constructs the input feed for the INITIAL PREFILL pass
      with an empty past KV-cache (past_sequence_length=0).
    - It does NOT implement cached autoregressive token decoding or rolling KV-cache
      management, which belong to Person 2's generation interface.

    Layout & Validation:
    - Input tokens (input_ids, attention_mask, position_ids) must match 2D shapes:
      (batch_size, sequence_length).
    - Past KV-cache inputs are strictly validated to conform to the 4D causal LM layout:
      (batch_size, num_heads, past_sequence_length, head_dim).
      Empty prefill arrays are allocated with sequence length 0: (batch_size, num_heads, 0, head_dim).
      Unsupported ranks or dynamic head dimensions fail descriptively.
    - Control inputs (e.g., 'use_cache_branch') derive shape and boolean dtype from session
      metadata, rejecting unsupported types or shapes.
    - Unsupported tensor dtypes raise a descriptive TypeError.
    - Unknown or unexpected input names raise a descriptive ValueError instead of silently
      constructing dummy arrays.
    """
    input_ids = np.ascontiguousarray(input_ids, dtype=np.int64)
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must be 2D with shape (batch_size, sequence_length), got rank {input_ids.ndim} (shape: {input_ids.shape})"
        )

    batch_size, seq_len = input_ids.shape
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError(
            f"input_ids must have positive batch and sequence dimensions, got shape {input_ids.shape}"
        )

    if attention_mask is None:
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    else:
        attention_mask = np.ascontiguousarray(attention_mask, dtype=np.int64)
        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must be 2D, got rank {attention_mask.ndim} (shape: {attention_mask.shape})"
            )
        if attention_mask.shape[0] <= 0 or attention_mask.shape[1] <= 0:
            raise ValueError(
                f"attention_mask must have positive batch and sequence dimensions, got shape {attention_mask.shape}"
            )
        if attention_mask.shape[0] != batch_size:
            raise ValueError(
                f"attention_mask batch dimension ({attention_mask.shape[0]}) does not match input_ids batch dimension ({batch_size})"
            )
        if attention_mask.shape[1] != seq_len:
            raise ValueError(
                f"attention_mask sequence length ({attention_mask.shape[1]}) does not match input_ids sequence length ({seq_len}) for prefill inference"
            )

    feed: dict[str, np.ndarray] = {}

    for inp in session.get_inputs():
        name = inp.name
        dtype = ONNX_TO_NUMPY_TYPE.get(inp.type)
        if dtype is None:
            raise TypeError(
                f"Unsupported ONNX tensor type '{inp.type}' for input '{name}'. "
                f"Supported types: {sorted(ONNX_TO_NUMPY_TYPE.keys())}"
            )

        # Validate static batch dimension if specified by model graph
        if (
            name in ("input_ids", "attention_mask", "position_ids") or is_kv_cache_input(name)
        ) and len(inp.shape) > 0:
            static_batch = inp.shape[0]
            if isinstance(static_batch, int) and static_batch > 0 and static_batch != batch_size:
                raise ValueError(
                    f"Model input '{name}' has static batch dimension {static_batch}, "
                    f"which does not match input batch size {batch_size}."
                )

        if name == "input_ids":
            feed[name] = input_ids
        elif name == "attention_mask":
            feed[name] = attention_mask
        elif name == "position_ids":
            pos = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
            feed[name] = np.repeat(pos, batch_size, axis=0)
        elif name == "use_cache_branch":
            if inp.type != "tensor(bool)":
                raise TypeError(
                    f"Expected boolean tensor type 'tensor(bool)' for control input '{name}', got '{inp.type}'"
                )
            if len(inp.shape) == 0:
                feed[name] = np.bool_(False)
            elif list(inp.shape) in ([1], ['1']):
                feed[name] = np.array([False], dtype=np.bool_)
            else:
                raise ValueError(
                    f"Unsupported shape {inp.shape} for control input '{name}'. Expected scalar [] or shape [1]."
                )
        elif is_kv_cache_input(name):
            if len(inp.shape) != 4:
                raise ValueError(
                    f"Unsupported KV-cache tensor rank {len(inp.shape)} for input '{name}'. "
                    f"Expected 4D tensor with layout (batch_size, num_heads, past_sequence_length, head_dim), "
                    f"got shape {inp.shape}."
                )
            num_heads_dim = inp.shape[1]
            if isinstance(num_heads_dim, int) and num_heads_dim <= 0:
                raise ValueError(
                    f"Invalid num_heads dimension ({num_heads_dim}) for KV-cache tensor '{name}'."
                )
            past_seq_dim = inp.shape[2]
            if isinstance(past_seq_dim, int) and past_seq_dim > 0:
                raise ValueError(
                    f"KV-cache input '{name}' has positive fixed past-sequence dimension ({past_seq_dim}). "
                    f"Prefill inference requires past-sequence dimension to support an empty cache (dynamic or statically 0)."
                )
            if isinstance(past_seq_dim, int) and past_seq_dim < 0:
                raise ValueError(
                    f"Invalid negative past-sequence dimension ({past_seq_dim}) for KV-cache input '{name}'."
                )
            head_dim = inp.shape[3]
            if not isinstance(head_dim, int) or head_dim <= 0:
                raise ValueError(
                    f"Unsupported or dynamic head dimension '{head_dim}' for KV-cache tensor '{name}'. "
                    f"Expected static positive integer head_dim."
                )
            num_heads = num_heads_dim if isinstance(num_heads_dim, int) else 0
            if num_heads == 0:
                raise ValueError(
                    f"Unable to derive static head count for KV-cache tensor '{name}' with shape {inp.shape}."
                )
            feed[name] = np.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype)
        else:
            raise ValueError(
                f"Unsupported or unrecognized required model input '{name}'. "
                f"Expected standard token inputs ('input_ids', 'attention_mask', 'position_ids', 'use_cache_branch') "
                f"or past KV-cache tensors matching 'past_key_values.*'."
            )

    return feed
