from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def load_tokenizer(
    model_path_or_id: str | Path,
    *,
    fix_mistral_regex: bool | None = None,
    **kwargs: Any,
) -> Any:
    """Load a Hugging Face tokenizer with model-specific compatibility handling.

    For Qwen models, fix_mistral_regex=True is applied unless explicitly configured.
    Fallback is strictly limited to TypeErrors caused by an unsupported fix_mistral_regex argument,
    preserving genuine tokenizer loading failures.
    """
    from transformers import AutoTokenizer

    model_str = str(model_path_or_id)
    path = Path(model_str)

    # Check if this model is Qwen-based
    is_qwen = "qwen" in model_str.lower()
    if not is_qwen and path.is_dir():
        cfg_file = path / "config.json"
        if cfg_file.is_file():
            try:
                cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                model_type = str(cfg.get("model_type", "")).lower()
                is_qwen = "qwen" in model_type
            except Exception as exc:
                warnings.warn(
                    f"Could not parse model config at {cfg_file} when inspecting tokenizer requirements: {exc}",
                    UserWarning,
                    stacklevel=2,
                )

    load_kwargs = dict(kwargs)

    # Apply fix_mistral_regex specifically to Qwen or when explicitly passed
    if fix_mistral_regex is not None:
        load_kwargs["fix_mistral_regex"] = fix_mistral_regex
    elif is_qwen:
        load_kwargs["fix_mistral_regex"] = True

    try:
        return AutoTokenizer.from_pretrained(model_str, **load_kwargs)
    except TypeError as exc:
        # Fallback strictly if fix_mistral_regex was rejected as an unexpected argument
        if "fix_mistral_regex" in load_kwargs and "fix_mistral_regex" in str(exc):
            load_kwargs.pop("fix_mistral_regex")
            return AutoTokenizer.from_pretrained(model_str, **load_kwargs)
        raise


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    add_special_tokens: bool = True,
) -> dict[str, np.ndarray]:
    """Tokenize a text prompt into 2D int64 numpy arrays (input_ids, attention_mask)."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"prompt must be a non-empty string, got {type(prompt).__name__}: {prompt!r}"
        )

    encoded = tokenizer(
        prompt,
        return_tensors="np",
        add_special_tokens=add_special_tokens,
    )

    if not isinstance(encoded, dict) and not hasattr(encoded, "__getitem__"):
        raise TypeError(f"Tokenizer output must be dict-like, got {type(encoded).__name__}")

    for key in ("input_ids", "attention_mask"):
        if key not in encoded:
            raise ValueError(f"Tokenizer output missing required key '{key}'")

    raw_input_ids = encoded["input_ids"]
    raw_attention_mask = encoded["attention_mask"]

    arr_ids = np.asarray(raw_input_ids)
    arr_mask = np.asarray(raw_attention_mask)

    if arr_ids.size == 0:
        raise ValueError("Tokenizer produced empty input_ids array")
    if arr_mask.size == 0:
        raise ValueError("Tokenizer produced empty attention_mask array")

    # Reject non-integer token IDs rather than silently coercing them
    if np.issubdtype(arr_ids.dtype, np.floating) or arr_ids.dtype == np.bool_:
        raise TypeError(
            f"Tokenizer returned invalid input_ids dtype {arr_ids.dtype}. Expected integer token IDs."
        )
    if not np.issubdtype(arr_ids.dtype, np.integer):
        raise TypeError(
            f"Tokenizer returned non-integer input_ids dtype {arr_ids.dtype}. Expected integer token IDs."
        )

    # Validate attention_mask: accept only integer or boolean dtypes; reject float, complex, string, object, etc.
    if not (np.issubdtype(arr_mask.dtype, np.integer) or arr_mask.dtype == np.bool_):
        raise TypeError(
            f"Tokenizer returned unsupported attention_mask dtype '{arr_mask.dtype}'. "
            f"Expected integer or boolean dtype."
        )

    input_ids = np.ascontiguousarray(arr_ids, dtype=np.int64)
    attention_mask = np.ascontiguousarray(arr_mask, dtype=np.int64)

    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must be 2D with shape (batch_size, sequence_length), got rank {input_ids.ndim} (shape: {input_ids.shape})"
        )
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must be 2D with shape (batch_size, sequence_length), got rank {attention_mask.ndim} (shape: {attention_mask.shape})"
        )

    # Explicitly validate positive batch and sequence dimensions
    batch_size, seq_len = input_ids.shape
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError(
            f"input_ids must have positive batch and sequence dimensions, got shape {input_ids.shape}"
        )
    if attention_mask.shape[0] <= 0 or attention_mask.shape[1] <= 0:
        raise ValueError(
            f"attention_mask must have positive batch and sequence dimensions, got shape {attention_mask.shape}"
        )

    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"input_ids shape {input_ids.shape} does not match attention_mask shape {attention_mask.shape}"
        )

    if (input_ids < 0).any():
        raise ValueError("Tokenizer produced negative token IDs")

    # Validate attention-mask values are binary (0 or 1)
    if not np.isin(attention_mask, [0, 1]).all():
        raise ValueError("attention_mask values must be binary (0 or 1)")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def decode_tokens(
    tokenizer: Any,
    token_ids: Sequence[int] | np.ndarray,
    *,
    skip_special_tokens: bool = False,
) -> str:
    """Decode a sequence or numpy array of token IDs back into text."""
    if token_ids is None:
        raise ValueError("token_ids cannot be None")

    if isinstance(token_ids, np.ndarray):
        if np.issubdtype(token_ids.dtype, np.floating) or token_ids.dtype == np.bool_:
            raise TypeError(
                f"token_ids must be integers, got floating-point or boolean dtype {token_ids.dtype}"
            )
        token_list = token_ids.flatten().tolist()
    elif isinstance(token_ids, (list, tuple)):
        token_list = list(token_ids)
    elif isinstance(token_ids, Sequence):
        token_list = list(token_ids)
    else:
        raise TypeError(
            f"token_ids must be a Sequence or numpy array, got {type(token_ids).__name__}"
        )

    for idx, tok in enumerate(token_list):
        if isinstance(tok, bool) or isinstance(tok, (float, np.floating)):
            raise TypeError(
                f"Invalid token ID type at index {idx}: {type(tok).__name__}. Token IDs must be integers."
            )
        if not isinstance(tok, (int, np.integer)) or tok < 0:
            raise ValueError(
                f"Invalid token ID at index {idx}: {tok!r}. Token IDs must be non-negative integers."
            )

    return tokenizer.decode(token_list, skip_special_tokens=skip_special_tokens)
