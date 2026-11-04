"""Tokenizer-based greedy generation for decoder-only ONNX language models."""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnxruntime as ort

from models.llm_inputs import create_prefill_feed
from models.loader import load_llm_model
from models.tokenizer import decode_tokens as decode_model_tokens
from models.tokenizer import encode_prompt as encode_model_prompt
from models.tokenizer import load_tokenizer as load_model_tokenizer


class GenerationError(RuntimeError):
    """Base error for unsupported or invalid ONNX generation requests."""


class MissingTokenizerError(GenerationError):
    """The model directory does not contain a loadable Hugging Face tokenizer."""


class UnsupportedModelError(GenerationError):
    """The ONNX graph does not expose the inputs required by generation."""


class IncompatibleModelError(GenerationError):
    """The ONNX graph exposes an incomplete or incompatible cache interface."""


class Tokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]: ...
    def decode(self, token_ids: list[int], **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    use_cache: bool = True
    seed: int = 0
    execution_provider: str = "CPUExecutionProvider"

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    config: GenerationConfig = field(default_factory=GenerationConfig)
    request_id: str | None = None
    priority: int = 0
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")


@dataclass(frozen=True)
class KVCacheMetadata:
    past_names: tuple[str, ...]
    present_names: tuple[str, ...]
    layer_count: int
    input_to_output: tuple[tuple[str, str], ...]

    @property
    def available(self) -> bool:
        return bool(self.past_names and self.present_names)


@dataclass(frozen=True)
class ModelMetadata:
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    logits_name: str
    cache: KVCacheMetadata
    model_family: str | None = None
    tokenizer_path: str | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    num_layers: int | None = None
    num_attention_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    max_sequence_length: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    cache_used: bool
    time_to_first_token_ms: float
    decode_latency_ms: float
    inter_token_latency_ms: float
    tokens_per_second: float
    metadata: KVCacheMetadata
    request_id: str | None = None
    prompt_token_count: int = 0
    generated_token_count: int = 0
    prefill_latency_ms: float = 0.0


def load_tokenizer(model_dir: str | Path) -> Tokenizer:
    """Load a local Hugging Face tokenizer without downloading model files."""
    path = Path(model_dir).expanduser().resolve()
    if not path.is_dir():
        raise MissingTokenizerError(f"Tokenizer directory not found: {path}")
    expected = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                "sentencepiece.model", "spiece.model")
    if not any((path / name).is_file() for name in expected):
        raise MissingTokenizerError(
            f"No Hugging Face tokenizer files found in {path}; expected one of "
            f"{', '.join(expected)}"
        )
    try:
        return load_model_tokenizer(path, local_files_only=True)
    except Exception as exc:
        raise MissingTokenizerError(f"Could not load tokenizer from {path}: {exc}") from exc


def _cache_names(names: list[str], marker: str) -> list[str]:
    return sorted(
        (name for name in names if marker in name.lower()),
        key=lambda name: (int(re.search(r"(?:\.|_)(\d+)", name).group(1))
                          if re.search(r"(?:\.|_)(\d+)", name) else -1, name),
    )


def inspect_cache_interface(session: Any) -> KVCacheMetadata:
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    past = _cache_names(input_names, "past")
    present = _cache_names(output_names, "present")
    if not past and not present:
        return KVCacheMetadata((), (), 0, ())
    if len(past) != len(present) or not past:
        raise IncompatibleModelError(
            "ONNX model exposes an incomplete KV-cache interface: "
            f"{len(past)} past inputs and {len(present)} present outputs"
        )
    pairs = tuple(zip(past, present))
    return KVCacheMetadata(tuple(past), tuple(present), len(past) // 2, pairs)


def build_model_metadata(session: Any, source: Any | None = None) -> ModelMetadata:
    """Build the generation view from an ORT session or a future loader metadata object."""
    inputs = tuple(item.name for item in session.get_inputs())
    outputs = tuple(item.name for item in session.get_outputs())
    cache = inspect_cache_interface(session)
    logits_name = next((name for name in outputs if "logits" in name.lower()), outputs[0])

    def value(name: str, default: Any = None) -> Any:
        if source is None:
            return default
        if isinstance(source, dict):
            return source.get(name, default)
        return getattr(source, name, default)

    return ModelMetadata(
        input_names=inputs,
        output_names=outputs,
        logits_name=logits_name,
        cache=cache,
        model_family=value("model_family", value("architecture")),
        tokenizer_path=value("tokenizer_path"),
        eos_token_id=value("eos_token_id"),
        pad_token_id=value("pad_token_id"),
        num_layers=value("num_layers", value("num_hidden_layers", value("layer_count"))),
        num_attention_heads=value("num_attention_heads"),
        num_kv_heads=value("num_kv_heads", value("num_key_value_heads")),
        head_dim=value("head_dim"),
        max_sequence_length=value(
            "max_sequence_length", value("max_position_embeddings", value("context_length"))
        ),
    )


def _numpy_dtype(onnx_type: str) -> np.dtype:
    values = {
        "tensor(int64)": np.int64, "tensor(int32)": np.int32,
        "tensor(float)": np.float32, "tensor(float16)": np.float16,
        "tensor(double)": np.float64, "tensor(bool)": np.bool_,
    }
    if onnx_type not in values:
        raise UnsupportedModelError(f"Unsupported ONNX input type: {onnx_type}")
    return np.dtype(values[onnx_type])


def _token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    if encoded is None:
        raise GenerationError("Tokenizer did not return input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    while encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    result = [int(value) for value in encoded]
    if not result:
        raise ValueError("prompt must tokenize to at least one token")
    return result


class OnnxTextGenerator:
    """Generate text from a local ONNX decoder model and tokenizer directory."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        model_path: str | Path | None = None,
        tokenizer: Tokenizer | None = None,
        session: Any | None = None,
        metadata: Any | None = None,
        model_spec: Any | None = None,
    ) -> None:
        target = Path(model_dir).expanduser().resolve()
        directory = target if target.is_dir() else target.parent
        if session is None and model_spec is None:
            model_spec = load_llm_model(
                model_path if model_path is not None else target
            )
        if model_spec is not None and model_path is None:
            model_path = model_spec.model_path
        self.session = session or ort.InferenceSession(
            str(Path(model_path).expanduser().resolve()),
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = tokenizer or load_tokenizer(directory)
        source_metadata = metadata
        if source_metadata is None and model_spec is not None:
            source_metadata = model_spec.metadata
        self.model_metadata = build_model_metadata(self.session, source_metadata)
        self.metadata = self.model_metadata.cache
        inputs = {item.name: item for item in self.session.get_inputs()}
        self._input_ids = next((name for name in inputs if name.lower() == "input_ids"
                                or "input_ids" in name.lower()), None)
        self._attention_mask = next((name for name in inputs if "attention_mask" in name.lower()), None)
        self._position_ids = next((name for name in inputs if "position_ids" in name.lower()), None)
        self._inputs = inputs
        if self._input_ids is None:
            raise UnsupportedModelError("ONNX model is missing an input_ids input")

    def _encode(self, prompt: str) -> list[int]:
        try:
            encoded = encode_model_prompt(self.tokenizer, prompt)
            return _token_ids(encoded["input_ids"])
        except (TypeError, ValueError, KeyError):
            return _token_ids(self.tokenizer(prompt))

    def _cache_zeros(self, batch_size: int) -> dict[str, np.ndarray]:
        feeds: dict[str, np.ndarray] = {}
        for name in self.metadata.past_names:
            item = self._inputs[name]
            shape: list[int] = []
            for index, dimension in enumerate(item.shape):
                if isinstance(dimension, int) and dimension >= 0:
                    shape.append(dimension)
                elif index == 0:
                    shape.append(batch_size)
                elif index == 2 and len(item.shape) >= 4:
                    shape.append(0)
                else:
                    shape.append(1)
            feeds[name] = np.zeros(shape, dtype=_numpy_dtype(item.type))
        return feeds

    def _run(self, input_ids: list[int], attention_length: int, position: int,
             cache: dict[str, np.ndarray] | None, *, prefill: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if prefill:
            ids = np.asarray([input_ids], dtype=np.int64)
            feeds = create_prefill_feed(self.session, ids)
        else:
            feeds = self._decode_feed(input_ids[-1:], attention_length, position, cache)
        names = [item.name for item in self.session.get_outputs()]
        outputs = self.session.run(None, feeds)
        output_map = dict(zip(names, outputs))
        logits_name = self.model_metadata.logits_name
        logits = np.asarray(output_map[logits_name])
        present = {
            output: output_map[output]
            for _, output in self.metadata.input_to_output
            if output in output_map
        }
        return logits, present

    def _decode_feed(
        self,
        input_ids: list[int],
        attention_length: int,
        position: int,
        cache: dict[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray]:
        feeds: dict[str, np.ndarray] = {}
        item = self._inputs[self._input_ids]
        dtype = _numpy_dtype(item.type)
        feeds[self._input_ids] = np.asarray([input_ids], dtype=dtype)
        if self._attention_mask:
            mask_type = _numpy_dtype(self._inputs[self._attention_mask].type)
            feeds[self._attention_mask] = np.ones((1, attention_length), dtype=mask_type)
        if self._position_ids:
            pos_type = _numpy_dtype(self._inputs[self._position_ids].type)
            values = np.asarray([position], dtype=pos_type)
            feeds[self._position_ids] = values.reshape(1, -1)
        if cache:
            missing = set(self.metadata.past_names) - set(cache)
            if missing:
                raise IncompatibleModelError(
                    f"Missing cache tensors for decode: {sorted(missing)}"
                )
            feeds.update({name: cache[name] for name in self.metadata.past_names})
        if "use_cache_branch" in self._inputs:
            item = self._inputs["use_cache_branch"]
            if item.type != "tensor(bool)":
                raise IncompatibleModelError(
                    f"Expected boolean use_cache_branch input, got {item.type}"
                )
            feeds["use_cache_branch"] = np.asarray(True if len(item.shape) == 0 else [True])
        return feeds

    def generate(self, request: GenerationRequest) -> GenerationResult:
        config = request.config
        prompt_ids = self._encode(request.prompt)
        input_shape = self._inputs[self._input_ids].shape
        if len(input_shape) > 1 and isinstance(input_shape[1], int) and input_shape[1] > 0:
            if len(prompt_ids) > input_shape[1]:
                raise ValueError(
                    f"Prompt has {len(prompt_ids)} tokens, but the ONNX model "
                    f"supports at most {input_shape[1]}"
                )
        if config.use_cache and not self.metadata.available:
            raise IncompatibleModelError(
                "Cached generation requested, but the ONNX model exposes no compatible "
                "past_key_values/present_key_values interface; use use_cache=False "
                "or export the model with KV-cache inputs and outputs."
            )
        random.seed(config.seed)
        np.random.seed(config.seed)
        started = time.perf_counter()
        cache = self._cache_zeros(1) if config.use_cache else None
        logits, present_outputs = self._run(prompt_ids, len(prompt_ids), len(prompt_ids) - 1,
                                    cache, prefill=True)
        present = {
            past: present_outputs[present_name]
            for past, present_name in self.metadata.input_to_output
        }
        first_token_at = time.perf_counter()
        generated: list[int] = []
        eos = config.eos_token_id
        if eos is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        decode_started = time.perf_counter()
        for _ in range(config.max_new_tokens):
            token = int(np.argmax(logits[0, -1]))
            generated.append(token)
            if eos is not None and token == eos:
                break
            length = len(prompt_ids) + len(generated)
            if config.use_cache:
                logits, present_outputs = self._run(
                    [token], length, length - 1, present, prefill=False
                )
                present = {
                    past: present_outputs[present_name]
                    for past, present_name in self.metadata.input_to_output
                }
            else:
                logits, _ = self._run(
                    prompt_ids + generated, length, length - 1, None, prefill=True
                )
        finished = time.perf_counter()
        elapsed = finished - started
        decode_elapsed = max(0.0, finished - decode_started)
        decode_ms = decode_elapsed * 1000.0
        first_ms = (first_token_at - started) * 1000.0
        generated_count = len(generated)
        inter = decode_ms / max(generated_count, 1)
        tps = generated_count / decode_elapsed if decode_elapsed > 0 else 0.0
        all_ids = prompt_ids + generated
        return GenerationResult(
            text=decode_model_tokens(
                self.tokenizer, all_ids, skip_special_tokens=True
            ),
            token_ids=tuple(all_ids),
            prompt_token_ids=tuple(prompt_ids),
            generated_token_ids=tuple(generated),
            cache_used=config.use_cache,
            time_to_first_token_ms=first_ms,
            decode_latency_ms=decode_ms,
            inter_token_latency_ms=inter,
            tokens_per_second=tps,
            metadata=self.metadata,
            request_id=request.request_id,
            prompt_token_count=len(prompt_ids),
            generated_token_count=generated_count,
            prefill_latency_ms=first_ms,
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Reserved integration point for the batching contributor."""
        raise NotImplementedError(
            "Batch generation is not implemented yet; use generate(request) "
            "for single-request generation."
        )
