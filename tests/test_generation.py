from types import SimpleNamespace

import numpy as np
import pytest

from inferbench.benchmark.correctness import compare_generated_tokens
from models.generation import (
    GenerationConfig,
    GenerationRequest,
    IncompatibleModelError,
    MissingTokenizerError,
    OnnxTextGenerator,
    build_model_metadata,
    inspect_cache_interface,
    load_tokenizer,
)


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2]}

    def decode(self, token_ids, **kwargs):
        return " ".join(map(str, token_ids))


class FakeSession:
    def __init__(self, cached):
        self.cached = cached
        self.calls = []
        inputs = [
            SimpleNamespace(name="input_ids", type="tensor(int64)", shape=["batch", "sequence"]),
            SimpleNamespace(name="attention_mask", type="tensor(int64)", shape=["batch", "sequence"]),
            SimpleNamespace(name="position_ids", type="tensor(int64)", shape=["batch", "sequence"]),
        ]
        outputs = [SimpleNamespace(name="logits", type="tensor(float)", shape=["batch", "sequence", 10])]
        if cached:
            inputs.extend([
                SimpleNamespace(name="past_key_values.0.key", type="tensor(float)",
                                shape=[1, 1, 0, 2]),
                SimpleNamespace(name="past_key_values.0.value", type="tensor(float)",
                                shape=[1, 1, 0, 2]),
            ])
            outputs.extend([
                SimpleNamespace(name="present.0.key", type="tensor(float)",
                                shape=[1, 1, "sequence", 2]),
                SimpleNamespace(name="present.0.value", type="tensor(float)",
                                shape=[1, 1, "sequence", 2]),
            ])
        self._inputs = inputs
        self._outputs = outputs

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _, feeds):
        self.calls.append(feeds)
        ids = feeds["input_ids"][0]
        next_token = 3 if len(ids) == 2 else 4
        logits = np.full((1, len(ids), 10), -1.0, dtype=np.float32)
        logits[0, -1, next_token] = 2.0
        if len(ids) > 3:
            logits[0, -1, 9] = 3.0
        result = [logits]
        if self.cached:
            result.extend([
                np.zeros((1, 1, len(ids), 2), dtype=np.float32),
                np.zeros((1, 1, len(ids), 2), dtype=np.float32),
            ])
        return result


def test_generation_config_and_request_validate():
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationConfig(max_new_tokens=-1)
    with pytest.raises(ValueError, match="empty"):
        GenerationRequest("   ")


def test_long_prompt_rejected_for_static_sequence():
    session = FakeSession(cached=False)
    session.get_inputs()[0].shape = [1, 1]
    generator = OnnxTextGenerator("/tmp", tokenizer=FakeTokenizer(), session=session)
    with pytest.raises(ValueError, match="supports at most"):
        generator.generate(GenerationRequest(
            "hello", GenerationConfig(use_cache=False)
        ))


def test_missing_tokenizer_files_are_clear(tmp_path):
    with pytest.raises(MissingTokenizerError, match="tokenizer files"):
        load_tokenizer(tmp_path)


def test_uncached_generation_respects_eos_and_limit():
    session = FakeSession(cached=False)
    generator = OnnxTextGenerator("/tmp", tokenizer=FakeTokenizer(), session=session)
    result = generator.generate(GenerationRequest("hello", GenerationConfig(
        max_new_tokens=2, use_cache=False,
    )))
    assert result.generated_token_ids == (3, 4)
    assert len(session.calls[-1]["input_ids"][0]) == 4


def test_eos_terminates_before_max_new_tokens():
    generator = OnnxTextGenerator("/tmp", tokenizer=FakeTokenizer(), session=FakeSession(cached=False))
    result = generator.generate(GenerationRequest(
        "hello", GenerationConfig(max_new_tokens=5, use_cache=False)
    ))
    assert result.generated_token_ids == (3, 4, 9)


def test_cached_and_uncached_generation_match():
    cached = OnnxTextGenerator("/tmp", tokenizer=FakeTokenizer(), session=FakeSession(cached=True))
    uncached = OnnxTextGenerator("/tmp", tokenizer=FakeTokenizer(), session=FakeSession(cached=False))
    request = GenerationRequest("hello", GenerationConfig(max_new_tokens=2, use_cache=True))
    cached_result = cached.generate(request)
    uncached_result = uncached.generate(
        GenerationRequest("hello", GenerationConfig(max_new_tokens=2, use_cache=False))
    )
    assert compare_generated_tokens(cached_result, uncached_result)["passed"]
    assert len(cached.session.calls[1]["input_ids"][0]) == 1


def test_cache_interface_and_missing_cache_error():
    no_cache = FakeSession(cached=False)
    assert not inspect_cache_interface(no_cache).available
    generator = OnnxTextGenerator("/tmp", tokenizer=FakeTokenizer(), session=no_cache)
    with pytest.raises(IncompatibleModelError, match="no compatible"):
        generator.generate(GenerationRequest("hello"))


def test_request_metadata_and_batch_boundary():
    generator = OnnxTextGenerator("/tmp", tokenizer=FakeTokenizer(), session=FakeSession(False))
    result = generator.generate(GenerationRequest(
        "hello",
        GenerationConfig(max_new_tokens=1, use_cache=False),
        request_id="req-1",
        priority=4,
        deadline_s=2.0,
    ))
    assert result.request_id == "req-1"
    assert result.prompt_token_count == 2
    assert result.generated_token_count == 1
    assert result.prefill_latency_ms >= 0
    with pytest.raises(NotImplementedError, match="Batch generation"):
        generator.generate_batch([])


def test_loader_metadata_can_be_adapted():
    session = FakeSession(False)
    source = SimpleNamespace(
        architecture="qwen2",
        context_length=4096,
        eos_token_id=9,
    )
    metadata = build_model_metadata(session, source)
    assert metadata.model_family == "qwen2"
    assert metadata.max_sequence_length == 4096
    assert metadata.eos_token_id == 9
