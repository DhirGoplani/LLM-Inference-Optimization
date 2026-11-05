import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from inferbench.benchmark.correctness import compare_generated_tokens
from models.batching import (
    BatchConfig,
    DynamicBatcher,
    benchmark_batching,
    form_batch_indices,
)
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


class VariableTokenizer(FakeTokenizer):
    def __call__(self, text, **kwargs):
        return {"input_ids": {"short": [1, 5], "long": [1, 2, 6]}[text]}


class BatchFakeSession:
    def __init__(self):
        self.cached = True
        self.calls = []
        self._inputs = [
            SimpleNamespace(name="input_ids", type="tensor(int64)", shape=["batch", "sequence"]),
            SimpleNamespace(name="attention_mask", type="tensor(int64)", shape=["batch", "sequence"]),
            SimpleNamespace(name="position_ids", type="tensor(int64)", shape=["batch", "sequence"]),
            SimpleNamespace(name="past_key_values.0.key", type="tensor(float)",
                             shape=["batch", 1, 0, 2]),
            SimpleNamespace(name="past_key_values.0.value", type="tensor(float)",
                             shape=["batch", 1, 0, 2]),
        ]
        self._outputs = [
            SimpleNamespace(name="logits", type="tensor(float)", shape=["batch", "sequence", 10]),
            SimpleNamespace(name="present.0.key", type="tensor(float)",
                             shape=["batch", 1, "sequence", 2]),
            SimpleNamespace(name="present.0.value", type="tensor(float)",
                             shape=["batch", 1, "sequence", 2]),
        ]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _, feeds):
        self.calls.append(feeds)
        input_ids = feeds["input_ids"]
        batch_size, sequence_length = input_ids.shape
        logits = np.full((batch_size, sequence_length, 10), -1.0, dtype=np.float32)
        past = feeds["past_key_values.0.key"]
        present = np.zeros(
            (batch_size, 1, past.shape[2] + sequence_length, 2), dtype=np.float32
        )
        if past.shape[2]:
            present[:, :, :past.shape[2], :] = past
            markers = past[:, 0, 0, 0].astype(int)
            counts = past[:, 0, -1, 1].astype(int) + 1
        else:
            mask = feeds["attention_mask"]
            markers = np.asarray([
                input_ids[row, np.flatnonzero(mask[row])[-1]]
                for row in range(batch_size)
            ])
            counts = np.zeros(batch_size, dtype=int)
        for row, (marker, count) in enumerate(zip(markers, counts)):
            if marker == 5:
                token = 3 if count == 0 else 9
            else:
                token = (7, 8, 9)[min(count, 2)]
            logits[row, -1, token] = 2.0
            present[row, 0, 0, 0] = marker
            present[row, 0, -1, 1] = count
        return [logits, present, present.copy()]


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
    assert generator.generate_batch([]) == []


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


def test_batch_generation_preserves_outputs_masks_positions_and_cache_rows():
    session = BatchFakeSession()
    generator = OnnxTextGenerator(
        "/tmp", tokenizer=VariableTokenizer(), session=session
    )
    results = generator.generate_batch([
        GenerationRequest(
            "short",
            GenerationConfig(max_new_tokens=3),
            request_id="short-id",
        ),
        GenerationRequest(
            "long",
            GenerationConfig(max_new_tokens=3),
            request_id="long-id",
        ),
    ])

    assert [result.request_id for result in results] == ["short-id", "long-id"]
    assert results[0].generated_token_ids == (3, 9)
    assert results[1].generated_token_ids == (7, 8, 9)
    assert session.calls[0]["input_ids"].tolist() == [[0, 1, 5], [1, 2, 6]]
    assert session.calls[0]["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert session.calls[0]["position_ids"].tolist() == [[0, 0, 1], [0, 1, 2]]
    assert session.calls[1]["input_ids"].shape == (2, 1)
    assert session.calls[2]["input_ids"].shape == (1, 1)
    assert generator.last_batch_metrics[0].effective_batch_size == 2
    assert generator.last_batch_metrics[0].padding_overhead_ratio == pytest.approx(0.2)


def test_bucketed_batch_results_match_single_request_generation():
    prompts = [
        GenerationRequest(
            "short", GenerationConfig(max_new_tokens=3), request_id="short-id"
        ),
        GenerationRequest(
            "long", GenerationConfig(max_new_tokens=3), request_id="long-id"
        ),
    ]
    single = OnnxTextGenerator(
        "/tmp", tokenizer=VariableTokenizer(), session=BatchFakeSession()
    )
    expected = [single.generate(request) for request in prompts]

    batched = OnnxTextGenerator(
        "/tmp", tokenizer=VariableTokenizer(), session=BatchFakeSession()
    )
    bucketed = batched.generate_batch(prompts)
    unbucketed = batched.generate_batch(
        prompts,
        BatchConfig(
            max_batch_size=4,
            max_total_batch_tokens=20,
            max_padding_ratio=1.0,
            prompt_length_buckets=None,
        ),
    )

    assert [result.generated_token_ids for result in bucketed] == [
        result.generated_token_ids for result in expected
    ]
    assert [result.generated_token_ids for result in unbucketed] == [
        result.generated_token_ids for result in expected
    ]


def test_batch_formation_enforces_bucket_token_and_padding_limits():
    config = BatchConfig(
        max_batch_size=3,
        max_total_batch_tokens=10,
        max_padding_ratio=0.5,
        prompt_length_buckets=(4, 8),
    )
    assert form_batch_indices([2, 3, 7, 8], config) == [(0, 1), (2,), (3,)]
    unbucketed = BatchConfig(
        max_batch_size=3,
        max_total_batch_tokens=20,
        max_padding_ratio=1.0,
        prompt_length_buckets=None,
    )
    assert form_batch_indices([2, 3, 7], unbucketed) == [(0, 1), (2,)]


def test_dynamic_batcher_combines_concurrent_requests_and_returns_own_results():
    observed_batches = []

    def execute_batch(requests):
        observed_batches.append([request.request_id for request in requests])
        return [
            SimpleNamespace(request_id=request.request_id, generated_token_count=1)
            for request in requests
        ]

    batcher = DynamicBatcher(
        execute_batch,
        token_counter=lambda request: len(request.prompt),
        config=BatchConfig(
            max_batch_size=4,
            max_total_batch_tokens=20,
            batching_window_ms=30,
            max_padding_ratio=1.0,
            prompt_length_buckets=(4, 8),
        ),
    )

    async def run_requests():
        tasks = [
            asyncio.create_task(batcher.submit(GenerationRequest(
                prompt, GenerationConfig(use_cache=False), request_id=request_id
            )))
            for prompt, request_id in [
                ("ab", "one"),
                ("abc", "two"),
                ("1234567", "three"),
            ]
        ]
        results = await asyncio.gather(*tasks)
        await batcher.close()
        return results

    results = asyncio.run(run_requests())
    assert [result.request_id for result in results] == ["one", "two", "three"]
    assert observed_batches == [["one", "two"], ["three"]]
    assert [metric.effective_batch_size for metric in batcher.last_batch_metrics] == [2, 1]


def test_batching_benchmark_compares_modes_and_concurrency_levels():
    generator = OnnxTextGenerator(
        "/tmp", tokenizer=VariableTokenizer(), session=BatchFakeSession()
    )
    requests = [
        GenerationRequest(
            prompt,
            GenerationConfig(max_new_tokens=2),
            request_id=request_id,
        )
        for prompt, request_id in [("short", "one"), ("long", "two")]
    ]

    results = asyncio.run(benchmark_batching(
        generator,
        requests,
        [1, 2],
        BatchConfig(batching_window_ms=1),
    ))

    assert [(result.mode, result.concurrency) for result in results] == [
        ("single", 1),
        ("single", 2),
        ("unbucketed", 1),
        ("unbucketed", 2),
        ("bucketed", 1),
        ("bucketed", 2),
    ]
    assert all(result.request_count == 2 for result in results)
    assert all(result.generated_tokens_per_second > 0 for result in results)
