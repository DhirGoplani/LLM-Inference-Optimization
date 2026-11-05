"""End-to-end scheduler tests: load, overload, priorities, timeouts.

Most use FakeGenerator (no model). The last test uses the real Qwen ONNX model and
is skipped when models/qwen2.5_0.5b/model.onnx is absent.
"""

import asyncio
from pathlib import Path

import pytest

from inferbench.benchmark.correctness import evaluate_generation_quality
from inferbench.benchmark.report import run_scheduler_scenario
from inferbench.benchmark.scenarios import build_scenario
from inferbench.scheduler import RequestScheduler, SchedulerConfig
from models.batching import BatchConfig
from models.generation import GenerationConfig, GenerationRequest
from test.fake_llm import FakeGenerator

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen2.5_0.5b"


def run(coro):
    return asyncio.run(coro)


def test_overload_is_rejected_explicitly_and_nothing_hangs():
    scenario = build_scenario("overload", max_new_tokens=2, num_requests=12)
    out = run(run_scheduler_scenario(
        FakeGenerator(), scenario, BatchConfig(max_batch_size=2, batching_window_ms=0.0)
    ))
    assert out["hung"] is False
    assert out["scheduler_config"]["max_queue_length"] == 3
    assert out["rejected"] == 9
    assert out["outcome_counts"] == {"completed": 3, "rejected:queue_full": 9}
    assert out["metrics"]["counts"]["rejected_by_reason"] == {"queue_full": 9}
    assert sum(out["outcome_counts"].values()) == 12
    assert len(out["request_outcomes"]) == 12
    assert sum(item["outcome"] == "completed" for item in out["request_outcomes"]) == 3
    assert sum(item["outcome"] == "rejected:queue_full" for item in out["request_outcomes"]) == 9


def test_scheduler_report_records_a_cancelled_request():
    scenario = build_scenario("concurrent", max_new_tokens=2, num_requests=3)
    out = run(run_scheduler_scenario(
        FakeGenerator(cached_s=0.01),
        scenario,
        BatchConfig(max_batch_size=1, batching_window_ms=0.0),
        cancel_request_id="concurrent-000",
    ))
    outcomes = {item["request_id"]: item["outcome"] for item in out["request_outcomes"]}
    assert outcomes["concurrent-000"] == "cancelled"
    assert out["outcome_counts"]["cancelled"] == 1


def test_high_priority_requests_finish_first_in_a_burst():
    scenario = build_scenario("mixed_priorities", max_new_tokens=2)
    out = run(run_scheduler_scenario(
        FakeGenerator(), scenario, BatchConfig(max_batch_size=1, batching_window_ms=0.0),
        scheduler_overrides={"aging_interval_s": None},
    ))
    order = [item["priority"] for item in out["completion_order"]]
    assert len(order) == 9
    assert order[:3] == [5, 5, 5]
    waits = out["mean_queue_wait_ms_by_priority"]
    assert waits["5"] < waits["0"]


def test_timeouts_under_load_end_explicitly():
    scenario = build_scenario("mixed_priorities", max_new_tokens=4)
    out = run(run_scheduler_scenario(
        FakeGenerator(cached_s=0.01), scenario,
        BatchConfig(max_batch_size=1, batching_window_ms=0.0),
        scheduler_overrides={"default_timeout_s": 0.06, "aging_interval_s": None},
    ))
    counts = out["outcome_counts"]
    assert out["hung"] is False
    assert counts.get("timed_out", 0) >= 1 and counts.get("completed", 0) >= 1
    assert sum(counts.values()) == 9
    assert out["metrics"]["counts"]["timed_out"] == counts["timed_out"]


def test_many_concurrent_clients_each_get_their_own_result():
    generator = FakeGenerator(cached_s=0.0005)

    async def go():
        scheduler = RequestScheduler(
            lambda reqs: generator.generate_batch(reqs, None),
            token_counter=lambda r: generator.count_prompt_tokens(r.prompt),
            config=SchedulerConfig(max_batch_size=8, batching_window_ms=2.0, max_queue_length=200),
            batch_metrics_source=lambda: generator.last_batch_metrics,
        )
        requests = [
            GenerationRequest(
                f"prompt number {i} " + "word " * (i % 7),
                GenerationConfig(max_new_tokens=3),
                request_id=f"load-{i}",
                priority=i % 4,
            )
            for i in range(60)
        ]
        results = await asyncio.gather(*(scheduler.submit(r) for r in requests))
        await scheduler.close()
        return requests, results, scheduler.snapshot()

    requests, results, snap = run(go())
    assert [r.request_id for r in results] == [r.request_id for r in requests]
    assert snap["counts"]["completed"] == 60 and snap["counts"]["failed"] == 0
    assert snap["batching"]["max_effective_batch_size"] > 1
    assert snap["batching"]["max_effective_batch_size"] <= 8
    assert snap["active"]["queued"] == 0 and snap["active"]["in_flight"] == 0


@pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").is_file(),
    reason="models/qwen2.5_0.5b/model.onnx is not present",
)
def test_real_qwen_through_the_scheduler_keeps_each_requests_own_output():
    from models.generation import OnnxTextGenerator

    generator = OnnxTextGenerator(MODEL_DIR)
    prompts = ["The capital of France is", "The sky is", "Explain caching in one line."]
    requests = [
        GenerationRequest(p, GenerationConfig(max_new_tokens=4), request_id=f"q{i}", priority=i)
        for i, p in enumerate(prompts)
    ]
    singles = [generator.generate(r) for r in requests]
    batch_config = BatchConfig(max_batch_size=4, batching_window_ms=20.0)

    async def go():
        scheduler = RequestScheduler(
            lambda reqs: generator.generate_batch(reqs, batch_config),
            token_counter=lambda r: generator.count_prompt_tokens(r.prompt),
            config=SchedulerConfig(max_batch_size=4, batching_window_ms=20.0),
            batch_metrics_source=lambda: generator.last_batch_metrics,
        )
        results = await asyncio.gather(*(scheduler.submit(r) for r in requests))
        await scheduler.close()
        return results

    results = run(go())
    assert [r.request_id for r in results] == ["q0", "q1", "q2"]
    quality = evaluate_generation_quality(
        singles, results, eos_token_id=getattr(generator.tokenizer, "eos_token_id", None),
        max_new_tokens=4, min_exact_match_rate=0.5,
    )
    assert quality["passed"], quality