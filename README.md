# InferBench AutoTune

A cross-platform (macOS and Windows) tool that takes an ONNX model, tries different
ONNX Runtime settings, and reports the fastest setup that still gives correct outputs.

| Part | Owner | Where |
|------|-------|-------|
| Benchmark and correctness engine | Member 2 | `inferbench/benchmark/` |
| Optimization strategies | Member 3 | `src/inferbench_optimizers/` |
| Shared types (`BenchmarkConfig`, `BenchmarkResult`) | shared | `inferbench/common/` |

## Setup

Python 3.12 or newer.

```bash
python -m venv venv
# Windows PowerShell:  venv\Scripts\Activate.ps1
# macOS/Linux:         source venv/bin/activate

pip install -e ".[test]"
pytest
```

## Dashboard workflows

Start the FastAPI backend and Streamlit dashboard from the repository root:

```bash
uvicorn inferbench.api.main:app --host 127.0.0.1 --port 8000
streamlit run inferbench/dashboard/app.py
```

The dashboard provides model inspection and execution-provider diagnostics,
local greedy LLM generation with an optional KV cache, and the existing
AutoTune/single-benchmark workflows. Select an ONNX file from `models/` and
use **LLM Generation** to run a local tokenizer and ONNX decoder; the model
directory must contain the tokenizer files required by the selected export.

The MobileNet test model is not stored in git. To create it (needs the heavy `modelgen` extras):

```bash
pip install -e ".[modelgen]"
python scripts/makemodel.py
```

Tests that need the model are skipped if it is missing.

## Local ONNX LLM generation

`models.generation.OnnxTextGenerator` loads an ONNX decoder and a local
Hugging Face tokenizer, then performs deterministic greedy generation:

```python
from models.generation import GenerationConfig, GenerationRequest, OnnxTextGenerator

generator = OnnxTextGenerator("models/qwen2.5-0.5b")
result = generator.generate(GenerationRequest(
    "Explain KV cache in one sentence.",
    GenerationConfig(max_new_tokens=32),
))
print(result.text, result.tokens_per_second)
```

`GenerationRequest` also accepts an optional `request_id`, `priority`, and
`deadline_s`. Batched generation preserves each request ID and supports
independent EOS and generation limits:

```python
from models.batching import BatchConfig, DynamicBatcher

batch_config = BatchConfig(
    max_batch_size=8,
    max_total_batch_tokens=8192,
    max_padding_ratio=0.5,
    prompt_length_buckets=(128, 512, 2048, 4096),
)
results = generator.generate_batch(requests, batch_config)
print(generator.last_batch_metrics)
```

For an apples-to-apples load comparison, the benchmark helper reports request
throughput, generated-token throughput, mean latency, and p95 latency for single,
unbucketed, and bucketed execution at each requested concurrency:

```python
from models.batching import benchmark_batching

measurements = await benchmark_batching(generator, workload, [1, 4, 8], batch_config)
```

`generate_batch` forms same-bucket groups immediately. For concurrent callers,
`DynamicBatcher` collects requests during its configured window and resolves
each submitter with its own result:

```python
batcher = DynamicBatcher(
    lambda pending: generator.generate_batch(list(pending), batch_config),
    token_counter=lambda request: generator.count_prompt_tokens(request.prompt),
    config=batch_config,
)
# Submit requests with `await batcher.submit(request)`, then call `await batcher.close()`.
```

Set `prompt_length_buckets=None` to compare unbucketed batching with the default
bucketed policy. `BatchMetrics` reports effective batch size, prompt and padded
token counts, padding overhead, formation and execution time, and aggregate
generated tokens per second. Each `GenerationResult` retains per-request
latency and throughput measurements. `generate(request)` remains available for
single-request execution.

The directory must contain an `.onnx` file and tokenizer files such as
`tokenizer.json`/`tokenizer_config.json` and `vocab.json` or a SentencePiece
model. Cached generation requires the export to expose matching past-cache
inputs and present-cache outputs; use `GenerationConfig(use_cache=False)` for
an uncached export.

### Optional Hugging Face parity check

After downloading the original Qwen weights (about 1 GB), compare the first
20 greedy tokens for three fixed prompts:

```bash
python scripts/validate_qwen_generation.py
```

The pytest version is opt-in because it downloads/loads a large model:

```bash
RUN_QWEN_PARITY=1 python -m pytest -q tests/test_qwen_generation_parity.py
```

## Request scheduling and evaluation (Member 4)

`inferbench/scheduler/` puts a request scheduler in front of the batching and
generation code. `inferbench/benchmark/report.py` then compares the five
configurations from `division.md` and writes a JSON and Markdown report.

```python
from inferbench.scheduler import RequestScheduler, SchedulerConfig
from models.batching import BatchConfig

batch_config = BatchConfig(max_batch_size=4)
scheduler = RequestScheduler(
    lambda requests: generator.generate_batch(requests, batch_config),
    token_counter=lambda request: generator.count_prompt_tokens(request.prompt),
    config=SchedulerConfig(max_queue_length=32, default_timeout_s=60, max_batch_size=4),
    batch_metrics_source=lambda: generator.last_batch_metrics,
)
async with scheduler:
    result = await scheduler.submit(GenerationRequest("Hello", priority=1, deadline_s=30))
    scheduler.cancel("some-request-id")          # queued or in-flight
    print(scheduler.snapshot())                   # queue wait, TTFT, tokens/s, ...
```

Rules the scheduler enforces:

- FIFO by default; a **higher** `priority` is served first, ties stay FIFO.
- Aging (`aging_interval_s`): each interval of waiting adds one priority level, so
  low-priority requests cannot starve.
- Overload fails fast: a full queue, a too-long prompt or a too-long generation
  raises `QueueFullError` / `PromptTooLongError` / `GenerationTooLongError`.
- `deadline_s` (or `default_timeout_s`) bounds queue wait plus execution. A request
  that expires while queued never runs.
- `cancel(request_id)` removes a queued request. A request already inside a batch
  cannot be interrupted (the executor call blocks), so the caller is released
  at once and the late result is discarded.
- Lifecycle states: queued, batched, prefilling, decoding, completed, cancelled,
  timed_out, failed. `decoding` is only recorded if the executor calls
  `scheduler.report_progress(...)`.

Run the comparison (writes `results/llm_report_<model>_<time>.json` and `.md`):

```bash
python scripts/run_llm_benchmark.py --model-dir models/qwen2.5_0.5b
python scripts/run_llm_benchmark.py --scenario mixed_lengths --max-new-tokens 16 --scheduling-scenarios
```

Scenarios: `single`, `concurrent`, `short`, `long`, `mixed_lengths`,
`mixed_priorities`, `overload`. A configuration is only recommended when it passes
the quality checks (EOS behaviour, length limits, exact-match rate against the
baseline's tokens) **and** shows a measured benefit (tokens/s at least 1.10x the
baseline, or peak memory at most 1/1.10). The report also records the model
SHA-256, onnxruntime and Python versions, provider, hardware, seed, benchmark
configuration, git commit and timestamp.

The scheduler and report tests use a fake generator and need no model:

```bash
python -m pytest tests/test_scheduler.py tests/test_scheduler_e2e.py tests/test_llm_report.py tests/test_llm_correctness.py
```

## Benchmark engine (Member 2)

```python
from inferbench.common.benchmarktypes import BenchmarkConfig
from inferbench.benchmark.runner import make_baseline, evaluate_config
from inferbench.benchmark.export import save_results

config = BenchmarkConfig(model_path="models/mobilenet_v2.onnx")
baseline = make_baseline(config)                  # run once per model
result = evaluate_config(config, baseline)        # benchmark + correctness check
save_results([baseline.result, result])           # writes results/<model>_<time>.json
```

- `evaluate_config` never raises. A crash gives `success=False`; wrong outputs give
  `correct=False`. Only results with `is_valid == True` should be recommended.
- Latency (mean, p50, p90, p95, p99), throughput and peak memory are measured per run.

## Optimization library (Member 3)

### Included strategies

- ONNX Runtime graph optimization levels
- CPU intra-op thread tuning
- ONNX Runtime sequential/parallel execution mode
- Batch-size candidates
- Precision candidates gated by declared capabilities
- ONNX Runtime memory-pattern candidates
- Compilation strategy slot gated on an explicit compiler adapter

### Quick example

```python
from inferbench_optimizers import OptimizationContext, default_registry

context = OptimizationContext(
    model_format="onnx",
    runtime="onnxruntime",
    execution_providers=("CPUExecutionProvider",),
    logical_cpu_count=8,
    supports_int8=True,
    metadata={"max_batch_size": 8},
)

registry = default_registry()
candidates, skipped = registry.generate_candidates(context)

print(f"Generated {len(candidates)} candidates")
for candidate in candidates:
    print(candidate.strategy, candidate.config)

print("Skipped strategies:", skipped)
```

To benchmark the generated candidates, pass them through the integration
adapter. It creates a `BenchmarkConfig` for each candidate, reuses one
correctness baseline, and selects only valid results:

```python
from inferbench.common.benchmarktypes import BenchmarkConfig
from inferbench_optimizers import benchmark_candidates, select_best

base_config = BenchmarkConfig(model_path="models/model.onnx")
results = benchmark_candidates(base_config, candidates)
best = select_best(results, metric="latency_p50_ms")
```

### Safety and correctness boundaries

- Candidate generation is not proof that a configuration improves performance.
- Precision candidates only appear when capability flags are explicitly set.
- Quantization candidates require a separate artifact transformation adapter and task-specific quality validation.
- Compilation is skipped unless a compiler adapter is supplied.
- The benchmark runner must validate correctness and measure performance before any candidate is recommended.

### Next integration steps

1. Align `OptimizationContext` with the team's shared `ModelSpec` and hardware schemas.
2. Add adapters that translate candidate dictionaries to ONNX Runtime session options.
3. Connect Member 2's benchmark interface.
4. Add constraint-aware ranking and report generation.
5. Add further strategy plugins incrementally.