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
`deadline_s` so future scheduling and batching layers can preserve request
metadata. `generate_batch(...)` is intentionally a not-yet-implemented
integration stub; dynamic batching belongs to the batching contributor.

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