# InferBench Optimization Library — Member 3

This is the initial plugin-based optimization strategy library. It generates candidates and filters them by declared capabilities. It does **not** execute or benchmark a model yet; Member 2's benchmark runner will evaluate the generated candidates.

## Included strategies

- ONNX Runtime graph optimization levels
- CPU intra-op thread tuning
- ONNX Runtime sequential/parallel execution mode
- Batch-size candidates
- Precision candidates gated by declared capabilities
- ONNX Runtime memory-pattern candidates
- Compilation strategy slot gated on an explicit compiler adapter

## Install

Python 3.12+:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install -e ".[test]"
pytest
```

Install ONNX dependencies separately if needed:

```bash
pip install -e ".[onnx]"
```

## Quick example

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

## Safety and correctness boundaries

- Candidate generation is not proof that a configuration improves performance.
- Precision candidates only appear when capability flags are explicitly set.
- Quantization candidates require a separate artifact transformation adapter and task-specific quality validation.
- Compilation is skipped unless a compiler adapter is supplied.
- The benchmark runner must validate correctness and measure performance before any candidate is recommended.

## Next integration steps

1. Align `OptimizationContext` with the team's shared `ModelSpec` and hardware schemas.
2. Add adapters that translate candidate dictionaries to ONNX Runtime session options.
3. Connect Member 2's benchmark interface.
4. Add constraint-aware ranking and report generation.
5. Add further strategy plugins incrementally.
