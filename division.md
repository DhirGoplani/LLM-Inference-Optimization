# LLM Inference Optimization Engine: Team Division

This document divides the work between four people to turn the existing
ONNX Runtime optimization prototype into an LLM inference optimization engine.
The scope includes Llama 3.2 1B, Qwen2.5 0.5B, and Gemma 2 2B where compatible
ONNX exports are available.

## Project boundaries

### Included

- Local ONNX decoder-only LLM inference
- Hugging Face tokenizer loading
- Greedy autoregressive generation
- KV-cache reuse
- Dynamic batching
- Prompt-length bucketing
- Request scheduling
- TTFT, decode latency, token throughput, memory, and queue metrics
- Correctness and quality validation
- Existing ONNX Runtime graph, thread, batching, and memory optimizations

### Excluded from this phase

- INT8 and INT4 quantization
- A mandatory vLLM backend
- Training or fine-tuning
- Distributed inference across multiple machines
- Replacing ONNX Runtime with a custom CUDA kernel implementation

vLLM and paged KV caching may be documented as future work. The initial
implementation should remain compatible with the project's ONNX Runtime
architecture.

## Shared contracts to agree on first

All contributors should agree on these contracts before implementing
cross-cutting code:

```python
generate(request: GenerationRequest) -> GenerationResult
generate_batch(requests: list[GenerationRequest]) -> list[GenerationResult]
```

The shared request and result types should contain:

- Prompt text or pre-tokenized input
- Maximum new tokens
- EOS and padding behavior
- Optional seed
- Request ID
- Priority
- Deadline or timeout
- Generated text and token IDs
- Prompt-token count and generated-token count
- TTFT
- Decode latency
- Inter-token latency
- Tokens per second
- Error information

The model metadata contract should expose:

- Model family and architecture
- Input and output tensor names
- Tokenizer location
- EOS and PAD token IDs
- Number of layers, attention heads, KV heads, and head dimension
- Maximum sequence length
- Whether past/present KV-cache tensors are available
- Cache tensor names, shapes, and data types

No contributor should hard-code tensor names for only one model. Model-specific
name mappings belong in the model adapter or metadata layer.

## Person 1: Model integration and compatibility

### Objective

Make the three target LLMs discoverable, loadable, and verifiable as local ONNX
models without breaking the existing generic ONNX model workflow.

### Responsibilities

1. Define or extend the LLM model specification.
2. Locate an ONNX graph and external `.onnx.data` files in a model directory.
3. Load local tokenizer files using the Hugging Face tokenizer interface.
4. Inspect ONNX inputs and outputs.
5. Identify:
   - `input_ids`
   - `attention_mask`
   - `position_ids`
   - past KV-cache inputs
   - logits
   - present KV-cache outputs
6. Record architecture metadata:
   - model family
   - hidden size
   - layer count
   - attention and KV-head counts
   - head dimension
   - context length
   - EOS/PAD token IDs
7. Verify that every model can create an ONNX Runtime session.
8. Produce clear errors for:
   - missing model files
   - missing tokenizer files
   - invalid ONNX graphs
   - unsupported model architectures
   - incompatible cache interfaces
9. Provide a small fixture or metadata-only test model for CI. Large model
   downloads must not be required by the test suite.
10. Document model export assumptions and local directory layout.

### Suggested ownership

- `models/schema.py`
- `models/loader.py`
- New model metadata/inspection modules
- Model compatibility tests
- Relevant dependency and documentation updates

### Acceptance criteria

- Each available target model passes model and tokenizer inspection.
- The loader reports all required generation and cache tensors.
- Existing MobileNet and generic ONNX tests still pass.
- Tests skip cleanly when large model directories are absent.
- A model report can explain why a model is supported or rejected.

### Dependencies

This work can begin immediately. Person 2 needs the metadata contract, but
Person 1 can use a synthetic ONNX fixture while the final model exports are
being prepared.

## Person 2: Generation and KV-cache reuse

### Objective

Implement deterministic decoder-only text generation over ONNX Runtime and
reuse KV-cache tensors during decoding.

### Responsibilities

1. Define typed:
   - `GenerationConfig`
   - `GenerationRequest`
   - `GenerationResult`
   - cache/model adapter metadata
2. Load a local tokenizer and encode prompts.
3. Implement a prefill pass for the full prompt.
4. Implement one-token-at-a-time decode passes.
5. Select the next token with greedy argmax decoding.
6. Stop on EOS or `max_new_tokens`.
7. Decode generated token IDs back to text.
8. Support:
   - attention masks
   - position IDs
   - model-specific input names
   - multiple cache layers
   - dynamic and static cache shapes where possible
9. Reuse returned present-cache tensors as past-cache inputs on the next step.
10. Provide an explicit uncached mode for compatible comparison models.
11. Measure:
   - prefill latency
   - TTFT
   - decode latency
   - inter-token latency
   - generated token count
   - tokens per second
12. Validate:
   - EOS behavior
   - maximum token limits
   - empty and oversized prompts
   - missing input/output tensors
   - cache shape mismatches
   - deterministic output with greedy decoding
13. Do not implement dynamic batching or scheduling; expose a clean single
   request API for Person 3 and Person 4.

### Suggested ownership

- `models/generation.py`
- `inferbench/benchmark/llm.py`
- Generation-specific correctness helpers
- Generation unit and integration tests

### Acceptance criteria

- A local supported model generates text from a prompt.
- Cached decoding does not resend the complete prompt for every token.
- Cached and uncached results match under deterministic greedy decoding where
  both modes are available.
- Generation failures are explicit and actionable.
- Existing generic benchmark behavior remains unchanged.

### Dependencies

Needs Person 1's metadata contract, but can develop against a synthetic
decoder fixture and a mocked tokenizer interface in parallel.

## Person 3: Dynamic batching and prompt-length bucketing

### Objective

Increase serving throughput by forming efficient batches while preserving
per-request results and KV-cache state.

### Responsibilities

1. Implement a batch request representation.
2. Add configurable:
   - maximum batch size
   - maximum total batch tokens
   - batching window
   - maximum padding ratio
3. Implement dynamic batching for requests arriving during the batching
   window.
4. Implement prompt-length buckets, for example:
   - 1-128 tokens
   - 129-512 tokens
   - 513-2048 tokens
   - 2049-4096 tokens
5. Pad only requests within the same bucket.
6. Build correct attention masks and position IDs for padded batches.
7. Maintain independent generated-token state and KV-cache state for every
   request.
8. Remove completed requests from an active batch without corrupting remaining
   requests.
9. Demultiplex logits and generated text back to the correct request ID.
10. Measure:
    - batch formation time
    - effective batch size
    - padding overhead
    - aggregate tokens per second
    - per-request latency
    - throughput at different concurrency levels
11. Compare:
    - single-request execution
    - unbucketed dynamic batching
    - bucketed dynamic batching
12. Test mixed prompt lengths and requests that finish at different times.

### Suggested ownership

- New batching package under `inferbench/` or `models/`
- Batch construction and padding utilities
- Batching tests and load fixtures
- Benchmark workload definitions for short, medium, and long prompts

### Acceptance criteria

- Requests are combined without exceeding configured batch limits.
- Padding is restricted to the selected prompt bucket.
- Every request receives only its own output.
- A completed request does not block the rest of the batch.
- Batching produces measurable throughput and padding metrics.
- The single-request generation API remains usable.

### Dependencies

Needs Person 2's generation and cache interfaces for final integration.
Batch formation, bucketing, padding, and demultiplexing can be developed
independently using fake generation callbacks.

## Person 4: Request scheduling, evaluation, and reporting

### Objective

Provide reliable request control and prove that the optimizations improve
performance without unacceptable quality or correctness regressions.

### Responsibilities

1. Implement a request scheduler with:
   - FIFO behavior as the default
   - priority levels
   - maximum queue length
   - request timeout and deadline
   - cancellation
   - maximum prompt length
   - maximum generation length
   - fair scheduling or starvation prevention
   - overload rejection
2. Connect scheduler output to Person 3's batching interface.
3. Track request lifecycle states:
   - queued
   - batched
   - prefilling
   - decoding
   - completed
   - cancelled
   - timed out
   - failed
4. Add metrics for:
   - queue wait time
   - service time
   - TTFT
   - inter-token latency
   - end-to-end latency
   - requests per second
   - tokens per second
   - active requests
   - rejected and timed-out requests
5. Extend correctness and quality evaluation beyond generic tensor equality:
   - token agreement
   - logits agreement where available
   - EOS correctness
   - output length constraints
   - NaN/Inf checks
   - deterministic greedy output checks
6. Build benchmark scenarios:
   - one request
   - concurrent requests
   - short prompts
   - long prompts
   - mixed prompt lengths
   - mixed priorities
   - queue overload
7. Generate a final comparison report containing:
   - baseline
   - KV-cache reuse
   - dynamic batching
   - prompt bucketing
   - scheduling
   - combined configuration
8. Record environment and reproducibility information:
   - model hash
   - runtime version
   - provider
   - hardware
   - Python version
   - benchmark configuration
   - seed
   - timestamp

### Suggested ownership

- New scheduler/service modules under `inferbench/`
- `inferbench/benchmark/correctness.py`
- Benchmark export/reporting modules
- Scheduler, load, quality, and end-to-end tests

### Acceptance criteria

- Requests obey queue, priority, timeout, and cancellation rules.
- Overloaded workloads fail explicitly rather than hanging.
- The report compares latency, throughput, memory, and quality.
- Results are reproducible and include environment metadata.
- The final report demonstrates a meaningful improvement over baseline.

### Dependencies

The scheduler can be developed against a fake batch executor. Final wiring
depends on Person 3's batch API and Person 2's result/metric types.

## Parallel work plan

All four people can begin in parallel after agreeing on the shared contracts.

```text
Person 1: model metadata and fixtures  ─────┐
Person 2: generation and KV cache       ────┼── Integration
Person 3: batching and bucketing        ────┤
Person 4: scheduler and evaluation      ───┘
```

### Integration checkpoints

1. **Contract checkpoint**
   - Agree on request, result, model metadata, and error types.
2. **Single-model checkpoint**
   - Person 1 verifies Qwen2.5-0.5B or another small model.
   - Person 2 generates a response from one request.
3. **Cache checkpoint**
   - Cached decode produces valid output and metrics.
4. **Batch checkpoint**
   - Person 3 runs several requests through dynamic batching.
5. **Scheduling checkpoint**
   - Person 4 applies queue, priority, timeout, and cancellation rules.
6. **Final evaluation checkpoint**
   - Run all three models where compatible and publish baseline versus
     optimized results.

## Suggested branch ownership

Use separate branches to reduce conflicts:

```text
feature/model-integration
feature/generation-kv-cache
feature/dynamic-batching
feature/scheduling-evaluation
```

Avoid editing another person's owned modules unless the interface contract
requires it. Put shared type changes in a small, reviewed module and notify
all contributors before changing it.

## Final demonstration

Use the following configurations:

```text
1. ONNX Runtime baseline, single request
2. Baseline + KV-cache reuse
3. KV-cache + dynamic batching
4. KV-cache + dynamic batching + prompt bucketing
5. All optimizations + request scheduling
```

For each configuration report:

- TTFT
- p50 and p95 end-to-end latency
- inter-token latency
- tokens per second
- requests per second
- peak memory
- queue wait time
- effective batch size
- padding overhead
- generated-token correctness or quality

The final claim should be evidence-based: recommend a configuration only when
it passes correctness and quality constraints and demonstrates a measured
performance or memory benefit over the baseline.
