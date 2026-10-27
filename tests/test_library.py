from inferbench_optimizers import OptimizationContext, default_registry


def test_registry_has_initial_strategies():
    registry = default_registry()
    names = registry.names()
    assert "graph_optimization" in names
    assert "cpu_thread_tuning" in names
    assert "batch_size" in names
    assert "precision_mode" in names


def test_onnx_cpu_context_generates_candidates():
    context = OptimizationContext(
        model_format="onnx",
        runtime="onnxruntime",
        execution_providers=("CPUExecutionProvider",),
        logical_cpu_count=8,
        supports_int8=True,
        supports_fp16=True,
    )
    candidates, skipped = default_registry().generate_candidates(context)
    assert candidates
    assert not any(c.strategy == "compilation" for c in candidates)
    assert any(c.config.get("graph_optimization") == "ORT_ENABLE_ALL" for c in candidates)
    assert any(c.config.get("intra_op_threads") == 8 for c in candidates)
    assert any(c.config.get("precision") == "int8" for c in candidates)


def test_cpu_strategy_skips_without_cpu_provider():
    context = OptimizationContext(
        execution_providers=("CUDAExecutionProvider",),
        logical_cpu_count=8,
    )
    candidates, skipped = default_registry().generate_candidates(
        context, strategy_names=["cpu_thread_tuning"]
    )
    assert candidates == []
    assert "cpu_thread_tuning" in skipped


def test_compilation_requires_adapter():
    context = OptimizationContext()
    candidates, skipped = default_registry().generate_candidates(
        context, strategy_names=["compilation"]
    )
    assert candidates == []
    assert "compilation" in skipped


def test_batch_size_respects_max_batch():
    context = OptimizationContext(metadata={"max_batch_size": 4})
    candidates, _ = default_registry().generate_candidates(
        context, strategy_names=["batch_size"]
    )
    assert {c.config["batch_size"] for c in candidates} == {1, 2, 4}
