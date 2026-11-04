
from models.loader import load_model
from models.optimization_adapter import model_spec_to_optimization_context
from inferbench.optimizer.registry import get_tuning_registry


def test_model_ingestion_to_optimizer_candidates():
    spec = load_model("models/mobilenet_v2.onnx")
    context = model_spec_to_optimization_context(spec)

    registry = get_tuning_registry()
    candidates, skipped = registry.generate_candidates(context)

    candidate_strategies = {
        candidate.strategy for candidate in candidates
    }

    assert "graph_optimization" in registry.names()
    assert "cpu_thread_tuning" in registry.names()

    assert "graph_optimization" in candidate_strategies
    assert "cpu_thread_tuning" in candidate_strategies

    assert len(candidates) > 0

    graph_candidates = [
        candidate for candidate in candidates
        if candidate.strategy == "graph_optimization"
    ]

    assert len(graph_candidates) == 4

    for candidate in candidates:
        assert candidate.config

    print("Model:", spec.model_path)
    print("Providers:", context.execution_providers)
    print("Registered strategies:", registry.names())
    print("Generated candidates:", len(candidates))
    print("Skipped strategies:", skipped)

    for candidate in candidates:
        print(candidate.strategy, candidate.config)
