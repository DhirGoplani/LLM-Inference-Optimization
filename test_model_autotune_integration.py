
from inferbench.common.benchmarktypes import BenchmarkConfig
from inferbench.optimizer.adapter import run_autotune
from models.loader import load_model
from models.optimization_adapter import model_spec_to_optimization_context
from inferbench.optimizer.context import OptimizationContext


def test_model_autotune_workflow():
    config = BenchmarkConfig(
        model_path="models/mobilenet_v2.onnx",
        execution_provider="CPUExecutionProvider",
        graph_optimization_level="all",
        num_threads=1,
        warmup_runs=1,
        timed_runs=3,
        batch_size=1,
    )

    result = run_autotune(config)

    print("Model:", result["model_path"])
    print("Baseline success:", result["baseline"]["success"])
    print("Candidates evaluated:", result["candidates_evaluated"])
    print("Candidates valid:", result["candidates_valid"])
    print("Best candidate:", result["best"])
    print("Speedup:", result["speedup"])

    assert result["model_path"] == config.model_path
    assert result["error_message"] if "error_message" in result else True
    assert result["candidates_evaluated"] > 0


def test_autotune_accepts_model_optimization_context(monkeypatch):
    from inferbench.optimizer import adapter

    spec = load_model("models/mobilenet_v2.onnx")
    context = model_spec_to_optimization_context(spec)

    captured = {}

    class FakeRegistry:
        def generate_candidates(self, supplied_context, strategy_names=None):
            captured["context"] = supplied_context
            return [], {}

    monkeypatch.setattr(
        adapter,
        "get_tuning_registry",
        lambda: FakeRegistry(),
    )

    monkeypatch.setattr(
        adapter,
        "make_baseline",
        lambda config: adapter.Baseline(
            config=config,
            result=adapter.failed_result(
                config,
                "Baseline intentionally skipped in this unit test",
            ),
        ),
    )

    config = BenchmarkConfig(
        model_path=spec.model_path,
        warmup_runs=0,
        timed_runs=1,
    )

    adapter.run_autotune(config, context=context)

    assert captured["context"] is context
