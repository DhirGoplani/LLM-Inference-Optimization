"""Comparison report, environment capture and export. Uses a fake generator."""

import asyncio
import json

import pytest

from inferbench.benchmark.environment import collect_environment, file_sha256, model_fingerprint
from inferbench.benchmark.export import llm_report_to_json, load_llm_report
from inferbench.benchmark.report import (
    CONFIGURATIONS,
    recommend,
    render_markdown,
    run_comparison,
    save_report_files,
)
from inferbench.benchmark.scenarios import SCENARIO_NAMES, build_scenario, with_cache
from models.batching import BatchConfig
from tests.fake_llm import FakeGenerator


def compare(generator, **kwargs):
    scenario = build_scenario("concurrent", max_new_tokens=4, num_requests=6)
    return asyncio.run(run_comparison(
        generator, scenario, BatchConfig(max_batch_size=4, batching_window_ms=5.0),
        hash_model=False, warmup=False, **kwargs,
    ))


def test_all_five_configurations_run_and_are_recommended_on_evidence():
    report = compare(FakeGenerator())
    keys = [c["key"] for c in report["configurations"]]
    assert keys == [c.key for c in CONFIGURATIONS]
    assert all(c["success"] for c in report["configurations"])
    assert all(c["quality"]["passed"] for c in report["configurations"])
    rec = report["recommendation"]
    assert rec["configuration"] in {"kv_cache", "batching", "bucketing", "scheduling"}
    assert rec["tokens_per_second_x"] > 1.1
    kv = next(c for c in report["configurations"] if c["key"] == "kv_cache")
    assert kv["vs_baseline"]["tokens_per_second_x"] > 1.5
    scheduled = next(c for c in report["configurations"] if c["key"] == "scheduling")
    assert scheduled["metrics"]["batching"]["mean_effective_batch_size"] > 1
    assert scheduled["metrics"]["counts"]["completed"] == 6


def test_report_is_strict_json_and_has_environment_metadata():
    report = compare(FakeGenerator())
    text = llm_report_to_json(report)  # allow_nan=False: raises on NaN/Inf
    data = json.loads(text)
    env = data["environment"]
    for key in ("timestamp_utc", "seed", "python", "platform", "hardware", "runtime", "git",
                "benchmark_config", "provider"):
        assert key in env
    assert env["benchmark_config"]["scenario"] == "concurrent"
    assert env["benchmark_config"]["batch_config"]["max_batch_size"] == 4
    assert env["runtime"]["onnxruntime"]


def test_a_batching_quality_regression_blocks_the_recommendation():
    report = compare(FakeGenerator(corrupt_batches=True))
    by_key = {c["key"]: c for c in report["configurations"]}
    assert by_key["kv_cache"]["quality"]["passed"]
    for key in ("batching", "bucketing", "scheduling"):
        assert not by_key[key]["quality"]["passed"]
    assert report["recommendation"]["configuration"] == "kv_cache"
    assert any("failed quality" in item for item in report["recommendation"]["rejected"])


def test_failed_baseline_means_no_recommendation():
    report = compare(FakeGenerator(fail_uncached=True))
    base = report["configurations"][0]
    assert not base["success"] and base["error_count"] == 6
    assert report["recommendation"]["configuration"] is None
    assert "baseline" in report["recommendation"]["reason"]


def test_recommend_requires_a_measured_benefit():
    def config(key, tps_x, mem_x, passed=True, success=True):
        return {
            "key": key, "title": key, "success": success,
            "quality": {"passed": passed},
            "vs_baseline": None if key == "baseline" else {
                "tokens_per_second_x": tps_x, "peak_memory_ratio": mem_x},
        }

    base = config("baseline", None, None)
    assert recommend([base, config("kv_cache", 1.02, 1.0)])["configuration"] is None
    assert recommend([base, config("kv_cache", 1.5, 1.0)])["configuration"] == "kv_cache"
    assert recommend([base, config("kv_cache", 1.0, 0.5)])["configuration"] == "kv_cache"  # memory win
    assert recommend([base, config("kv_cache", 3.0, 1.0, passed=False)])["configuration"] is None
    assert recommend([base, config("kv_cache", 3.0, 1.0, success=False)])["configuration"] is None
    best = recommend([base, config("kv_cache", 1.5, 1.0), config("batching", 2.5, 1.0)])
    assert best["configuration"] == "batching"


def test_markdown_and_saved_files(tmp_path):
    report = compare(FakeGenerator(), only=["baseline", "kv_cache"])
    assert [c["key"] for c in report["configurations"]] == ["baseline", "kv_cache"]
    markdown = render_markdown(report)
    for heading in ("## Results", "## Improvement over the baseline", "## Recommendation",
                    "## Environment"):
        assert heading in markdown
    json_path, md_path = save_report_files(report, tmp_path)
    assert json_path.suffix == ".json" and md_path.suffix == ".md"
    assert load_llm_report(json_path)["scenario"]["name"] == "concurrent"
    assert "LLM inference comparison report" in md_path.read_text(encoding="utf-8")


def test_unknown_configuration_selection_is_an_error():
    scenario = build_scenario("short", max_new_tokens=2, num_requests=2)
    with pytest.raises(ValueError):
        asyncio.run(run_comparison(FakeGenerator(), scenario, only=["nope"], warmup=False))


def test_identical_runs_have_identical_output_digests():
    first = compare(FakeGenerator(), only=["kv_cache"])
    second = compare(FakeGenerator(), only=["kv_cache"])
    assert first["configurations"][0]["output_digest"] == second["configurations"][0]["output_digest"]


def test_scenarios_are_deterministic_and_complete():
    for name in SCENARIO_NAMES:
        a, b = build_scenario(name, seed=3), build_scenario(name, seed=3)
        assert [r.prompt for r in a.requests] == [r.prompt for r in b.requests]
        assert len({r.request_id for r in a.requests}) == len(a.requests)
        assert a.concurrency >= 1
    assert build_scenario("overload").expect_rejections
    priorities = [r.priority for r in build_scenario("mixed_priorities").requests]
    assert max(priorities) > min(priorities)
    lengths = {len(r.prompt.split()) for r in build_scenario("mixed_lengths", seed=1,
                                                              num_requests=30).requests}
    assert len(lengths) == 3
    with pytest.raises(ValueError):
        build_scenario("nope")


def test_with_cache_only_changes_the_cache_flag():
    scenario = build_scenario("short", num_requests=2)
    off = with_cache(scenario.requests, False)
    assert all(not r.config.use_cache for r in off)
    assert [r.prompt for r in off] == [r.prompt for r in scenario.requests]


def test_environment_fingerprints_model_files(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"graph-bytes")
    (tmp_path / "model.onnx.data").write_bytes(b"weights")
    info = model_fingerprint(tmp_path)
    assert [f["file"] for f in info["files"]] == ["model.onnx", "model.onnx.data"]
    assert info["files"][0]["sha256"] == file_sha256(tmp_path / "model.onnx")
    assert info["sha256"] == model_fingerprint(tmp_path)["sha256"]
    (tmp_path / "model.onnx").write_bytes(b"changed")
    assert model_fingerprint(tmp_path)["sha256"] != info["sha256"]
    env = collect_environment(tmp_path, seed=7, hash_model=False)
    assert env["seed"] == 7 and env["model"]["sha256"] is None


def _load_benchmark_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_llm_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_llm_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_script_end_to_end_with_a_fake_generator(tmp_path, capsys):
    script = _load_benchmark_script()
    code = script.main(
        [
            "--model-dir", str(tmp_path),
            "--scenario", "concurrent",
            "--num-requests", "4",
            "--max-new-tokens", "3",
            "--results-dir", str(tmp_path / "out"),
            "--scheduling-scenarios",
            "--no-hash",
            "--no-warmup",
        ],
        generator_factory=lambda model_dir: FakeGenerator(),
    )
    assert code == 0
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert len(files) == 2 and files[0].endswith(".json") and files[1].endswith(".md")
    report = load_llm_report(tmp_path / "out" / files[0])
    assert [s["scenario"] for s in report["scheduling_scenarios"]] == ["mixed_priorities", "overload"]
    assert report["scheduling_scenarios"][1]["rejected"] > 0
    assert "## Scheduling scenarios" in (tmp_path / "out" / files[1]).read_text(encoding="utf-8")


def test_benchmark_script_rejects_unknown_configuration(tmp_path):
    script = _load_benchmark_script()
    assert script.main(["--only", "bogus"], generator_factory=lambda d: FakeGenerator()) == 2
    assert script.parse_buckets("none") is None
    assert script.parse_buckets("64,256") == (64, 256)