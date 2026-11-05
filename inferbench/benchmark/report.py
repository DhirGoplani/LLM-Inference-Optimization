"""Run the five demonstration configurations and build the comparison report.

Configurations (from division.md), each run on the same workload:

    baseline    ONNX Runtime, single request, no KV cache
    kv_cache    baseline + KV-cache reuse
    batching    KV cache + dynamic batching (no prompt buckets)
    bucketing   KV cache + dynamic batching + prompt-length buckets
    scheduling  all optimizations + the request scheduler

All of them are driven the same way (closed loop: ``scenario.concurrency``
clients, each sending its next request when the previous one returns) and
measured with the same ``ServingMetrics`` definitions, so the numbers compare.
Quality is judged against the baseline's tokens. A configuration is only
recommended if it passes the quality checks AND shows a measured benefit.

The generator is duck-typed: anything with ``generate``, ``generate_batch``,
``count_prompt_tokens`` and ``last_batch_metrics`` works (the tests use a fake).
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from inferbench.benchmark.correctness import evaluate_generation_quality
from inferbench.benchmark.environment import collect_environment
from inferbench.benchmark.export import save_llm_report
from inferbench.benchmark.metrics import PeakMemoryMonitor
from inferbench.benchmark.scenarios import Scenario, with_cache
from inferbench.scheduler import (
    RequestRejectedError,
    RequestScheduler,
    RequestState,
    SchedulerConfig,
    ServingMetrics,
)
from models.batching import BatchConfig, DynamicBatcher


@dataclass(frozen=True)
class ConfigSpec:
    key: str
    title: str
    use_cache: bool
    mode: str  # "sequential" | "batched" | "scheduled"
    bucketed: bool


CONFIGURATIONS: tuple[ConfigSpec, ...] = (
    ConfigSpec("baseline", "ONNX Runtime baseline, single request", False, "sequential", False),
    ConfigSpec("kv_cache", "Baseline + KV-cache reuse", True, "sequential", False),
    ConfigSpec("batching", "KV cache + dynamic batching", True, "batched", False),
    ConfigSpec("bucketing", "KV cache + dynamic batching + prompt bucketing", True, "batched", True),
    ConfigSpec("scheduling", "All optimizations + request scheduling", True, "scheduled", True),
)


@dataclass
class ConfigRun:
    spec: ConfigSpec
    success: bool
    metrics: dict[str, Any]
    wall_time_s: float
    peak_memory_mb: float
    memory_delta_mb: float
    errors: dict[str, str] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)  # request_id -> result (not exported)
    quality: dict[str, Any] | None = None
    output_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.spec.key,
            "title": self.spec.title,
            "use_cache": self.spec.use_cache,
            "mode": self.spec.mode,
            "bucketed": self.spec.bucketed,
            "success": self.success,
            "error_count": len(self.errors),
            "errors": dict(list(self.errors.items())[:5]),
            "wall_time_s": self.wall_time_s,
            "peak_memory_mb": self.peak_memory_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "metrics": self.metrics,
            "quality": self.quality,
            "output_digest": self.output_digest,
        }


def _digest(results: dict[str, Any]) -> str | None:
    if not results:
        return None
    text = "|".join(
        f"{rid}:{','.join(map(str, results[rid].generated_token_ids))}" for rid in sorted(results)
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _eos_token_id(generator: Any) -> int | None:
    eos = getattr(getattr(generator, "tokenizer", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(getattr(generator, "model_metadata", None), "eos_token_id", None)
    return eos


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


async def run_configuration(
    generator: Any,
    spec: ConfigSpec,
    scenario: Scenario,
    batch_config: BatchConfig,
    scheduler_overrides: dict[str, Any] | None = None,
) -> ConfigRun:
    """Run one configuration on one scenario. Never raises: failures are recorded."""
    requests = with_cache(scenario.requests, spec.use_cache)
    metrics = ServingMetrics()
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    config_for_mode = batch_config if spec.bucketed else replace(batch_config, prompt_length_buckets=None)

    def count(request: Any) -> int:
        return generator.count_prompt_tokens(request.prompt)

    batcher: DynamicBatcher | None = None
    scheduler: RequestScheduler | None = None
    lock = asyncio.Lock()

    if spec.mode == "batched":
        def batch_callback(pending: Any) -> Any:
            pending = list(pending)
            for request in pending:
                metrics.on_dispatch(request.request_id)
            out = generator.generate_batch(pending, config_for_mode)
            recorded = False
            for item in getattr(generator, "last_batch_metrics", ()) or ():
                metrics.on_batch(
                    item.effective_batch_size,
                    getattr(item, "padding_overhead_ratio", 0.0),
                    getattr(item, "batch_formation_time_ms", 0.0),
                )
                recorded = True
            if not recorded:
                metrics.on_batch(len(pending))
            return out

        batcher = DynamicBatcher(batch_callback, token_counter=count, config=config_for_mode)
    elif spec.mode == "scheduled":
        scheduler_config = SchedulerConfig(
            max_queue_length=max(64, len(requests)),
            max_batch_size=config_for_mode.max_batch_size,
            max_batch_tokens=config_for_mode.max_total_batch_tokens,
            batching_window_ms=config_for_mode.batching_window_ms,
            **(scheduler_overrides or {}),
        )
        scheduler = RequestScheduler(
            lambda reqs: generator.generate_batch(reqs, config_for_mode),
            token_counter=count,
            config=scheduler_config,
            metrics=metrics,
            batch_metrics_source=lambda: getattr(generator, "last_batch_metrics", ()),
        )

    async def serve(request: Any) -> Any:
        if scheduler is not None:
            return await scheduler.submit(request)
        rid = request.request_id
        metrics.on_enqueue(rid)
        try:
            if batcher is not None:
                result = await batcher.submit(request)
            else:
                async with lock:
                    metrics.on_dispatch(rid)
                    result = await asyncio.to_thread(generator.generate, request)
        except Exception:
            metrics.on_finish(rid, RequestState.FAILED)
            raise
        metrics.on_finish(rid, RequestState.COMPLETED, result)
        return result

    pending = deque(requests)

    async def client() -> None:
        while pending:
            request = pending.popleft()
            try:
                results[request.request_id] = await serve(request)
            except Exception as exc:
                errors[request.request_id] = _error(exc)

    workers = max(1, min(scenario.concurrency, len(requests)))
    started = time.perf_counter()
    with PeakMemoryMonitor() as memory:
        try:
            await asyncio.gather(*(client() for _ in range(workers)))
        finally:
            if batcher is not None:
                await batcher.close()
            if scheduler is not None:
                await scheduler.close()
    wall = time.perf_counter() - started

    return ConfigRun(
        spec=spec,
        success=not errors and len(results) == len(requests),
        metrics=metrics.snapshot(),
        wall_time_s=wall,
        peak_memory_mb=memory.peak_mb,
        memory_delta_mb=memory.delta_mb,
        errors=errors,
        results=results,
        output_digest=_digest(results),
    )


def _ratio(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return new / old


def _improvement(run: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    m, b = run["metrics"], base["metrics"]

    def stat(source: dict[str, Any], key: str, field_name: str) -> float | None:
        value = source.get(key)
        return value.get(field_name) if value else None

    return {
        "tokens_per_second_x": _ratio(m["tokens_per_second"], b["tokens_per_second"]),
        "requests_per_second_x": _ratio(m["requests_per_second"], b["requests_per_second"]),
        "end_to_end_p50_ratio": _ratio(
            stat(m, "end_to_end_ms", "p50"), stat(b, "end_to_end_ms", "p50")
        ),
        "end_to_end_p95_ratio": _ratio(
            stat(m, "end_to_end_ms", "p95"), stat(b, "end_to_end_ms", "p95")
        ),
        "ttft_incl_queue_p50_ratio": _ratio(
            stat(m, "ttft_incl_queue_ms", "p50"), stat(b, "ttft_incl_queue_ms", "p50")
        ),
        "peak_memory_ratio": _ratio(run["peak_memory_mb"], base["peak_memory_mb"]),
    }


def recommend(
    configs: list[dict[str, Any]], min_improvement: float = 1.10
) -> dict[str, Any]:
    """Pick the best configuration that passes quality AND beats the baseline.

    A benefit means tokens/s at least ``min_improvement`` times the baseline, or
    peak memory at most 1/min_improvement of the baseline.
    """
    baseline = next((c for c in configs if c["key"] == "baseline"), None)
    if baseline is None or not baseline["success"]:
        return {"configuration": None, "reason": "The baseline did not run successfully, so nothing can be compared against it."}
    best: tuple[float, dict[str, Any]] | None = None
    rejected: list[str] = []
    for config in configs:
        if config["key"] == "baseline":
            continue
        quality = config.get("quality") or {}
        if not config["success"]:
            rejected.append(f"{config['key']}: run failed")
            continue
        if not quality.get("passed"):
            rejected.append(f"{config['key']}: failed quality checks")
            continue
        gain = config["vs_baseline"]
        tps = gain["tokens_per_second_x"]
        mem = gain["peak_memory_ratio"]
        beneficial = (tps is not None and tps >= min_improvement) or (
            mem is not None and mem <= 1.0 / min_improvement
        )
        if not beneficial:
            rejected.append(f"{config['key']}: no measured benefit over the baseline")
            continue
        score = tps if tps is not None else 0.0
        if best is None or score > best[0]:
            best = (score, config)
    if best is None:
        return {
            "configuration": None,
            "reason": "No configuration passed the quality checks and showed a measured benefit.",
            "rejected": rejected,
        }
    config = best[1]
    return {
        "configuration": config["key"],
        "title": config["title"],
        "tokens_per_second_x": config["vs_baseline"]["tokens_per_second_x"],
        "peak_memory_ratio": config["vs_baseline"]["peak_memory_ratio"],
        "reason": "Best tokens/s among the configurations that pass the quality checks.",
        "rejected": rejected,
    }


async def run_comparison(
    generator: Any,
    scenario: Scenario,
    batch_config: BatchConfig | None = None,
    only: list[str] | None = None,
    model_path: str | Path | None = None,
    seed: int = 0,
    min_exact_match_rate: float = 0.9,
    min_improvement: float = 1.10,
    warmup: bool = True,
    hash_model: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the configurations on one scenario and return the full report dict."""
    say = progress or (lambda message: None)
    batch_config = batch_config or BatchConfig()
    selected = [c for c in CONFIGURATIONS if only is None or c.key in only]
    if not selected:
        raise ValueError("no configurations selected")

    if warmup:
        say("warming up the model")
        try:
            first = with_cache(scenario.requests[:1], True)[0]
            warm = replace(first, config=replace(first.config, max_new_tokens=2), request_id="warmup")
            await asyncio.to_thread(generator.generate, warm)
        except Exception as exc:  # a failing warmup is reported by the real runs
            say(f"warmup failed: {_error(exc)}")

    runs: list[ConfigRun] = []
    for spec in selected:
        say(f"running {spec.key}: {spec.title}")
        run = await run_configuration(generator, spec, scenario, batch_config)
        say(
            f"  {spec.key}: {'ok' if run.success else 'FAILED'} in {run.wall_time_s:.1f}s, "
            f"{run.metrics['tokens_per_second']:.1f} tok/s"
        )
        runs.append(run)

    eos = _eos_token_id(generator)
    reference = next((r for r in runs if r.spec.key == "baseline" and r.results), None)
    if reference is None:
        reference = next((r for r in runs if r.results), None)
    for run in runs:
        if not run.results:
            continue
        run.quality = evaluate_generation_quality(
            None if run is reference else list(reference.results.values()) if reference else None,
            list(run.results.values()),
            eos_token_id=eos,
            max_new_tokens=scenario.max_new_tokens,
            min_exact_match_rate=min_exact_match_rate,
        )

    configs = [run.to_dict() for run in runs]
    base = next((c for c in configs if c["key"] == "baseline"), None)
    for config in configs:
        config["vs_baseline"] = (
            _improvement(config, base) if base and config["key"] != "baseline" else None
        )

    benchmark_config = {
        "scenario": scenario.name,
        "requests": len(scenario.requests),
        "concurrency": scenario.concurrency,
        "max_new_tokens": scenario.max_new_tokens,
        "batch_config": asdict(batch_config),
        "min_exact_match_rate": min_exact_match_rate,
        "min_improvement": min_improvement,
        "configurations": [c.key for c in selected],
    }
    model_info = None
    environment = collect_environment(
        model_path, seed=seed, benchmark_config=benchmark_config, hash_model=hash_model
    )
    if model_path is not None:
        model_info = environment.get("model")
    return {
        "schema_version": 1,
        "model": model_info,
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "requests": len(scenario.requests),
            "concurrency": scenario.concurrency,
            "prompt_tokens": [
                generator.count_prompt_tokens(r.prompt) for r in scenario.requests
            ],
        },
        "reference_configuration": reference.spec.key if reference else None,
        "configurations": configs,
        "recommendation": recommend(configs, min_improvement),
        "environment": environment,
    }


async def run_scheduler_scenario(
    generator: Any,
    scenario: Scenario,
    batch_config: BatchConfig | None = None,
    scheduler_overrides: dict[str, Any] | None = None,
    overall_timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Send a whole scenario to the scheduler as one burst and report what happened.

    Used for the priority and overload scenarios: it records, per request, whether
    it completed, was rejected, timed out, was cancelled or failed, the order in
    which requests finished, and the mean queue wait for each priority level.
    """
    batch_config = batch_config or BatchConfig()
    overrides = {**scenario.scheduler_overrides, **(scheduler_overrides or {})}
    config = SchedulerConfig(
        max_batch_size=batch_config.max_batch_size,
        max_batch_tokens=batch_config.max_total_batch_tokens,
        batching_window_ms=batch_config.batching_window_ms,
        **overrides,
    )
    metrics = ServingMetrics()
    scheduler = RequestScheduler(
        lambda reqs: generator.generate_batch(reqs, batch_config),
        token_counter=lambda r: generator.count_prompt_tokens(r.prompt),
        config=config,
        metrics=metrics,
        batch_metrics_source=lambda: getattr(generator, "last_batch_metrics", ()),
    )
    outcomes: dict[str, str] = {}
    handles = {}
    requests = with_cache(scenario.requests, True)

    async def burst() -> None:
        for request in requests:
            try:
                handles[request.request_id] = scheduler.enqueue(request)
            except RequestRejectedError as exc:
                outcomes[request.request_id] = f"rejected:{exc.reason}"
        for rid, handle in handles.items():
            try:
                await handle.result()
                outcomes[rid] = "completed"
            except Exception as exc:
                outcomes[rid] = handle.state.value if handle.state.terminal else _error(exc)
        await scheduler.close()

    hung = False
    try:
        await asyncio.wait_for(burst(), overall_timeout_s)
    except TimeoutError:
        hung = True
        await scheduler.close(drain=False)

    waits: dict[int, list[float]] = {}
    finish_order: list[tuple[float, str, int]] = []
    for rid, handle in handles.items():
        states = {state: at for state, at in handle.history}
        started = states.get(RequestState.BATCHED)
        if started is not None:
            waits.setdefault(handle.request.priority, []).append(
                (started - handle.enqueued_at) * 1000.0
            )
        done = states.get(RequestState.COMPLETED)
        if done is not None:
            finish_order.append((done, rid, handle.request.priority))
    finish_order.sort()
    counts: dict[str, int] = {}
    for outcome in outcomes.values():
        counts[outcome] = counts.get(outcome, 0) + 1
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "requests": len(requests),
        "scheduler_config": asdict(config),
        "outcome_counts": counts,
        "rejected": sum(v for k, v in counts.items() if k.startswith("rejected")),
        "hung": hung,
        "mean_queue_wait_ms_by_priority": {
            str(p): sum(v) / len(v) for p, v in sorted(waits.items(), reverse=True)
        },
        "completion_order": [{"request_id": r, "priority": p} for _, r, p in finish_order],
        "metrics": scheduler.snapshot(),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _stat(metrics: dict[str, Any], key: str, name: str) -> float | None:
    value = metrics.get(key)
    return value.get(name) if value else None


def render_markdown(report: dict[str, Any]) -> str:
    """A readable Markdown version of the report."""
    scenario = report["scenario"]
    lines = [
        "# LLM inference comparison report",
        "",
        f"- Model: {(report.get('model') or {}).get('name', 'unknown')}",
        f"- Scenario: **{scenario['name']}** - {scenario['description']}",
        f"- Requests: {scenario['requests']}, concurrency: {scenario['concurrency']}",
        f"- Quality reference: {report.get('reference_configuration')}",
        "",
        "## Results",
        "",
        "| Configuration | OK | TTFT p50 (ms) | TTFT p50 incl. queue (ms) | E2E p50 (ms) | E2E p95 (ms) "
        "| Inter-token (ms) | Tokens/s | Req/s | Peak mem (MB) | Queue wait p95 (ms) "
        "| Eff. batch | Padding | Exact match | Quality |",
        "|" + "---|" * 15,
    ]
    for config in report["configurations"]:
        m = config["metrics"]
        quality = config.get("quality") or {}
        padding = m["batching"]["mean_padding_overhead_ratio"]
        lines.append(
            "| {title} | {ok} | {ttft} | {ttftq} | {p50} | {p95} | {itl} | {tps} | {rps} | {mem} "
            "| {qw} | {batch} | {pad} | {exact} | {q} |".format(
                title=config["title"],
                ok=_fmt(config["success"]),
                ttft=_fmt(_stat(m, "ttft_ms", "p50")),
                ttftq=_fmt(_stat(m, "ttft_incl_queue_ms", "p50")),
                p50=_fmt(_stat(m, "end_to_end_ms", "p50")),
                p95=_fmt(_stat(m, "end_to_end_ms", "p95")),
                itl=_fmt(_stat(m, "inter_token_latency_ms", "mean"), 2),
                tps=_fmt(m["tokens_per_second"]),
                rps=_fmt(m["requests_per_second"], 2),
                mem=_fmt(config["peak_memory_mb"], 0),
                qw=_fmt(_stat(m, "queue_wait_ms", "p95")),
                batch=_fmt(m["batching"]["mean_effective_batch_size"], 2),
                pad=_fmt(None if padding is None else padding * 100.0, 0) + ("%" if padding is not None else ""),
                exact=_fmt(quality.get("exact_match_rate"), 2),
                q="pass" if quality.get("passed") else ("FAIL" if quality else "-"),
            )
        )
    lines += ["", "## Improvement over the baseline", ""]
    lines += ["| Configuration | Tokens/s | Req/s | E2E p95 | Peak memory |", "|---|---|---|---|---|"]
    for config in report["configurations"]:
        gain = config.get("vs_baseline")
        if not gain:
            continue
        lines.append(
            f"| {config['title']} | {_fmt(gain['tokens_per_second_x'], 2)}x "
            f"| {_fmt(gain['requests_per_second_x'], 2)}x "
            f"| {_fmt(gain['end_to_end_p95_ratio'], 2)}x of baseline "
            f"| {_fmt(gain['peak_memory_ratio'], 2)}x of baseline |"
        )
    rec = report["recommendation"]
    lines += ["", "## Recommendation", ""]
    if rec.get("configuration"):
        lines.append(
            f"**{rec['title']}** - {_fmt(rec['tokens_per_second_x'], 2)}x baseline tokens/s "
            f"at {_fmt(rec['peak_memory_ratio'], 2)}x peak memory. {rec['reason']}"
        )
    else:
        lines.append(f"No configuration recommended. {rec['reason']}")
    for item in rec.get("rejected", []):
        lines.append(f"- not recommended - {item}")
    failed = [c for c in report["configurations"] if c["error_count"]]
    if failed:
        lines += ["", "## Errors", ""]
        for config in failed:
            lines.append(f"- {config['key']}: {config['error_count']} failed, e.g. {next(iter(config['errors'].values()))}")
    scheduling = report.get("scheduling_scenarios") or []
    if scheduling:
        lines += ["", "## Scheduling scenarios", ""]
        for item in scheduling:
            lines.append(
                f"- **{item['scenario']}**: {item['requests']} requests -> {item['outcome_counts']}"
                f"; rejected {item['rejected']}; hung: {_fmt(item['hung'])}"
            )
            if item["mean_queue_wait_ms_by_priority"]:
                waits = ", ".join(
                    f"priority {p}: {_fmt(w)} ms" for p, w in item["mean_queue_wait_ms_by_priority"].items()
                )
                lines.append(f"  - mean queue wait - {waits}")
    env = report["environment"]
    lines += [
        "",
        "## Environment",
        "",
        f"- Timestamp (UTC): {env['timestamp_utc']}, seed: {env['seed']}",
        f"- Python {env['python']['version']}, onnxruntime {env['runtime']['onnxruntime']}, provider {env['provider']}",
        f"- Hardware: {env['platform']['processor'] or env['platform']['machine']}, "
        f"{env['hardware']['logical_cpus']} logical CPUs, {env['hardware']['total_ram_gb']} GB RAM",
        f"- Model SHA-256: {(env.get('model') or {}).get('sha256') or 'not computed'}",
        f"- Git commit: {env['git']['commit']} (dirty: {_fmt(env['git']['dirty'])})",
        "",
    ]
    return "\n".join(lines)


def save_report_files(
    report: dict[str, Any], results_dir: str | Path = "results"
) -> tuple[Path, Path]:
    """Write the report as .json and .md next to each other; returns both paths."""
    json_path = save_llm_report(report, results_dir=results_dir)
    md_path = json_path.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path