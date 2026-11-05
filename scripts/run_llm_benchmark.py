#!/usr/bin/env python3
r"""Compare baseline, KV cache, batching, bucketing and scheduling on one model.

Examples (PowerShell, from the repository root):

    python scripts\run_llm_benchmark.py --model-dir models\qwen2.5_0.5b
    python scripts\run_llm_benchmark.py --scenario mixed_lengths --max-new-tokens 16 --scheduling-scenarios
    python scripts\run_llm_benchmark.py --only kv_cache,batching,bucketing --num-requests 6

Writes results\llm_report_<model>_<time>.json and a matching .md file.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inferbench.benchmark.report import (  # noqa: E402
    CONFIGURATIONS,
    render_markdown,
    run_comparison,
    run_scheduler_scenario,
    save_report_files,
)
from inferbench.benchmark.scenarios import SCENARIO_NAMES, build_scenario  # noqa: E402
from models.batching import BatchConfig  # noqa: E402


def parse_buckets(text: str) -> tuple[int, ...] | None:
    if text.strip().lower() in ("none", ""):
        return None
    return tuple(int(part) for part in text.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", default="models/qwen2.5_0.5b")
    parser.add_argument("--scenario", default="mixed_lengths", choices=SCENARIO_NAMES)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated subset of: " + ", ".join(c.key for c in CONFIGURATIONS),
    )
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--window-ms", type=float, default=5.0)
    parser.add_argument("--buckets", default="128,512,2048,4096", help="'none' disables bucketing")
    parser.add_argument("--min-exact-match", type=float, default=0.9)
    parser.add_argument("--min-improvement", type=float, default=1.10)
    parser.add_argument(
        "--scheduling-scenarios",
        action="store_true",
        help="also run the mixed-priority and queue-overload scenarios on the scheduler",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-hash", action="store_true", help="skip hashing the model files")
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def default_generator(model_dir: str) -> Any:
    from models.generation import OnnxTextGenerator

    return OnnxTextGenerator(model_dir)


def main(
    argv: list[str] | None = None,
    generator_factory: Callable[[str], Any] = default_generator,
) -> int:
    args = build_parser().parse_args(argv)
    only = [key.strip() for key in args.only.split(",")] if args.only else None
    known = {c.key for c in CONFIGURATIONS}
    if only and not set(only) <= known:
        print(f"Unknown configuration(s): {sorted(set(only) - known)}. Choose from {sorted(known)}")
        return 2

    print(f"Loading {args.model_dir} ...")
    generator = generator_factory(args.model_dir)
    batch_config = BatchConfig(
        max_batch_size=args.max_batch_size,
        batching_window_ms=args.window_ms,
        prompt_length_buckets=parse_buckets(args.buckets),
    )
    scenario = build_scenario(
        args.scenario,
        max_new_tokens=args.max_new_tokens,
        num_requests=args.num_requests,
        seed=args.seed,
    )
    token_counts = [generator.count_prompt_tokens(r.prompt) for r in scenario.requests]
    print(
        f"Scenario {scenario.name}: {len(scenario.requests)} requests, "
        f"prompt tokens {token_counts}, concurrency {scenario.concurrency}"
    )

    report = asyncio.run(run_comparison(
        generator,
        scenario,
        batch_config,
        only=only,
        model_path=args.model_dir,
        seed=args.seed,
        min_exact_match_rate=args.min_exact_match,
        min_improvement=args.min_improvement,
        warmup=not args.no_warmup,
        hash_model=not args.no_hash,
        progress=print,
    ))

    if args.scheduling_scenarios:
        report["scheduling_scenarios"] = []
        for name in ("mixed_priorities", "overload"):
            print(f"running scheduling scenario: {name}")
            extra = build_scenario(name, max_new_tokens=args.max_new_tokens, seed=args.seed)
            report["scheduling_scenarios"].append(
                asyncio.run(run_scheduler_scenario(generator, extra, batch_config))
            )

    json_path, md_path = save_report_files(report, args.results_dir)
    print()
    print(render_markdown(report))
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())