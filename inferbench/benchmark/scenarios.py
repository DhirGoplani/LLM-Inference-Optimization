"""Benchmark workloads for the LLM comparison and scheduling tests.

Every scenario is deterministic for a given seed: the same prompts, priorities
and request ids come out every time, so runs can be compared and repeated.
Prompt length is controlled in words (about 1.3 tokens per word for the Qwen
tokenizer); the CLI prints the real token counts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any

from models.generation import GenerationConfig, GenerationRequest

_WORDS = (
    "memory cache token batch queue latency model layer tensor kernel thread "
    "stream decode prefill attention vector matrix runtime provider session graph "
    "buffer window sequence prompt output cost speed budget schedule request server "
    "network storage compute hardware throughput pipeline gradient weight bias shape "
    "length bucket padding mask position index logits sample greedy search engine "
    "report metric trace profile cluster worker signal channel packet device driver"
).split()

SHORT_WORDS = 8
MEDIUM_WORDS = 40
LONG_WORDS = 220

SCENARIO_NAMES = (
    "single",
    "concurrent",
    "short",
    "long",
    "mixed_lengths",
    "mixed_priorities",
    "overload",
)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    requests: tuple[GenerationRequest, ...]
    concurrency: int
    # SchedulerConfig fields this scenario forces (for example a tiny queue).
    scheduler_overrides: dict[str, Any] = field(default_factory=dict)
    expect_rejections: bool = False

    @property
    def max_new_tokens(self) -> int:
        return max(r.config.max_new_tokens for r in self.requests)


def build_prompt(approx_words: int, rng: random.Random) -> str:
    words = [rng.choice(_WORDS) for _ in range(approx_words)]
    return "Summarize these notes in one sentence: " + " ".join(words)


def with_cache(requests, use_cache: bool) -> list[GenerationRequest]:
    """Copies of the requests with KV-cache reuse switched on or off."""
    return [
        replace(request, config=replace(request.config, use_cache=use_cache))
        for request in requests
    ]


def build_scenario(
    name: str,
    *,
    max_new_tokens: int = 16,
    num_requests: int | None = None,
    seed: int = 0,
) -> Scenario:
    """Build one of ``SCENARIO_NAMES``. ``num_requests`` overrides the default count."""
    if name not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {name!r}; choose from {', '.join(SCENARIO_NAMES)}")
    rng = random.Random(seed)
    config = GenerationConfig(max_new_tokens=max_new_tokens, seed=seed)

    def make(index: int, words: int, priority: int = 0) -> GenerationRequest:
        return GenerationRequest(
            build_prompt(words, rng),
            config,
            request_id=f"{name}-{index:03d}",
            priority=priority,
        )

    if name == "single":
        count = num_requests or 1
        return Scenario(
            name,
            "One request at a time (no concurrency).",
            tuple(make(i, MEDIUM_WORDS) for i in range(count)),
            concurrency=1,
        )
    if name == "concurrent":
        count = num_requests or 8
        return Scenario(
            name,
            "Several medium prompts at once.",
            tuple(make(i, MEDIUM_WORDS) for i in range(count)),
            concurrency=min(4, count),
        )
    if name == "short":
        count = num_requests or 8
        return Scenario(
            name,
            "Many short prompts.",
            tuple(make(i, SHORT_WORDS) for i in range(count)),
            concurrency=min(4, count),
        )
    if name == "long":
        count = num_requests or 4
        return Scenario(
            name,
            "A few long prompts (prefill dominated).",
            tuple(make(i, LONG_WORDS) for i in range(count)),
            concurrency=min(2, count),
        )
    if name == "mixed_lengths":
        count = num_requests or 8
        sizes = [SHORT_WORDS, MEDIUM_WORDS, LONG_WORDS]
        weights = [4, 3, 1]
        chosen = rng.choices(sizes, weights=weights, k=count)
        return Scenario(
            name,
            "Short, medium and long prompts mixed together.",
            tuple(make(i, words) for i, words in enumerate(chosen)),
            concurrency=min(4, count),
        )
    if name == "mixed_priorities":
        count = num_requests or 9
        high = max(1, count // 3)
        requests = [
            make(i, MEDIUM_WORDS, priority=5 if i >= count - high else 0)
            for i in range(count)
        ]
        return Scenario(
            name,
            "A burst where the last third of the requests are high priority.",
            tuple(requests),
            concurrency=count,
        )
    count = num_requests or 24  # overload
    return Scenario(
        name,
        "A burst larger than the queue: extra requests must be rejected, not hang.",
        tuple(make(i, SHORT_WORDS) for i in range(count)),
        concurrency=count,
        scheduler_overrides={"max_queue_length": max(2, count // 4)},
        expect_rejections=True,
    )