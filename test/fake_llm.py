"""A fake generator with the same surface as OnnxTextGenerator, for tests without a model.

Cost model (so the comparison has something to find): uncached decoding is slow,
cached decoding is faster, and a batch costs about one request's time plus a small
per-request overhead. Tokens depend only on the prompt, so every mode produces the
same output unless ``corrupt_batches`` is set.
"""

import time
import zlib
from types import SimpleNamespace


class FakeGenerator:
    def __init__(self, uncached_s=0.01, cached_s=0.002, corrupt_batches=False,
                 fail_uncached=False):
        self.uncached_s = uncached_s
        self.cached_s = cached_s
        self.corrupt_batches = corrupt_batches
        self.fail_uncached = fail_uncached
        self.tokenizer = SimpleNamespace(eos_token_id=None)
        self.last_batch_metrics = ()
        self.batch_calls = []

    def count_prompt_tokens(self, prompt):
        return len(prompt.split())

    def _result(self, request, corrupt=False):
        n = request.config.max_new_tokens
        seed = zlib.crc32(request.prompt.encode())
        ids = tuple((seed + i) % 1000 for i in range(n))
        if corrupt and ids:
            ids = ids[:-1] + ((ids[-1] + 1) % 1000,)
        return SimpleNamespace(
            request_id=request.request_id,
            generated_token_ids=ids,
            generated_token_count=n,
            prompt_token_count=self.count_prompt_tokens(request.prompt),
            time_to_first_token_ms=2.0,
            decode_latency_ms=5.0,
            inter_token_latency_ms=1.0,
            tokens_per_second=100.0,
        )

    def generate(self, request):
        if self.fail_uncached and not request.config.use_cache:
            raise RuntimeError("uncached mode not supported by this fake")
        per_token = self.cached_s if request.config.use_cache else self.uncached_s
        time.sleep(per_token * request.config.max_new_tokens)
        return self._result(request)

    def generate_batch(self, requests, batch_config=None):
        self.batch_calls.append([r.request_id for r in requests])
        time.sleep(self.cached_s * max(r.config.max_new_tokens for r in requests)
                   + 0.0005 * len(requests))
        lengths = [self.count_prompt_tokens(r.prompt) for r in requests]
        padded = max(lengths) * len(lengths)
        self.last_batch_metrics = (SimpleNamespace(
            effective_batch_size=len(requests),
            padding_overhead_ratio=(padded - sum(lengths)) / sum(lengths),
            batch_formation_time_ms=0.1,
        ),)
        return [self._result(r, corrupt=self.corrupt_batches) for r in requests]