"""Scheduler tests. They use a fake batch executor, so no model is needed."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from inferbench.scheduler import (
    DuplicateRequestError,
    GenerationTooLongError,
    PromptTooLongError,
    QueueFullError,
    RequestCancelledError,
    RequestRejectedError,
    RequestScheduler,
    RequestState,
    RequestTimeoutError,
    SchedulerClosedError,
    SchedulerConfig,
)
from models.generation import GenerationConfig, GenerationRequest


def make_request(rid, prompt="hello world", priority=0, deadline_s=None, max_new_tokens=4):
    return GenerationRequest(
        prompt,
        GenerationConfig(max_new_tokens=max_new_tokens),
        request_id=rid,
        priority=priority,
        deadline_s=deadline_s,
    )


def make_result(request):
    return SimpleNamespace(
        request_id=request.request_id,
        text=f"out-{request.request_id}",
        generated_token_ids=(1, 2),
        generated_token_count=2,
        prompt_token_count=2,
        time_to_first_token_ms=1.0,
        inter_token_latency_ms=0.5,
    )


class FakeExecutor:
    """Blocking executor that records every batch it receives."""

    def __init__(self, delay=0.0, fail=False):
        self.delay = delay
        self.fail = fail
        self.batches = []

    def __call__(self, requests):
        self.batches.append([r.request_id for r in requests])
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        return [make_result(r) for r in requests]


def run(coro):
    return asyncio.run(coro)


def single(**kwargs):
    base = dict(max_batch_size=1, batching_window_ms=0.0, aging_interval_s=None)
    base.update(kwargs)
    return SchedulerConfig(**base)


def test_fifo_is_the_default_order():
    async def go():
        executor = FakeExecutor(delay=0.02)
        async with RequestScheduler(executor, config=single()) as scheduler:
            handles = [scheduler.enqueue(make_request(f"r{i}")) for i in range(4)]
            await asyncio.gather(*(h.result() for h in handles))
        return executor.batches

    assert run(go()) == [["r0"], ["r1"], ["r2"], ["r3"]]


def test_higher_priority_is_served_first_and_ties_stay_fifo():
    async def go():
        executor = FakeExecutor(delay=0.02)
        async with RequestScheduler(executor, config=single()) as scheduler:
            first = scheduler.enqueue(make_request("running"))
            await asyncio.sleep(0.005)  # let "running" start so the rest queue up
            rest = [
                scheduler.enqueue(make_request("low", priority=0)),
                scheduler.enqueue(make_request("high-a", priority=5)),
                scheduler.enqueue(make_request("mid", priority=2)),
                scheduler.enqueue(make_request("high-b", priority=5)),
            ]
            await asyncio.gather(first.result(), *(h.result() for h in rest))
        return [b[0] for b in executor.batches]

    assert run(go()) == ["running", "high-a", "high-b", "mid", "low"]


def test_aging_lets_an_old_low_priority_request_overtake():
    now = [0.0]
    scheduler = RequestScheduler(
        FakeExecutor(),
        config=single(aging_interval_s=1.0),
        clock=lambda: now[0],
    )

    async def go():
        scheduler.enqueue(make_request("old-low", priority=0))
        now[0] = 3.0  # waited 3 aging intervals -> effective priority 3
        scheduler._queue[0].enqueued_at = 0.0
        scheduler._queue.append(scheduler._queue[0].__class__(
            make_request("new-high", priority=2), 2, 99, 3.0,
            asyncio.get_running_loop().create_future(), None,
        ))
        picked = scheduler._pick_batch(now=3.0)
        scheduler._worker.cancel()
        return [h.request_id for h in picked]

    assert run(go()) == ["old-low"]


def test_without_aging_high_priority_keeps_winning():
    now = [0.0]
    scheduler = RequestScheduler(FakeExecutor(), config=single(aging_interval_s=None),
                                 clock=lambda: now[0])

    async def go():
        scheduler.enqueue(make_request("old-low", priority=0))
        scheduler._queue[0].enqueued_at = 0.0
        scheduler._queue.append(scheduler._queue[0].__class__(
            make_request("new-high", priority=2), 2, 99, 3.0,
            asyncio.get_running_loop().create_future(), None,
        ))
        picked = scheduler._pick_batch(now=1000.0)
        scheduler._worker.cancel()
        return [h.request_id for h in picked]

    assert run(go()) == ["new-high"]


def test_queue_full_is_rejected_explicitly_and_counted():
    async def go():
        executor = FakeExecutor(delay=0.05)
        scheduler = RequestScheduler(executor, config=single(max_queue_length=2))
        scheduler.enqueue(make_request("running"))
        await asyncio.sleep(0.01)
        scheduler.enqueue(make_request("q1"))
        scheduler.enqueue(make_request("q2"))
        with pytest.raises(QueueFullError):
            scheduler.enqueue(make_request("q3"))
        await scheduler.close()
        return scheduler.snapshot()

    snap = run(go())
    assert snap["counts"]["rejected"] == 1
    assert snap["counts"]["rejected_by_reason"] == {"queue_full": 1}
    assert snap["counts"]["completed"] == 3


def test_prompt_and_generation_limits():
    async def go():
        scheduler = RequestScheduler(
            FakeExecutor(),
            token_counter=lambda r: len(r.prompt.split()),
            config=single(max_prompt_tokens=3, max_new_tokens=8, max_batch_tokens=100),
        )
        with pytest.raises(PromptTooLongError):
            scheduler.enqueue(make_request("p", prompt="a b c d e"))
        with pytest.raises(GenerationTooLongError):
            scheduler.enqueue(make_request("g", prompt="a b", max_new_tokens=9))
        ok = scheduler.enqueue(make_request("ok", prompt="a b", max_new_tokens=8))
        await ok.result()
        await scheduler.close()

    run(go())


def test_expired_deadline_and_duplicate_id_and_closed_are_rejected():
    async def go():
        scheduler = RequestScheduler(FakeExecutor(delay=0.05), config=single())
        with pytest.raises(RequestRejectedError):
            scheduler.enqueue(make_request("d", deadline_s=0))
        scheduler.enqueue(make_request("dup"))
        with pytest.raises(DuplicateRequestError):
            scheduler.enqueue(make_request("dup"))
        await scheduler.close()
        with pytest.raises(SchedulerClosedError):
            scheduler.enqueue(make_request("late"))

    run(go())


def test_request_that_expires_in_the_queue_never_runs():
    async def go():
        executor = FakeExecutor(delay=0.15)
        scheduler = RequestScheduler(executor, config=single())
        running = scheduler.enqueue(make_request("running"))
        await asyncio.sleep(0.01)
        doomed = scheduler.enqueue(make_request("doomed", deadline_s=0.03))
        with pytest.raises(RequestTimeoutError):
            await doomed.result()
        await running.result()
        await scheduler.close()
        return executor.batches, doomed.state, scheduler.snapshot()

    batches, state, snap = run(go())
    assert batches == [["running"]]
    assert state is RequestState.TIMED_OUT
    assert snap["counts"]["timed_out"] == 1


def test_default_timeout_applies_when_request_has_no_deadline():
    async def go():
        scheduler = RequestScheduler(
            FakeExecutor(delay=0.15), config=single(default_timeout_s=0.03)
        )
        handle = scheduler.enqueue(make_request("slow"))
        with pytest.raises(RequestTimeoutError):
            await handle.result()
        await scheduler.close()

    run(go())


def test_timeout_while_running_releases_caller_and_drops_late_result():
    async def go():
        scheduler = RequestScheduler(FakeExecutor(delay=0.1), config=single())
        handle = scheduler.enqueue(make_request("slow", deadline_s=0.03))
        with pytest.raises(RequestTimeoutError):
            await handle.result()
        await scheduler.close()  # waits for the blocked executor call to finish
        return handle.state, scheduler.snapshot()

    state, snap = run(go())
    assert state is RequestState.TIMED_OUT  # not overwritten by the late result
    assert snap["counts"]["completed"] == 0


def test_cancel_queued_request():
    async def go():
        executor = FakeExecutor(delay=0.05)
        scheduler = RequestScheduler(executor, config=single())
        running = scheduler.enqueue(make_request("running"))
        await asyncio.sleep(0.01)
        victim = scheduler.enqueue(make_request("victim"))
        assert scheduler.cancel("victim") is True
        assert scheduler.cancel("victim") is False  # already finished
        assert scheduler.cancel("unknown") is False
        with pytest.raises(RequestCancelledError):
            await victim.result()
        await running.result()
        await scheduler.close()
        return executor.batches, scheduler.snapshot()

    batches, snap = run(go())
    assert batches == [["running"]]
    assert snap["counts"]["cancelled"] == 1


def test_cancelling_the_awaiting_task_cancels_the_request():
    async def go():
        scheduler = RequestScheduler(FakeExecutor(delay=0.05), config=single())
        blocker = scheduler.enqueue(make_request("blocker"))
        await asyncio.sleep(0.01)
        task = asyncio.create_task(scheduler.submit(make_request("abandoned")))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await blocker.result()
        await scheduler.close()
        return scheduler.state("abandoned")

    assert run(go()) is RequestState.CANCELLED


def test_executor_failure_marks_every_request_in_the_batch_failed():
    async def go():
        scheduler = RequestScheduler(
            FakeExecutor(fail=True),
            config=SchedulerConfig(max_batch_size=4, batching_window_ms=20.0),
        )
        handles = [scheduler.enqueue(make_request(f"f{i}")) for i in range(3)]
        for handle in handles:
            with pytest.raises(RuntimeError, match="boom"):
                await handle.result()
        await scheduler.close()
        return [h.state for h in handles], scheduler.snapshot()

    states, snap = run(go())
    assert states == [RequestState.FAILED] * 3
    assert snap["counts"]["failed"] == 3


def test_each_request_gets_only_its_own_result():
    async def go():
        scheduler = RequestScheduler(
            FakeExecutor(), config=SchedulerConfig(max_batch_size=8, batching_window_ms=20.0)
        )
        results = await asyncio.gather(
            *(scheduler.submit(make_request(f"r{i}", priority=i % 3)) for i in range(8))
        )
        await scheduler.close()
        return results

    results = run(go())
    assert [r.request_id for r in results] == [f"r{i}" for i in range(8)]


def test_wrong_request_id_in_result_is_a_failure():
    def bad_executor(requests):
        return [make_result(make_request("someone-else")) for _ in requests]

    async def go():
        scheduler = RequestScheduler(bad_executor, config=single())
        handle = scheduler.enqueue(make_request("mine"))
        with pytest.raises(RuntimeError, match="someone-else"):
            await handle.result()
        await scheduler.close()

    run(go())


def test_result_count_mismatch_fails_the_batch():
    async def go():
        scheduler = RequestScheduler(lambda reqs: [], config=single())
        handle = scheduler.enqueue(make_request("x"))
        with pytest.raises(RuntimeError, match="0 results for 1 requests"):
            await handle.result()
        await scheduler.close()

    run(go())


def test_batch_respects_size_and_token_limits():
    async def go():
        executor = FakeExecutor()
        scheduler = RequestScheduler(
            executor,
            token_counter=lambda r: len(r.prompt.split()),
            config=SchedulerConfig(
                max_batch_size=3, max_batch_tokens=10, batching_window_ms=30.0,
                aging_interval_s=None,
            ),
        )
        prompts = ["a b c d", "a b c d", "a b c d", "a", "a"]  # 4+4+4 > 10
        handles = [scheduler.enqueue(make_request(f"r{i}", prompt=p))
                   for i, p in enumerate(prompts)]
        await asyncio.gather(*(h.result() for h in handles))
        await scheduler.close()
        return executor.batches

    batches = run(go())
    assert all(len(b) <= 3 for b in batches)
    assert batches[0] == ["r0", "r1"]  # a third 4-token prompt would exceed 10
    assert sorted(i for b in batches for i in b) == [f"r{i}" for i in range(5)]


def test_lifecycle_states_are_recorded_in_order():
    async def go():
        scheduler = RequestScheduler(FakeExecutor(), config=single())
        handle = scheduler.enqueue(make_request("x"))
        await handle.result()
        await scheduler.close()
        return [state for state, _ in handle.history]

    assert run(go()) == [
        RequestState.QUEUED,
        RequestState.BATCHED,
        RequestState.PREFILLING,
        RequestState.COMPLETED,
    ]


def test_executor_can_report_decoding_progress():
    async def go():
        holder = {}

        def executor(requests):
            holder["scheduler"].report_progress(
                [r.request_id for r in requests], RequestState.DECODING
            )
            return [make_result(r) for r in requests]

        scheduler = RequestScheduler(executor, config=single())
        holder["scheduler"] = scheduler
        handle = scheduler.enqueue(make_request("x"))
        await handle.result()
        await scheduler.close()
        return [state for state, _ in handle.history]

    assert RequestState.DECODING in run(go())


def test_metrics_cover_queue_wait_service_ttft_and_throughput():
    async def go():
        scheduler = RequestScheduler(FakeExecutor(delay=0.03), config=single())
        await asyncio.gather(*(scheduler.submit(make_request(f"r{i}")) for i in range(3)))
        await scheduler.close()
        return scheduler.snapshot()

    snap = run(go())
    assert snap["counts"]["completed"] == 3
    assert snap["queue_wait_ms"]["max"] > snap["queue_wait_ms"]["p50"] >= 0
    assert snap["service_time_ms"]["mean"] >= 25
    assert snap["end_to_end_ms"]["max"] >= snap["service_time_ms"]["max"]
    assert snap["ttft_ms"]["mean"] == pytest.approx(1.0)
    assert snap["ttft_incl_queue_ms"]["max"] > snap["ttft_ms"]["max"]
    assert snap["generated_tokens"] == 6
    assert snap["requests_per_second"] > 0 and snap["tokens_per_second"] > 0
    assert snap["active"]["queued"] == 0 and snap["active"]["in_flight"] == 0
    assert snap["active"]["peak_queued"] >= 1


def test_batch_metrics_come_from_the_metrics_source():
    async def go():
        batch_metrics = (SimpleNamespace(
            effective_batch_size=2, padding_overhead_ratio=0.25, batch_formation_time_ms=1.5
        ),)
        scheduler = RequestScheduler(
            FakeExecutor(),
            config=SchedulerConfig(max_batch_size=2, batching_window_ms=30.0),
            batch_metrics_source=lambda: batch_metrics,
        )
        await asyncio.gather(*(scheduler.submit(make_request(f"r{i}")) for i in range(2)))
        await scheduler.close()
        return scheduler.snapshot()["batching"]

    batching = run(go())
    assert batching["mean_effective_batch_size"] == 2
    assert batching["mean_padding_overhead_ratio"] == pytest.approx(0.25)


def test_close_without_drain_cancels_waiting_requests():
    async def go():
        scheduler = RequestScheduler(FakeExecutor(delay=0.05), config=single())
        running = scheduler.enqueue(make_request("running"))
        await asyncio.sleep(0.01)
        waiting = scheduler.enqueue(make_request("waiting"))
        await scheduler.close(drain=False)
        with pytest.raises(RequestCancelledError):
            await waiting.result()
        return (await running.result()).request_id

    assert run(go()) == "running"


def test_auto_assigned_request_ids_are_unique():
    async def go():
        scheduler = RequestScheduler(FakeExecutor(), config=single())
        requests = [GenerationRequest("hi", GenerationConfig(max_new_tokens=2)) for _ in range(3)]
        results = await asyncio.gather(*(scheduler.submit(r) for r in requests))
        await scheduler.close()
        return [r.request_id for r in results]

    ids = run(go())
    assert len(set(ids)) == 3 and all(i.startswith("req-") for i in ids)