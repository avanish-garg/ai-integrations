"""Controlled main-agent hard stops preserve real pending calls and results."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from temporalio.client import Client, WorkflowUpdateFailedError
from temporalio.worker import Replayer
from tests.hybrid.models import Call, Entry, Reply, State
from tests.hybrid.policy import model_requests, rounds
from tests.hybrid.test_engine import alive, echo, make_burst
from tests.hybrid.test_main_recovery import assert_results
from tests.hybrid.test_worker_loss import launch
from tests.hybrid.test_workflow import runtime, until, worker
from tests.hybrid.workflows import HybridWorkflow

pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.skipif(
        not hasattr(ClaudeAgentOptions, "recover_pending_tool"),
        reason="requires the main-agent recovery SDK wheel",
    ),
]


@pytest.mark.parametrize("phase", ["approval", "after-completion", "partial-batch"])
@pytest.mark.parametrize("approved", [True, False], ids=["approve", "reject"])
async def test_pending_stop_keeps_native_batch(
    tmp_path: Path, phase: str, approved: bool
) -> None:
    api = rounds(1, 3)
    ledger: dict[str, Entry] = {}
    hold = asyncio.Event()

    async def original(call: Call) -> Reply:
        entry = Entry(call, scheduled=phase != "approval")
        ledger[call.id] = entry
        if phase == "after-completion" or (
            phase == "partial-batch" and call.arguments["n"] == 0
        ):
            entry.outcome = await echo(call)
        if phase == "partial-batch" and call.arguments["n"] == 0:
            assert entry.outcome is not None
            return entry.outcome
        await hold.wait()
        return await echo(call)

    burst = make_burst(tmp_path, api, original, recovery=True)
    running: asyncio.Task[Any] | None = None
    restored = None
    recovered: list[str] = []
    try:
        await burst.open()
        running = asyncio.create_task(burst.query("work"))

        async def ready() -> bool:
            if len(ledger) != 3 or not burst.batch_complete:
                return False
            if phase != "partial-batch":
                return True
            entries = cast(
                list[dict[str, Any]], await burst.store.load(burst.key) or []
            )
            return any(
                block.get("type") == "tool_result"
                for entry in entries or []
                for block in entry.get("message", {}).get("content", [])
                if isinstance(block, dict)
            )

        await until(ready)
        before = await burst.store.load(burst.key)
        pid = burst.pid
        assert await burst.suspend_pending()
        assert burst.suspended is not None
        assert not alive(pid) and not burst.callbacks
        assert await burst.store.load(burst.key) == before
        await asyncio.gather(running, return_exceptions=True)
        assert len(model_requests(api)) == 1
        rejected = set()

        async def recover(call: Call) -> Reply:
            recovered.append(call.id)
            entry = ledger[call.id]
            if entry.outcome is None:
                entry.outcome = (
                    await echo(call) if approved else Reply("rejected", True)
                )
                if not approved:
                    rejected.add(call.id)
            return entry.outcome

        restored = make_burst(
            tmp_path,
            api,
            recover,
            store=burst.store,
            session=burst.session_id,
            config="replacement",
            resume=True,
            recovery=True,
            recovery_entries=ledger,
        )
        await restored.open()
        assert restored.pid != pid
        assert (await restored.query("Continue the original work.")).result == "DONE 3"
        assert set(recovered) == set(burst.suspended.pending)
        assert_results(api, set(ledger), rejected)
        await restored.checkpoint()
    finally:
        if running and not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        if restored is not None:
            await restored.close()
        await burst.close()
        api.stop()


async def test_incomplete_stream_or_missing_call_keeps_cli_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = rounds(1, 3)
    hold = asyncio.Event()

    async def wait(call: Call) -> Reply:
        await hold.wait()
        return await echo(call)

    burst = make_burst(tmp_path, api, wait, recovery=True)
    running = None
    try:
        await burst.open()
        running = asyncio.create_task(burst.query("work"))

        async def ready() -> bool:
            return len(burst.calls) == 3 and burst.batch_complete

        await until(ready)
        burst.batch_complete = False
        assert not await burst.suspend_pending() and alive(burst.pid)
        burst.batch_complete = True
        load = burst.store.load

        async def missing(key: Any) -> Any:
            entries = await load(key)
            return [
                e
                for e in entries or []
                if e.get("uuid") != next(iter(burst.calls.values())).transcript_uuid
            ]

        with monkeypatch.context() as context:
            context.setattr(burst.store, "load", missing)
            assert not await burst.suspend_pending() and alive(burst.pid)
            assert burst.delivery.is_set()

        async def unavailable(key: Any) -> Any:
            del key
            raise OSError("injected storage read failure")

        with monkeypatch.context() as context:
            context.setattr(burst.store, "load", unavailable)
            assert not await burst.suspend_pending() and alive(burst.pid)
            assert burst.delivery.is_set()
            assert isinstance(burst.suspension_failure, OSError)
        hold.set()
        assert (await running).result == "DONE 3"
    finally:
        if running and not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        await burst.close()
        api.stop()


async def test_delayed_request_storage_keeps_cli_alive(tmp_path: Path) -> None:
    api = rounds(1, 3)
    hold = asyncio.Event()

    async def wait(call: Call) -> Reply:
        await hold.wait()
        return await echo(call)

    burst = make_burst(tmp_path, api, wait, recovery=True)
    burst.store.delay = 0.5
    running = None
    try:
        await burst.open()
        running = asyncio.create_task(burst.query("work"))
        await asyncio.wait_for(burst.store.writing.wait(), 10)
        assert not await burst.suspend_pending() and alive(burst.pid)

        async def ready() -> bool:
            return len(burst.calls) == 3 and burst.batch_complete

        await until(ready)
        assert await burst.suspend_pending() and not alive(burst.pid)
        await asyncio.gather(running, return_exceptions=True)
    finally:
        if running and not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        await burst.close()
        api.stop()


@pytest.mark.parametrize(
    "phase", ["approval", "running", "after-completion", "partial-batch"]
)
@pytest.mark.parametrize("approved", [True, False], ids=["approve", "reject"])
async def test_workflow_releases_cli_activity_with_pending_calls(
    client: Client, tmp_path: Path, phase: str, approved: bool
) -> None:
    api = rounds(1, 3, delay=2 if phase == "running" else 0)
    original = api.decide

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        blocks = original(body)
        for block in blocks:
            if block.get("type") == "tool_use":
                block["input"]["approval"] = phase == "approval" or (
                    phase == "partial-batch" and block["input"]["n"] != 0
                )
        return blocks

    api.decide = policy
    acts = runtime(client, tmp_path, api)
    acts.recovery = True
    if phase == "after-completion":
        acts.before_delivery = asyncio.Event()
    queue = "pending-suspension-" + uuid.uuid4().hex
    try:
        async with worker(client, queue, acts):
            handle = await client.start_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["work"]),
                id=queue,
                task_queue=queue,
            )

            async def ready() -> Any:
                snap = await handle.query(HybridWorkflow.snapshot)
                if len(snap.ledger) != 3:
                    return None
                done = sum(entry.outcome is not None for entry in snap.ledger.values())
                if phase == "after-completion" and done != 3:
                    return None
                if phase == "running" and not all(
                    entry.scheduled for entry in snap.ledger.values()
                ):
                    return None
                if phase == "partial-batch":
                    if done != 1:
                        return None
                    entries = cast(
                        list[dict[str, Any]],
                        await acts.store.load(acts.bursts[0].key) or [],
                    )
                    if not any(
                        block.get("type") == "tool_result"
                        for entry in entries or []
                        for block in entry.get("message", {}).get("content", [])
                        if isinstance(block, dict)
                    ):
                        return None
                return snap

            before = await until(ready)
            await handle.execute_update(HybridWorkflow.suspend)

            async def stopped() -> Any:
                snap = await handle.query(HybridWorkflow.snapshot)
                history = await handle.fetch_history()
                completed = any(
                    event.HasField("activity_task_completed_event_attributes")
                    and event.activity_task_completed_event_attributes.result.payloads
                    and b'"pending"'
                    in event.activity_task_completed_event_attributes.result.payloads[
                        0
                    ].data
                    for event in history.events
                )
                return snap if snap.pending is not None and completed else None

            paused = await until(stopped)
            proof = paused.pending
            assert proof is not None
            assert not alive(proof.pid) and acts.bursts[0].closed
            assert not acts.bursts[0].callbacks
            assert len(model_requests(api)) == 1 and len(acts.bursts) == 1
            assert set(proof.pending) | set(proof.delivered) == set(before.ledger)
            assert len(proof.delivered) == (1 if phase == "partial-batch" else 0)
            await handle.execute_update(HybridWorkflow.suspend)
            await handle.execute_update(HybridWorkflow.acknowledge_suspension, proof)
            with pytest.raises(WorkflowUpdateFailedError) as failed:
                await handle.execute_update(
                    HybridWorkflow.request, next(iter(before.ledger.values())).call
                )
            assert failed.value.cause and "suspended CLI attempt" in str(
                failed.value.cause
            )
            await handle.execute_update(HybridWorkflow.resume_pending)
            rejected = set()
            for entry in before.ledger.values():
                if entry.call.arguments.get("approval"):
                    # No replacement CLI/Activity starts before decisions/outcomes.
                    assert len(acts.bursts) == 1
                    await handle.execute_update(
                        HybridWorkflow.review, (entry.call.id, approved)
                    )
                    if not approved:
                        rejected.add(entry.call.id)
            state = await asyncio.wait_for(handle.result(), 30)
            assert state.answers == ["DONE 3"] and len(state.suspensions) == 1
            assert len(acts.bursts) == 2 and acts.bursts[1].pid != proof.pid
            assert (
                len(acts.tool_calls) == len(set(acts.tool_calls)) == 3 - len(rejected)
            )
            for tid, entry in before.ledger.items():
                if entry.outcome is not None:
                    assert state.ledger[tid].outcome == entry.outcome
            assert_results(api, set(before.ledger), rejected)
            history = await handle.fetch_history()
            await Replayer(workflows=[HybridWorkflow]).replay_workflow(history)
    finally:
        api.stop()


@pytest.mark.parametrize(
    "rollover", [False, True], ids=["worker-replay", "continue-as-new"]
)
async def test_suspended_work_restores_after_worker_replacement(
    client: Client, tmp_path: Path, rollover: bool
) -> None:
    api = rounds(1, 3, approval=True)
    queue = "suspended-replacement-" + uuid.uuid4().hex
    acts = runtime(client, tmp_path, api)
    acts.recovery = True
    try:
        async with worker(client, queue, acts):
            handle = await client.start_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["work"], continue_on_suspend=rollover),
                id=queue,
                task_queue=queue,
            )

            async def ready() -> bool:
                return len((await handle.query(HybridWorkflow.snapshot)).ledger) == 3

            await until(ready)
            await handle.execute_update(HybridWorkflow.suspend)

            async def stopped() -> Any:
                snap = await handle.query(HybridWorkflow.snapshot)
                return snap if snap.pending is not None else None

            paused = await until(stopped)
            assert not alive(paused.pending.pid)
        replacement = runtime(client, tmp_path, api)
        replacement.recovery = True
        async with worker(client, queue, replacement):
            await handle.execute_update(HybridWorkflow.resume_pending)
            for entry in paused.ledger.values():
                await handle.execute_update(
                    HybridWorkflow.review, (entry.call.id, True)
                )
            state = await asyncio.wait_for(handle.result(), 30)
        assert state.answers == ["DONE 3"] and len(state.suspensions) == 1
        assert len(replacement.bursts) == 1
        assert_results(api, set(paused.ledger))
        history = await handle.fetch_history()
        schedules = [
            event.activity_task_scheduled_event_attributes.activity_id
            for event in history.events
            if event.HasField("activity_task_scheduled_event_attributes")
            and event.activity_task_scheduled_event_attributes.activity_id.startswith(
                "tool-"
            )
        ]
        if not rollover:
            assert len(schedules) == len(set(schedules)) == 3
        else:
            assert not schedules, "Continue-As-New must reuse earlier outcomes"
        await Replayer(workflows=[HybridWorkflow]).replay_workflow(history)
    finally:
        api.stop()


async def test_repeated_suspension_within_one_task(
    client: Client, tmp_path: Path
) -> None:
    api = rounds(2, 3, approval=True)
    acts = runtime(client, tmp_path, api)
    acts.recovery = True
    queue = "repeat-suspension-" + uuid.uuid4().hex
    try:
        async with worker(client, queue, acts):
            handle = await client.start_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["work"]),
                id=queue,
                task_queue=queue,
            )
            seen: set[str] = set()
            for size in (3, 6):

                async def ready() -> Any:
                    snap = await handle.query(HybridWorkflow.snapshot)
                    return snap if len(snap.ledger) == size else None

                current = await until(ready)
                await handle.execute_update(HybridWorkflow.suspend)

                async def stopped() -> Any:
                    snap = await handle.query(HybridWorkflow.snapshot)
                    return snap if snap.pending is not None else None

                paused = await until(stopped)
                proof = paused.pending
                assert proof is not None and not alive(proof.pid)
                assert set(proof.pending) == set(current.ledger) - seen
                assert set(proof.delivered) == seen
                await handle.execute_update(HybridWorkflow.resume_pending)
                for tid in set(current.ledger) - seen:
                    await handle.execute_update(HybridWorkflow.review, (tid, True))
                seen = set(current.ledger)
            state = await asyncio.wait_for(handle.result(), 30)
        assert state.answers == ["DONE 6"] and len(state.suspensions) == 2
        assert len(acts.bursts) == 3
        assert len(acts.tool_calls) == len(set(acts.tool_calls)) == 6
        assert len(model_requests(api)) == 3 and not api.errors
        await Replayer(workflows=[HybridWorkflow]).replay_workflow(
            await handle.fetch_history()
        )
    finally:
        api.stop()


async def test_cancelling_suspended_work_drains_approval_handlers(
    client: Client, tmp_path: Path
) -> None:
    api = rounds(1, 3, approval=True)
    acts = runtime(client, tmp_path, api)
    acts.recovery = True
    queue = "cancel-suspension-" + uuid.uuid4().hex
    try:
        async with worker(client, queue, acts):
            handle = await client.start_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["work"]),
                id=queue,
                task_queue=queue,
            )

            async def ready() -> bool:
                return len((await handle.query(HybridWorkflow.snapshot)).ledger) == 3

            await until(ready)
            await handle.execute_update(HybridWorkflow.suspend)

            async def stopped() -> Any:
                return (await handle.query(HybridWorkflow.snapshot)).pending

            proof = await until(stopped)
            await handle.cancel()
            from temporalio.client import WorkflowFailureError
            from temporalio.exceptions import CancelledError

            with pytest.raises(WorkflowFailureError) as failed:
                await asyncio.wait_for(handle.result(), 10)
            assert isinstance(failed.value.cause, CancelledError)
            assert not alive(proof.pid) and not acts.tool_calls
    finally:
        api.stop()


@pytest.mark.parametrize("count", [1, 2], ids=["first-stop", "repeated-stop"])
async def test_worker_loss_after_suspension_ack_before_activity_completion(
    client: Client, address: str, tmp_path: Path, count: int
) -> None:
    api = rounds(count, 3, approval=True)
    queue = "lost-suspension-receipt-" + uuid.uuid4().hex
    processes: list[subprocess.Popen[bytes]] = []
    try:
        processes.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                1,
                False,
                recovery=True,
                suspension_hold=True,
            )
        )
        handle = await client.start_workflow(
            HybridWorkflow.run,
            State(str(uuid.uuid4()), ["work"]),
            id=queue,
            task_queue=queue,
        )
        seen: set[str] = set()
        for index in range(count):

            async def ready() -> Any:
                snap = await handle.query(HybridWorkflow.snapshot)
                return snap if len(snap.ledger) == (index + 1) * 3 else None

            current = await until(ready)
            await handle.execute_update(HybridWorkflow.suspend)

            async def stopped() -> Any:
                return (await handle.query(HybridWorkflow.snapshot)).pending

            proof = await until(stopped)
            assert not alive(proof.pid)
            processes[-1].kill()
            processes[-1].wait()
            processes.append(
                await launch(
                    address,
                    queue,
                    tmp_path,
                    api,
                    index + 2,
                    False,
                    recovery=True,
                    suspension_hold=index + 1 < count,
                )
            )
            await handle.execute_update(HybridWorkflow.resume_pending)
            for tid in set(current.ledger) - seen:
                await handle.execute_update(HybridWorkflow.review, (tid, True))
            seen = set(current.ledger)
        state = await asyncio.wait_for(handle.result(), 45)
        assert state.answers == [f"DONE {count * 3}"]
        assert len(state.suspensions) == count
        assert len(model_requests(api)) == count + 1 and not api.errors
        assert len((tmp_path / "cli-pids.jsonl").read_text().splitlines()) == count + 1
        history = await handle.fetch_history()
        schedules = [
            event.activity_task_scheduled_event_attributes.activity_id
            for event in history.events
            if event.HasField("activity_task_scheduled_event_attributes")
            and event.activity_task_scheduled_event_attributes.activity_id.startswith(
                "tool-"
            )
        ]
        assert len(schedules) == len(set(schedules)) == count * 3
        await Replayer(workflows=[HybridWorkflow]).replay_workflow(history)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        log = tmp_path / "cli-pids.jsonl"
        if log.exists():
            for row in map(json.loads, log.read_text().splitlines()):
                if alive(row["pid"]):
                    os.kill(row["pid"], signal.SIGKILL)
        api.stop()
