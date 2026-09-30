"""Real Temporal ledger, approvals, attempt fencing, cancellation and rollover."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.client import Client, WorkflowUpdateFailedError, WorkflowUpdateStage
from temporalio.worker import Replayer, Worker
from tests.helpers.fake_messages_api import engine_env, history_of
from tests.hybrid.activities import HybridActivities
from tests.hybrid.models import Attempt, Call, Checkpoint, State
from tests.hybrid.policy import chat, model_requests, rounds
from tests.hybrid.store import TranscriptStore
from tests.hybrid.test_engine import alive
from tests.hybrid.workflows import HybridWorkflow

pytestmark = pytest.mark.timeout(120)


async def until(check: Any, timeout: float = 30) -> Any:
    async def poll() -> Any:
        while True:
            value = await check()
            if value:
                return value
            await asyncio.sleep(0.02)

    return await asyncio.wait_for(poll(), timeout)


def runtime(client: Client, tmp_path: Path, api: Any) -> HybridActivities:
    return HybridActivities(
        client,
        tmp_path,
        engine_env(api, str(tmp_path / "cfg")),
        TranscriptStore(tmp_path / "store.db"),
    )


def worker(client: Client, queue: str, acts: HybridActivities) -> Worker:
    return Worker(
        client,
        task_queue=queue,
        workflows=[HybridWorkflow],
        activities=[acts.burst, acts.tool],
    )


async def test_temporal_batch_and_history_replay(
    client: Client, tmp_path: Path
) -> None:
    api = rounds(1, 3, delay=0.5)
    acts = runtime(client, tmp_path, api)
    queue = "hybrid-" + uuid.uuid4().hex
    try:
        async with worker(client, queue, acts):
            handle = await client.start_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["batch"]),
                id=queue,
                task_queue=queue,
            )
            state = await handle.result()
        assert state.answers == ["DONE 3"]
        assert len(acts.tool_calls) == len(state.ledger) == 3
        assert max(acts.started.values()) < min(acts.finished.values())
        assert all(e.scheduled and e.outcome for e in state.ledger.values())
        assert len(state.checkpoints) == 1
        history = await handle.fetch_history()
        schedules = [
            e.activity_task_scheduled_event_attributes.activity_id
            for e in history.events
            if e.HasField("activity_task_scheduled_event_attributes")
        ]
        assert set(schedules) >= {"tool-" + tid for tid in state.ledger}
        assert (
            sum(
                e.HasField("workflow_execution_update_completed_event_attributes")
                for e in history.events
            )
            >= 5
        )
        await Replayer(workflows=[HybridWorkflow]).replay_workflow(history)
        assert len(model_requests(api)) == 2 and api.errors == []
    finally:
        api.stop()


async def test_approvals_duplicates_conflicts_and_stale_attempts(
    client: Client, tmp_path: Path
) -> None:
    api = rounds(1, 3, approval=True)
    acts = runtime(client, tmp_path, api)
    queue = "approval-" + uuid.uuid4().hex
    try:
        async with worker(client, queue, acts):
            handle = await client.start_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["review"]),
                id=queue,
                task_queue=queue,
            )

            async def pending() -> Any:
                snap = await handle.query(HybridWorkflow.snapshot)
                return snap if len(snap.ledger) == 3 else None

            snap = await until(pending)
            entries = list(snap.ledger.values())
            call = entries[0].call
            duplicate = await handle.start_update(
                HybridWorkflow.request,
                call,
                wait_for_stage=WorkflowUpdateStage.ACCEPTED,
            )
            bad = Call(
                call.id, call.name, {"n": -1}, call.attempt, call.transcript_uuid
            )
            with pytest.raises(WorkflowUpdateFailedError) as failed:
                await handle.execute_update(HybridWorkflow.request, bad)
            assert failed.value.cause and "conflicting" in str(failed.value.cause)
            stale = Attempt(call.attempt.burst, 0, "superseded")
            with pytest.raises(WorkflowUpdateFailedError) as failed:
                await handle.execute_update(HybridWorkflow.register, stale)
            assert failed.value.cause and "stale" in str(failed.value.cause)
            with pytest.raises(WorkflowUpdateFailedError):
                await handle.execute_update(
                    HybridWorkflow.request, Call("stale", "echo", {}, stale, "stored")
                )
            with pytest.raises(WorkflowUpdateFailedError):
                await handle.execute_update(
                    HybridWorkflow.acknowledge,
                    Checkpoint(call.attempt, "s", "u", [], 1),
                )
            assert not await acts.bursts[0].suspend_pending() and alive(
                acts.bursts[0].pid
            )
            for i, entry in enumerate(entries):
                await handle.execute_update(
                    HybridWorkflow.review, (entry.call.id, i != 1)
                )
            reply = await duplicate.result()
            # A second duplicate after completion must reuse the recorded outcome.
            assert await handle.execute_update(HybridWorkflow.request, call) == reply
            state = await handle.result()
        assert len(acts.tool_calls) == 2 and len(set(acts.tool_calls)) == 2
        rejected = state.ledger[entries[1].call.id].outcome
        assert rejected is not None and rejected.is_error
        _, _, history = history_of(model_requests(api)[-1])
        assert {h.id for h in history if h.is_error} == {entries[1].call.id}
        assert api.errors == []
    finally:
        api.stop()


async def test_continue_as_new_after_stopped_cli_and_finished_handlers(
    client: Client, tmp_path: Path
) -> None:
    api = chat()
    acts = runtime(client, tmp_path, api)
    queue = "rollover-" + uuid.uuid4().hex
    try:
        async with worker(client, queue, acts):
            handle = await client.start_workflow(
                HybridWorkflow.run,
                State(
                    str(uuid.uuid4()),
                    [f"round {n}" for n in range(3)],
                    burst_size=1,
                    continue_every=1,
                ),
                id=queue,
                task_queue=queue,
            )
            state = await handle.result()
        assert state.answers == [f"DONE {n}; remembered {n + 1}" for n in range(3)]
        assert len(state.ledger) == len(state.checkpoints) == 3
        assert len({b.pid for b in acts.bursts}) == 3
        assert all(b.closed and not alive(b.pid) for b in acts.bursts)
        assert state.checkpoint is not None
        assert (
            handle.first_execution_run_id
            != state.checkpoint.attempt.token.split(":")[0]
        )
        assert api.errors == []
    finally:
        api.stop()


async def test_cancellation_stops_cli_and_pending_callbacks(
    client: Client, tmp_path: Path
) -> None:
    api = rounds(1, approval=True)
    acts = runtime(client, tmp_path, api)
    queue = "cancel-" + uuid.uuid4().hex
    try:
        async with worker(client, queue, acts):
            handle = await client.start_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["approval"]),
                id=queue,
                task_queue=queue,
            )

            async def pending() -> Any:
                return (await handle.query(HybridWorkflow.snapshot)).ledger

            await until(pending)
            await handle.cancel()

            # Cancellation of a Workflow must drain its accepted approval Update too.
            async def stopped() -> bool:
                return bool(acts.bursts[0].closed and not alive(acts.bursts[0].pid))

            await until(stopped)
            assert not acts.bursts[0].callbacks and not acts.tool_calls
    finally:
        api.stop()


async def test_retry_after_side_effect_uses_stable_idempotency_key(
    client: Client, tmp_path: Path
) -> None:
    api = rounds(1)
    acts = runtime(client, tmp_path, api)
    acts.fail_after_effect = True
    queue = "effect-retry-" + uuid.uuid4().hex
    try:
        async with worker(client, queue, acts):
            state = await client.execute_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["echo"]),
                id=queue,
                task_queue=queue,
            )
        assert len(acts.tool_calls) == 2 and len(set(acts.tool_calls)) == 1
        with acts.store.connect() as db:
            assert db.execute("SELECT count(*) FROM effects").fetchone()[0] == 1
        entry = next(iter(state.ledger.values()))
        assert entry.outcome and queue + ":tool-" + entry.call.id in entry.outcome.text
    finally:
        api.stop()


async def test_new_registration_fences_an_actual_older_attempt(
    client: Client, tmp_path: Path
) -> None:
    api = rounds(1, approval=True)
    acts = runtime(client, tmp_path, api)
    queue = "fence-" + uuid.uuid4().hex
    try:
        async with worker(client, queue, acts):
            handle = await client.start_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["approve"]),
                id=queue,
                task_queue=queue,
            )

            async def pending() -> Any:
                snap = await handle.query(HybridWorkflow.snapshot)
                return snap if snap.ledger else None

            snap = await until(pending)
            call = next(iter(snap.ledger.values())).call
            newer = Attempt(call.attempt.burst, call.attempt.number + 1, "replacement")
            await handle.execute_update(HybridWorkflow.register, newer)
            with pytest.raises(WorkflowUpdateFailedError) as failed:
                await handle.execute_update(HybridWorkflow.request, call)
            assert failed.value.cause and "superseded" in str(failed.value.cause)
            assert not acts.tool_calls
            await handle.cancel()
    finally:
        api.stop()
