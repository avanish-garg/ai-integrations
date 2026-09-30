"""Main-agent recovery through the opt-in sibling SDK's pre-spawn callback."""

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

from temporalio.client import Client
from temporalio.worker import Replayer
from tests.helpers.fake_messages_api import history_of
from tests.hybrid.models import Call, Entry, Reply, State
from tests.hybrid.policy import chat, model_requests, rounds
from tests.hybrid.store import TranscriptStore
from tests.hybrid.test_engine import alive, echo, make_burst
from tests.hybrid.test_worker_loss import launch
from tests.hybrid.test_workflow import until
from tests.hybrid.workflows import HybridWorkflow

pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.skipif(
        not hasattr(ClaudeAgentOptions, "recover_pending_tool"),
        reason="requires the main-agent recovery SDK wheel from the sibling checkout",
    ),
]


def assert_results(api: Any, ids: set[str], rejected: set[str] | None = None) -> None:
    requests = model_requests(api)
    uses, _, history = history_of(requests[-1])
    assert set(uses) == ids == {h.id for h in history}
    assert {h.id for h in history if h.is_error} == (rejected or set())
    assert len(requests) == 2, "recovery must not ask the model to rediscover tools"
    assert api.errors == []


@pytest.mark.parametrize(
    "phase", ["before-scheduling", "after-completion", "partial-batch"]
)
@pytest.mark.parametrize("fork", [False, True], ids=["resume", "stored-fork"])
async def test_cli_loss_recovers_original_main_calls(
    tmp_path: Path, phase: str, fork: bool
) -> None:
    api = rounds(1, 3)
    ledger: dict[str, Entry] = {}
    seen: set[str] = set()
    ready = asyncio.Event()
    hold = asyncio.Event()

    async def pending(call: Call) -> Reply:
        seen.add(call.id)
        if phase != "before-scheduling":
            entry = Entry(call, scheduled=True)
            ledger[call.id] = entry
            if phase == "after-completion" or call.arguments["n"] == 0:
                entry.outcome = await echo(call)
        if len(seen) == 3:
            ready.set()
        if phase == "partial-batch" and call.arguments["n"] == 0:
            entry = ledger[call.id]
            assert entry.outcome is not None
            return entry.outcome
        await hold.wait()
        return await echo(call)

    burst = make_burst(tmp_path, api, pending)
    task: asyncio.Task[Any] | None = None
    restored = None
    recovered: list[str] = []
    newly_executed: list[str] = []
    try:
        await burst.open()
        task = asyncio.create_task(burst.query("work"))
        await asyncio.wait_for(ready.wait(), 15)
        delivered: set[str] = set()
        if phase == "partial-batch":
            first = next(c.id for c in burst.calls.values() if c.arguments["n"] == 0)

            async def mirrored() -> bool:
                entries = cast(
                    list[dict[str, Any]] | None, await burst.store.load(burst.key)
                )
                return any(
                    b.get("tool_use_id") == first
                    for e in entries or []
                    for b in e.get("message", {}).get("content", [])
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                )

            await until(mirrored)
            delivered.add(first)
        original_ids = set(burst.calls)
        old_pid = burst.pid
        burst.sdk._transport._process.kill()  # type: ignore[union-attr]
        with pytest.raises(Exception, match="Command failed|Process|exit code"):
            await task
        await burst.close()
        session = burst.session_id
        if fork:
            from claude_agent_sdk import fork_session_via_store

            session = (
                await fork_session_via_store(
                    cast(Any, burst.store), session, directory=str(tmp_path)
                )
            ).session_id  # type: ignore[arg-type]

        async def recover(call: Call) -> Reply:
            recovered.append(call.id)
            entry = ledger.setdefault(call.id, Entry(call))
            assert entry.call.identity() == call.identity()
            if entry.outcome is None:
                newly_executed.append(call.id)
                entry.outcome = await echo(call)
            return entry.outcome

        # UUIDs change in an explicit fork; only native tool IDs identify work.
        # The SDK verifies the new branch's own storage proof before callbacks.
        restored = make_burst(
            tmp_path,
            api,
            recover,
            store=burst.store,
            session=session,
            config="main-replacement",
            resume=True,
            recovery=True,
            recovery_entries=ledger if not fork else None,
        )
        await restored.open()
        assert restored.pid != old_pid
        assert (await restored.query("Continue the original work.")).result == "DONE 3"
        assert set(recovered) == original_ids - delivered
        if phase == "after-completion":
            assert not newly_executed
        await restored.checkpoint()
        assert_results(api, original_ids)
    finally:
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if restored is not None:
            await restored.close()
        await burst.close()
        api.stop()


async def test_worker_loss_during_second_task_preserves_first_answer(
    client: Client, address: str, tmp_path: Path
) -> None:
    api = chat()
    original = api.decide

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        blocks = original(body)
        for block in blocks:
            if block.get("type") == "tool_use" and block["input"]["n"] == 1:
                block["input"]["approval"] = True
        return blocks

    api.decide = policy
    queue = "main-second-task-" + uuid.uuid4().hex
    procs: list[subprocess.Popen[bytes]] = []
    try:
        procs.append(
            await launch(address, queue, tmp_path, api, 1, False, recovery=True)
        )
        handle = await client.start_workflow(
            HybridWorkflow.run,
            State(str(uuid.uuid4()), ["round 0", "round 1"]),
            id=queue,
            task_queue=queue,
        )

        async def second_pending() -> Any:
            snap = await handle.query(HybridWorkflow.snapshot)
            return snap if 0 in snap.turns and len(snap.ledger) == 2 else None

        before = await until(second_pending)
        assert before.turns[0].answer == "DONE 0; remembered 1"
        procs[0].kill()
        for row in map(
            json.loads, (tmp_path / "cli-pids.jsonl").read_text().splitlines()
        ):
            if alive(row["pid"]):
                os.kill(row["pid"], signal.SIGKILL)
        procs[0].wait()
        procs.append(
            await launch(address, queue, tmp_path, api, 2, False, recovery=True)
        )

        async def replaced() -> bool:
            return (await handle.query(HybridWorkflow.snapshot)).attempts[0].number == 2

        await until(replaced)
        pending = next(
            e.call for e in before.ledger.values() if e.call.arguments.get("approval")
        )
        await handle.execute_update(HybridWorkflow.review, (pending.id, True))
        state = await asyncio.wait_for(handle.result(), 45)
        assert state.answers == ["DONE 0; remembered 1", "DONE 1; remembered 2"]
        assert len(model_requests(api)) == 4 and api.errors == []
        assert state.turns[0] == before.turns[0]
        history = await handle.fetch_history()
        schedules = [
            e.activity_task_scheduled_event_attributes.activity_id
            for e in history.events
            if e.HasField("activity_task_scheduled_event_attributes")
            and e.activity_task_scheduled_event_attributes.activity_id.startswith(
                "tool-"
            )
        ]
        assert len(schedules) == len(set(schedules)) == 2
        await Replayer(workflows=[HybridWorkflow]).replay_workflow(history)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        log = tmp_path / "cli-pids.jsonl"
        if log.exists():
            for row in map(json.loads, log.read_text().splitlines()):
                if alive(row["pid"]):
                    os.kill(row["pid"], signal.SIGKILL)
        api.stop()


@pytest.mark.parametrize(
    "phase", ["before-update", "approval", "after-completion", "partial-batch"]
)
@pytest.mark.parametrize("approved", [True, False], ids=["approve", "reject"])
async def test_replacement_worker_recovers_main_calls(
    client: Client, address: str, tmp_path: Path, phase: str, approved: bool
) -> None:
    api = rounds(1, 3 if phase == "partial-batch" else 1)
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
    queue = "main-recovery-" + uuid.uuid4().hex
    procs: list[subprocess.Popen[bytes]] = []
    try:
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                1,
                phase == "after-completion",
                recovery=True,
                request_hold=phase == "before-update",
            )
        )
        handle = await client.start_workflow(
            HybridWorkflow.run,
            State(str(uuid.uuid4()), ["work"]),
            id=queue,
            task_queue=queue,
        )
        store = TranscriptStore(tmp_path / "store.db")

        async def reached() -> Any:
            snap = await handle.query(HybridWorkflow.snapshot)
            if phase == "before-update":
                if not model_requests(api):
                    return None
                # The native assistant request must be mirrored even though no
                # request Update made it to the Workflow before the Worker died.
                with store.connect() as db:
                    rows = db.execute("SELECT data FROM entries").fetchall()
                stored_call = any(
                    b.get("type") == "tool_use"
                    for row in rows
                    for b in json.loads(row[0]).get("message", {}).get("content", [])
                    if isinstance(b, dict)
                )
                return snap if stored_call and not snap.ledger else None
            if phase == "approval":
                return snap if snap.ledger else None
            if phase == "after-completion":
                return (
                    snap
                    if snap.ledger and all(e.outcome for e in snap.ledger.values())
                    else None
                )
            if len(snap.ledger) != 3:
                return None
            done = [e.call.id for e in snap.ledger.values() if e.outcome]
            if len(done) != 1:
                return None
            with store.connect() as db:
                rows = db.execute("SELECT data FROM entries").fetchall()
            mirrored = any(
                b.get("tool_use_id") == done[0]
                for row in rows
                for b in json.loads(row[0]).get("message", {}).get("content", [])
                if isinstance(b, dict) and b.get("type") == "tool_result"
            )
            return snap if mirrored else None

        before = await until(reached)
        original_ids = set(before.ledger)
        if phase == "before-update":
            # The API records the original IDs independently of the Workflow.
            with store.connect() as db:
                rows = db.execute("SELECT data FROM entries").fetchall()
            original_ids = {
                b["id"]
                for row in rows
                for b in json.loads(row[0]).get("message", {}).get("content", [])
                if isinstance(b, dict) and b.get("type") == "tool_use"
            }
        assert original_ids
        procs[0].kill()
        # Model a supervising host stopping the old Worker's CLI. Otherwise
        # that orphan can make an interruption model request after pipe loss.
        for row in map(
            json.loads, (tmp_path / "cli-pids.jsonl").read_text().splitlines()
        ):
            if alive(row["pid"]):
                os.kill(row["pid"], signal.SIGKILL)
        procs[0].wait()
        procs.append(
            await launch(address, queue, tmp_path, api, 2, False, recovery=True)
        )

        async def registered() -> Any:
            snap = await handle.query(HybridWorkflow.snapshot)
            return (
                snap if snap.attempts.get(0) and snap.attempts[0].number == 2 else None
            )

        await until(registered)
        rejected = set()
        for entry in before.ledger.values():
            if entry.call.arguments.get("approval"):
                await handle.execute_update(
                    HybridWorkflow.review, (entry.call.id, approved)
                )
                if not approved:
                    rejected.add(entry.call.id)
        state = await asyncio.wait_for(handle.result(), 45)
        assert state.answers == [f"DONE {len(original_ids)}"]
        assert set(state.ledger) == original_ids
        for tid, entry in before.ledger.items():
            if entry.outcome is not None:
                assert state.ledger[tid].outcome == entry.outcome
        history = await handle.fetch_history()
        tool_ids = [
            e.activity_task_scheduled_event_attributes.activity_id
            for e in history.events
            if e.HasField("activity_task_scheduled_event_attributes")
            and e.activity_task_scheduled_event_attributes.activity_id.startswith(
                "tool-"
            )
        ]
        assert len(tool_ids) == len(set(tool_ids)) == len(original_ids - rejected)
        assert_results(api, original_ids, rejected)
        await Replayer(workflows=[HybridWorkflow]).replay_workflow(history)
        assert len((tmp_path / "cli-pids.jsonl").read_text().splitlines()) == 2
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        log = tmp_path / "cli-pids.jsonl"
        if log.exists():
            for row in map(json.loads, log.read_text().splitlines()):
                if alive(row["pid"]):
                    os.kill(row["pid"], signal.SIGKILL)
        api.stop()
