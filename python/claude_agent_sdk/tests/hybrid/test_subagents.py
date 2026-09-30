"""Parent/child native MCP calls, child persistence and process-loss probes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from tests.helpers.fake_messages_api import FakeMessagesAPI, history_of
from tests.hybrid.models import Call, Reply
from tests.hybrid.policy import model_requests
from tests.hybrid.test_engine import echo, make_burst

pytestmark = pytest.mark.timeout(120)
SUBTASK = "HYBRID CHILD: echo number 100."


def subagent_api(background: bool) -> FakeMessagesAPI:
    holder: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        api = holder[0]
        uses, texts, history = history_of(body)
        child = any(SUBTASK in text for text in texts)
        if child:
            if history:
                return [{"type": "text", "text": "CHILD DONE"}]
            return [api.tool_use("echo", {"n": 100})]
        if not any(name == "Agent" for name, _ in uses.values()):
            return [
                {
                    "type": "tool_use",
                    "id": api.next_id("toolu_agent"),
                    "name": "Agent",
                    "input": {
                        "description": "echo in a child",
                        "prompt": SUBTASK,
                        "subagent_type": "general-purpose",
                        "run_in_background": background,
                    },
                }
            ]
        if not any(h.name == "echo" for h in history):
            return [api.tool_use("echo", {"n": 1})]
        return [{"type": "text", "text": "PARENT DONE"}]

    api = FakeMessagesAPI(decide)
    holder.append(api)
    return api.start()


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "background"])
async def test_subagent_live_and_stored(tmp_path: Path, background: bool) -> None:
    api = subagent_api(background)
    seen: list[Call] = []

    async def tracked(call: Call) -> Reply:
        seen.append(call)
        return await echo(call)

    burst = make_burst(tmp_path, api, tracked, subagents=True)
    try:
        await burst.open()
        result = await burst.query("Delegate one echo, then echo number 1 yourself.")
        print(
            "HYBRID_SUBAGENT "
            + json.dumps(
                {
                    "background": background,
                    "result": result.result,
                    "failure": str(burst.failure),
                    "calls": [
                        {"id": c.id, "n": c.arguments["n"], "subpath": c.subpath}
                        for c in seen
                    ],
                    "observed": list(burst.observed),
                    "children": list(burst.active_children),
                }
            ),
            flush=True,
        )
        assert result.result == "PARENT DONE"
        assert sorted(c.arguments["n"] for c in seen) == [1, 100]
        child = next(c for c in seen if c.arguments["n"] == 100)
        assert child.subpath
        key = burst.key
        subkeys = await burst.store.list_subkeys(key)
        assert child.subpath in subkeys
        assert await burst.store.load({**key, "subpath": child.subpath})
        await burst.checkpoint()
        await burst.close()
        restored = make_burst(
            tmp_path,
            api,
            store=burst.store,
            session=burst.session_id,
            config="child-restore",
            resume=True,
            subagents=True,
        )
        try:
            await restored.open()
            assert any(k.get("subpath") == child.subpath for k in burst.store.loads)
            answer = await restored.query("Continue.")
            assert answer.result == "PARENT DONE"
        finally:
            await restored.close()
        assert api.errors == []
        assert len(model_requests(api)) >= 4
    finally:
        await burst.close()
        api.stop()


async def test_background_parent_child_requests_overlap(tmp_path: Path) -> None:
    api = subagent_api(True)
    seen: list[Call] = []
    both = asyncio.Event()

    async def concurrent(call: Call) -> Reply:
        seen.append(call)
        if len(seen) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 15)
        return await echo(call)

    burst = make_burst(tmp_path, api, concurrent, subagents=True)
    try:
        await burst.open()
        assert (
            await burst.query("Delegate and echo yourself.")
        ).result == "PARENT DONE"
        assert {c.arguments["n"] for c in seen} == {1, 100}
        assert len({c.id for c in seen}) == 2
        assert {bool(c.subpath) for c in seen} == {False, True}
        await burst.checkpoint()
    finally:
        await burst.close()
        api.stop()


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "background"])
@pytest.mark.parametrize(
    "completed", [False, True], ids=["pending", "completed-before-delivery"]
)
async def test_child_process_loss_does_not_restore_pending_call(
    tmp_path: Path, background: bool, completed: bool
) -> None:
    api = subagent_api(background)
    child_ready = asyncio.Event()
    release = asyncio.Event()
    child_calls: list[Call] = []
    outcomes: dict[str, Reply] = {}

    async def pending_child(call: Call) -> Reply:
        if call.arguments["n"] == 100:
            child_calls.append(call)
            if completed:
                outcomes[call.id] = await echo(call)
            child_ready.set()
            await release.wait()
        return await echo(call)

    burst = make_burst(tmp_path, api, pending_child, subagents=True)
    task: asyncio.Task[Any] | None = None
    try:
        await burst.open()
        task = asyncio.create_task(burst.query("Delegate and echo yourself."))
        await asyncio.wait_for(child_ready.wait(), 15)
        child = child_calls[0]
        assert child.subpath and await burst.store.load(
            {**burst.key, "subpath": child.subpath}
        )
        transport: Any = burst.sdk._transport
        transport._process.kill()
        with pytest.raises(Exception):
            await task
        await burst.close()
        restored_calls: list[Call] = []

        async def on_restore(call: Call) -> Reply:
            restored_calls.append(call)
            return await echo(call)

        restored = make_burst(
            tmp_path,
            api,
            on_restore,
            store=burst.store,
            session=burst.session_id,
            config="lost-child-replacement",
            resume=True,
            subagents=True,
        )
        try:
            await restored.open()
            await restored.query("Continue.")
            assert child.id not in [c.id for c in restored_calls]
            assert any(k.get("subpath") == child.subpath for k in burst.store.loads)
            saved = await burst.store.load({**burst.key, "subpath": child.subpath})
            assert saved
            assert not any(
                block.get("type") == "tool_result"
                and block.get("tool_use_id") == child.id
                and not block.get("is_error")
                for entry in cast(list[dict[str, Any]], saved)
                for block in entry.get("message", {}).get("content", [])
                if isinstance(block, dict)
            )
            print(
                "HYBRID_CHILD_LOSS "
                + json.dumps(
                    {
                        "background": background,
                        "lost_id": child.id,
                        "completed_outcomes": {k: v.text for k, v in outcomes.items()},
                        "restored_ids": [c.id for c in restored_calls],
                        "subpath": child.subpath,
                    }
                ),
                flush=True,
            )
            assert api.errors == []
        finally:
            await restored.close()
    finally:
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await burst.close()
        api.stop()


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "background"])
async def test_subagent_tools_are_temporal_activities(
    client: Any, tmp_path: Path, background: bool
) -> None:
    import uuid

    from tests.hybrid.models import State
    from tests.hybrid.test_workflow import runtime, worker
    from tests.hybrid.workflows import HybridWorkflow

    api = subagent_api(background)
    acts = runtime(client, tmp_path, api)
    acts.subagents = True
    queue = "child-activities-" + uuid.uuid4().hex
    try:
        async with worker(client, queue, acts):
            state = await client.execute_workflow(
                HybridWorkflow.run,
                State(str(uuid.uuid4()), ["Delegate and echo yourself."]),
                id=queue,
                task_queue=queue,
            )
        assert state.answers == ["PARENT DONE"]
        assert len(acts.tool_calls) == len(state.ledger) == 2
        assert {bool(e.call.subpath) for e in state.ledger.values()} == {False, True}
        assert all(
            e.scheduled and e.outcome and not e.outcome.is_error
            for e in state.ledger.values()
        )
        assert api.errors == []
    finally:
        api.stop()
