"""Real bundled CLI capabilities against the existing deterministic Messages API."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.claude_agent_sdk import (
    ClaudeAgentSdkRunner,
    SegmentInput,
    ToolOutcome,
    ToolSpec,
)
from tests.helpers.fake_messages_api import engine_env, history_of
from tests.hybrid.engine import SCHEMA, Burst, PrototypeBlocked, native_id
from tests.hybrid.models import Attempt, Call, Reply
from tests.hybrid.policy import chat, model_requests, rounds
from tests.hybrid.store import TranscriptStore

pytestmark = pytest.mark.timeout(120)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def echo(call: Call) -> Reply:
    return Reply(json.dumps({"n": call.arguments["n"]}))


def make_burst(tmp_path: Path, api: Any, execute: Any = echo, **kwargs: Any) -> Burst:
    store = kwargs.pop("store", TranscriptStore(tmp_path / "store.db"))
    session = kwargs.pop("session", str(uuid.uuid4()))
    config = kwargs.pop("config", "cfg")
    return Burst(
        tmp_path,
        engine_env(api, str(tmp_path / config)),
        store,
        session,
        Attempt(0, 1, "probe"),
        execute,
        **kwargs,
    )


@pytest.mark.parametrize(
    "meta", [None, {}, {"progressToken": "not-an-ID"}, {"claudecode/toolUseId": ""}]
)
def test_missing_native_id_is_rejected(meta: Any) -> None:
    with pytest.raises(PrototypeBlocked, match="missing native"):
        native_id(meta)


async def test_reuse_and_completed_task_suspension(tmp_path: Path) -> None:
    api = chat()
    burst = make_burst(tmp_path, api)
    try:
        await burst.open()
        pid = burst.pid
        for n in range(3):
            result = await burst.query(f"round {n}")
            assert result.result == f"DONE {n}; remembered {n + 1}"
            assert burst.pid == pid and alive(pid)
        uid, delivered = await burst.checkpoint()
        assert uid and len(delivered) == 3
        await burst.close()
        assert not alive(pid)
        restored = make_burst(
            tmp_path,
            api,
            store=burst.store,
            session=burst.session_id,
            config="fresh-machine",
            resume=True,
        )
        try:
            await restored.open()
            result = await restored.query("round 3")
            assert restored.pid != pid
            assert result.result == "DONE 3; remembered 4"
        finally:
            await restored.close()
        assert api.errors == []
    finally:
        await burst.close()
        api.stop()


async def test_native_batch_overlaps_without_rediscovery(tmp_path: Path) -> None:
    api = rounds(1, 3)
    entered: set[str] = set()
    release = asyncio.Event()

    async def parallel(call: Call) -> Reply:
        entered.add(call.id)
        if len(entered) == 3:
            release.set()
        await asyncio.wait_for(release.wait(), 10)
        return await echo(call)

    burst = make_burst(tmp_path, api, parallel)
    try:
        await burst.open()
        assert (await burst.query("three tools")).result == "DONE 3"
        assert len(entered) == 3
        requests = model_requests(api)
        assert len(requests) == 2
        uses, _, history = history_of(requests[-1])
        assert set(uses) == entered == {h.id for h in history}
        assert all(not h.is_error for h in history)
        listed = await burst.list_tools(None, None)
        assert listed.tools[0].model_dump(by_alias=True)["inputSchema"] == SCHEMA
        assert api.errors == []
    finally:
        await burst.close()
        api.stop()


async def test_pending_approval_preserves_process_and_callback(tmp_path: Path) -> None:
    api = rounds(1, approval=True)
    decision: asyncio.Future[Reply] = asyncio.get_running_loop().create_future()

    async def approve(call: Call) -> Reply:
        assert call.arguments["approval"]
        return await decision

    burst = make_burst(tmp_path, api, approve)
    task: asyncio.Task[Any] | None = None
    try:
        await burst.open()
        task = asyncio.create_task(burst.query("wait for approval"))
        await asyncio.wait_for(burst.callback_started.wait(), 10)
        assert not await burst.suspend_pending()
        assert alive(burst.pid) and not task.done()
        decision.set_result(Reply("approved"))
        assert (await task).result == "DONE 1"
        assert api.errors == []
    finally:
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await burst.close()
        api.stop()


@pytest.mark.parametrize("fail", [False, True], ids=["delayed-write", "failed-write"])
async def test_no_tool_before_transcript_storage(tmp_path: Path, fail: bool) -> None:
    api = rounds(1)
    store = TranscriptStore(tmp_path / "store.db")
    store.delay = 0.5 if not fail else 0
    store.fail = fail
    executed = asyncio.Event()

    async def tracked(call: Call) -> Reply:
        await store.wait_call(
            burst.key, call.id, "mcp__durable__echo", call.arguments, 1
        )
        executed.set()
        return await echo(call)

    burst = make_burst(tmp_path, api, tracked, store=store, storage_timeout=2)
    try:
        await burst.open()
        task = asyncio.create_task(burst.query("echo"))
        await asyncio.wait_for(store.writing.wait(), 10)
        if fail:
            with pytest.raises((RuntimeError, PrototypeBlocked)):
                await task
            assert not executed.is_set()
        else:
            assert not executed.is_set()
            assert (await task).result == "DONE 1"
            assert executed.is_set()
    finally:
        await burst.close()
        api.stop()


async def test_existing_single_call_approval_deferral(tmp_path: Path) -> None:
    api = rounds(1)
    runner = ClaudeAgentSdkRunner(
        session_store=TranscriptStore(tmp_path / "store.db"),
        cwd=str(tmp_path),
        env=engine_env(api, str(tmp_path / "cfg")),
    )  # type: ignore[arg-type]
    try:
        first = await runner.run(
            SegmentInput(
                session_id=str(uuid.uuid4()),
                prompt="echo",
                tools=[ToolSpec("echo", "Echo.", SCHEMA)],
            ),
            1,
        )
        assert not first.is_error and first.deferred is not None and first.checkpoint
        second = await runner.run(
            SegmentInput(
                session_id=first.session_id,
                prompt=None,
                checkpoint=first.checkpoint,
                tools=[ToolSpec("echo", "Echo.", SCHEMA)],
                injected={first.deferred.id: ToolOutcome({"n": 0})},
            ),
            1,
        )
        assert second.result == "DONE 1" and not second.is_error
        assert api.errors == []
    finally:
        api.stop()


async def test_inconsistent_native_id_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    api = rounds(1)
    burst = make_burst(tmp_path, api)
    try:
        burst.observed["toolu_real"] = ("mcp__durable__echo", {"n": 1})
        result = await burst.call_tool(
            SimpleNamespace(meta={"claudecode/toolUseId": "toolu_real"}),
            SimpleNamespace(name="echo", arguments={"n": 999}),
        )
        assert result.is_error and not burst.calls
        assert burst.failure and "disagrees" in str(burst.failure)
    finally:
        api.stop()
