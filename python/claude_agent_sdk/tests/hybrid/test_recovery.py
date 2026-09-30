"""Reproduce the published engine's pending-MCP-call recovery contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from tests.helpers.fake_messages_api import history_of
from tests.hybrid.models import Call, Reply
from tests.hybrid.policy import model_requests, rounds
from tests.hybrid.test_engine import echo, make_burst

pytestmark = pytest.mark.timeout(120)


async def inject(session: str, outcomes: dict[str, Reply]) -> Any:
    yield {
        "type": "user",
        "session_id": session,
        "parent_tool_use_id": None,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": reply.text,
                    "is_error": reply.is_error,
                }
                for tid, reply in outcomes.items()
            ],
        },
    }


@pytest.mark.parametrize(
    "phase", ["before-scheduling", "after-completion", "partial-batch"]
)
@pytest.mark.parametrize("fork", [False, True], ids=["resume", "fork"])
@pytest.mark.parametrize("delivery", [False, True], ids=["prompt", "inject-results"])
async def test_pending_recovery_probe(
    tmp_path: Path, phase: str, fork: bool, delivery: bool
) -> None:
    api = rounds(1, 3 if phase == "partial-batch" else 1)
    outcomes: dict[str, Reply] = {}
    release = asyncio.Event()
    ready = asyncio.Event()
    seen: list[Call] = []

    async def stopped(call: Call) -> Reply:
        seen.append(call)
        if phase == "partial-batch" and len(seen) == 3:
            ready.set()
        if phase == "before-scheduling":
            ready.set()
            await release.wait()
        elif phase == "after-completion":
            outcomes[call.id] = await echo(call)
            ready.set()
            await release.wait()
        else:
            if call.arguments["n"] == 0:
                outcomes[call.id] = await echo(call)
                return outcomes[call.id]
            if len(seen) == 3:
                ready.set()
            await release.wait()
        return await echo(call)

    burst = make_burst(tmp_path, api, stopped)
    task: asyncio.Task[Any] | None = None
    try:
        await burst.open()
        task = asyncio.create_task(burst.query("work"))
        await asyncio.wait_for(ready.wait(), 10)
        await asyncio.sleep(0.2)  # allow the first partial-batch result to mirror
        saved = await burst.store.transcripts(burst.key)
        original_ids = set(burst.calls)
        assert len(original_ids) == (3 if phase == "partial-batch" else 1)
        transport: Any = burst.sdk._transport
        transport._process.kill()
        with pytest.raises(Exception):
            await task
        await burst.close()
        session = burst.session_id
        if fork:
            from claude_agent_sdk import fork_session_via_store

            result = await fork_session_via_store(
                cast(Any, burst.store), session, directory=str(tmp_path)
            )  # type: ignore[arg-type]
            session = result.session_id
        restored_calls: list[Call] = []

        async def restored_tool(call: Call) -> Reply:
            restored_calls.append(call)
            return await echo(call)

        restored = make_burst(
            tmp_path,
            api,
            restored_tool,
            store=burst.store,
            session=session,
            config="another-worker",
            resume=True,
        )
        try:
            await restored.open()
            # A normal prompt probes whether the engine restores the suspended callbacks.
            prompt: Any = "Continue the pending work."
            if delivery:
                prompt = inject(session, {c.id: await echo(c) for c in seen})
            response = await restored.query(prompt)
            _, _, history = history_of(model_requests(api)[-1])
            evidence = {
                "phase": phase,
                "delivery": delivery,
                "fork": fork,
                "original_ids": sorted(original_ids),
                "restored_callback_ids": [c.id for c in restored_calls],
                "outcomes_before_loss": {k: v.text for k, v in outcomes.items()},
                "observed_results": [
                    {"id": h.id, "content": h.content, "is_error": h.is_error}
                    for h in history
                ],
                "response": response.result,
                "version": burst.version,
            }
            print("HYBRID_RECOVERY " + json.dumps(evidence), flush=True)
            # This assertion deliberately pins the blocker, rather than xfail-ing recovery.
            # Resume synthesizes interrupted results and does not resurrect native callbacks.
            assert not restored_calls
            assert any(h.id in original_ids and h.is_error for h in history)
            if phase == "after-completion":
                assert not any(not h.is_error and h.id in outcomes for h in history)
            assert saved[""] and api.errors == []
        finally:
            await restored.close()
    finally:
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await burst.close()
        api.stop()
