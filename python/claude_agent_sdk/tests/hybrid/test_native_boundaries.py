"""Probe startup hooks at native execution boundaries in the real CLI."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    SessionKey,
    ToolUseBlock,
    project_key_for_directory,
)

from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env
from tests.hybrid.models import Attempt
from tests.hybrid.native_store import NativeStore
from tests.hybrid.store import TranscriptStore
from tests.hybrid.test_native_workspace import native_api, native_requests


@pytest.mark.parametrize("event", ["PostToolUse", "PostToolBatch"])
async def test_startup_stop_hook_does_not_bound_deferred_resume(
    tmp_path: Path, event: str
) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    store.checkout(Attempt(0, 1, "probe"))
    api, _ = native_api(tmp_path)
    denied = FakeMessagesAPI(lambda _: [], primary_tools={"Read", "Edit"}).start()
    denied.fail_status = 400
    session = str(uuid.uuid4())

    async def defer(data: Any, tid: str | None, context: Any) -> Any:
        del data, tid, context
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "defer",
            }
        }

    options = ClaudeAgentOptions(
        cwd=str(store.workspace),
        cli_path=os.environ.get("HYBRID_CLI_PATH"),
        tools=["Read", "Edit"],
        allowed_tools=["Read", "Edit"],
        permission_mode="acceptEdits",
        setting_sources=[],
        env=engine_env(api, str(tmp_path / "cfg")),
        session_id=session,
        session_store=cast(Any, store),
        session_store_flush="eager",
        hooks={"PreToolUse": [HookMatcher(matcher="Read", hooks=[defer])]},
    )
    client = ClaudeSDKClient(options=options)
    try:
        await client.connect()
        await client.query("read")
        result = None
        async for message in client.receive_response():
            if isinstance(message, ResultMessage):
                result = message
        assert result is not None and result.deferred_tool_use is not None
        await client.disconnect()
        script = tmp_path / "stop.py"
        log = tmp_path / "hooks.jsonl"
        script.write_text(
            "import json,sys\n"
            "data=json.load(sys.stdin)\n"
            "with open(sys.argv[1],'a') as f: f.write(json.dumps(data)+'\\n')\n"
            "print(json.dumps({'continue':False,'stopReason':'native checkpoint boundary'}))\n"
        )
        options.hooks = None
        options.session_id = None
        options.resume = session
        options.env = engine_env(denied, str(tmp_path / "fresh"))
        options.settings = json.dumps(
            {
                "hooks": {
                    event: [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": shlex.join(
                                        [sys.executable, str(script), str(log)]
                                    ),
                                }
                            ]
                        }
                    ]
                }
            }
        )
        client = ClaudeSDKClient(options=options)
        await asyncio.wait_for(client.connect(), 10)
        result = None
        async for message in client.receive_response():
            if isinstance(message, ResultMessage):
                result = message
        assert log.exists() == (event == "PostToolUse")
        assert native_requests(denied)
        assert result is not None and result.is_error
    finally:
        await client.disconnect()
        api.stop()
        denied.stop()


async def test_cached_native_call_turn_is_bounded(tmp_path: Path) -> None:
    store = NativeStore(tmp_path)
    store.initialize("BEFORE\n")
    store.checkout(Attempt(0, 1, "probe"))
    api, _ = native_api(tmp_path)
    session = str(uuid.uuid4())
    observed: dict[str, Any] = {}

    async def defer(data: Any, tid: str | None, context: Any) -> Any:
        del data, tid, context
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "defer",
            }
        }

    options = ClaudeAgentOptions(
        cwd=str(store.workspace),
        cli_path=os.environ.get("HYBRID_CLI_PATH"),
        tools=["Read", "Edit"],
        allowed_tools=["Read", "Edit"],
        permission_mode="acceptEdits",
        setting_sources=[],
        env=engine_env(api, str(tmp_path / "cfg")),
        session_id=session,
        session_store=cast(Any, store),
        session_store_flush="eager",
        hooks={"PreToolUse": [HookMatcher(matcher="Read", hooks=[defer])]},
    )
    client = ClaudeSDKClient(options=options)
    replay = None
    try:
        await client.connect()
        await client.query("read")
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        observed.update(
                            type="tool_use",
                            id=block.id,
                            name=block.name,
                            input=block.input,
                        )
        await client.disconnect()
        key: SessionKey = {
            "session_id": session,
            "project_key": project_key_for_directory(str(store.workspace)),
        }
        entries = await store.load(key)
        assert entries is not None
        before = next(
            i
            for i, e in enumerate(entries)
            if any(
                isinstance(b, dict) and b.get("id") == observed["id"]
                for b in cast(dict[str, Any], e.get("message", {})).get("content", [])
            )
        )
        mirror = TranscriptStore(tmp_path / "replay.db")
        mirror._append(key, entries[:before])
        replay = FakeMessagesAPI(
            lambda _: [observed], primary_tools={"Read", "Edit"}
        ).start()
        options.hooks = None
        options.session_id = None
        options.resume = session
        options.env = engine_env(replay, str(tmp_path / "replay-cfg"))
        options.session_store = cast(Any, mirror)
        options.max_turns = 1
        client = ClaudeSDKClient(options=options)
        await asyncio.wait_for(client.connect(), 10)
        await client.query("")
        result = None
        async for message in client.receive_response():
            if isinstance(message, ResultMessage):
                result = message
        await client.disconnect()
        assert len(native_requests(replay)) == 1
        assert result is not None and result.subtype == "error_max_turns"
        copied = await mirror.load(key)
        assert copied is not None
        blocks = [
            b
            for e in copied
            for b in cast(dict[str, Any], e.get("message", {})).get("content", [])
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        assert len(blocks) == 1 and blocks[0]["tool_use_id"] == observed["id"]
        assert not blocks[0].get("is_error")
    finally:
        await client.disconnect()
        api.stop()
        if replay is not None:
            replay.stop()
