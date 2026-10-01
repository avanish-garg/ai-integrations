"""Exact recorded native calls executed in bounded, isolated CLI turns.

The local response cache replays an accepted assistant block; it never asks a
provider to regenerate a call. Only real native results reach the conversation.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    HookMatcher,
    MirrorErrorMessage,
    ResultMessage,
    SessionStoreEntry,
    StreamEvent,
    ToolUseBlock,
)

from temporalio import activity
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env
from tests.hybrid.models import Reply
from tests.hybrid.native_executor import NativeCLIActivities, heartbeating
from tests.hybrid.replay_models import ReplayCall, ReplayDecision
from tests.hybrid.replay_store import ReplayStore
from tests.hybrid.store import TranscriptStore


def result_entries(
    entries: list[SessionStoreEntry], tid: str
) -> list[SessionStoreEntry]:
    return [
        e
        for e in entries
        if any(
            isinstance(b, dict)
            and b.get("type") == "tool_result"
            and b.get("tool_use_id") == tid
            for b in cast(dict[str, Any], e.get("message", {})).get("content", [])
        )
    ]


class ReplayActivities(NativeCLIActivities[ReplayStore]):
    @activity.defn(name="native_replay_decision")
    async def decide(self, inp: ReplayDecision) -> ReplayDecision:
        async with heartbeating():
            cached = self.store.decision(inp.session_id, inp.round)
            if cached is not None:
                for call in cached:
                    self.store.receipt(call)
                return ReplayDecision(inp.session_id, inp.round, cached)
            attempt = self.attempt(inp.round, False)
            await asyncio.to_thread(self.store.checkout, attempt)
            key = self.store.key(inp.session_id)
            mirror = TranscriptStore(
                self.root / (attempt.token.replace(":", "_") + ".db")
            )
            mirror._append(key, await self.store.load(key) or [])
            observed: dict[str, tuple[str, dict[str, Any]]] = {}
            invoked: set[str] = set()
            complete = asyncio.Event()
            failures: list[str] = []

            async def prepare(data: Any, tid: str | None, context: Any) -> Any:
                del context
                try:
                    if (
                        self.store.phase
                        == data["tool_name"] + "-preparation-hook-failure"
                    ):
                        raise OSError("injected native preparation hook failure")
                    await asyncio.wait_for(complete.wait(), 5)
                    if (
                        not tid
                        or observed.get(tid) != (data["tool_name"], data["tool_input"])
                        or Path(data["tool_input"]["file_path"]).resolve()
                        != self.store.workspace / "note.txt"
                    ):
                        raise RuntimeError("native preparation identity differs")
                    invoked.add(tid)
                except Exception as exc:
                    failures.append(type(exc).__name__ + ": " + str(exc))
                # Preparation runs no accepted tool. Denial artifacts remain
                # private to this attempt and never enter the canonical store.
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "native preparation",
                    }
                }

            options = self.options(inp.session_id, inp.round, mirror)
            options.max_turns = 1
            options.hooks = {
                "PreToolUse": [HookMatcher(matcher="Read|Edit", hooks=[prepare])]
            }
            client = ClaudeSDKClient(options=options)
            result: ResultMessage | None = None
            try:
                await client.connect()
                self.record_cli(client)
                await client.query("native" if inp.round == 0 else "")
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                if not block.id:
                                    failures.append("missing original native call ID")
                                if block.id in observed:
                                    failures.append("duplicate original native call ID")
                                observed[block.id] = (block.name, dict(block.input))
                    elif (
                        isinstance(message, StreamEvent)
                        and message.event.get("type") == "message_stop"
                    ):
                        complete.set()
                    elif isinstance(message, MirrorErrorMessage):
                        raise RuntimeError(message.error)
                    elif isinstance(message, ResultMessage):
                        result = message
            finally:
                await client.disconnect()
            if (
                failures
                or result is None
                or (result.is_error and result.subtype != "error_max_turns")
            ):
                raise RuntimeError("; ".join(failures) or "native preparation failed")
            entries = await mirror.load(key)
            assert entries is not None
            if not observed:
                await asyncio.to_thread(self.store.terminal, inp.session_id, entries)
                return ReplayDecision(
                    inp.session_id, inp.round, answer=result.result or ""
                )
            if result.subtype != "error_max_turns":
                raise RuntimeError("native preparation continued beyond one response")
            calls: list[ReplayCall] = []
            # Preserve the real whole-batch assistant transcript, stopping
            # before this preparation's denial/validation artifacts.
            stop = next(
                i
                for i, e in enumerate(entries)
                if any(
                    isinstance(b, dict)
                    and b.get("type") == "tool_result"
                    and b.get("tool_use_id") in observed
                    for b in cast(dict[str, Any], e.get("message", {})).get(
                        "content", []
                    )
                )
            )
            source = entries[:stop]
            digest = hashlib.sha256(
                json.dumps(source, sort_keys=True).encode()
            ).hexdigest()
            for tid, (name, arguments) in observed.items():
                if (
                    name not in {"Read", "Edit"}
                    or Path(arguments["file_path"]).resolve()
                    != self.store.workspace / "note.txt"
                ):
                    raise RuntimeError(
                        "native preparation escaped its managed tool set"
                    )
                if tid not in invoked:
                    errors = result_entries(entries, tid)
                    if len(errors) != 1 or not any(
                        b.get("is_error")
                        for b in cast(dict[str, Any], errors[0].get("message", {}))[
                            "content"
                        ]
                    ):
                        raise RuntimeError(
                            "native preparation performed an unguarded effect"
                        )
                uid, subpath = await mirror.wait_call(key, tid, name, arguments, 5)
                assert not subpath and any(e.get("uuid") == uid for e in source)
                calls.append(
                    ReplayCall(
                        tid,
                        name,
                        arguments,
                        inp.session_id,
                        inp.round,
                        len(calls),
                        attempt.token,
                        digest,
                        uid,
                    )
                )
            if self.store.phase == calls[0].name + "-before-checkpoint-publication":
                with self.store.connect() as db:
                    db.execute(
                        "INSERT INTO native_events(id,phase,owner) VALUES (?,'checkpoint-unpublished',?)",
                        (calls[0].id, attempt.token),
                    )
                await self.store.held.wait()
            await asyncio.to_thread(self.store.freeze, calls, source)
            if (
                self.store.phase
                == calls[0].name + "-after-checkpoint-before-completion"
            ):
                with self.store.connect() as db:
                    db.execute(
                        "INSERT INTO native_events(id,phase,owner) VALUES (?,'checkpointed',?)",
                        (calls[0].id, attempt.token),
                    )
                await self.store.held.wait()
            return ReplayDecision(inp.session_id, inp.round, calls)

    @activity.defn(name="native_replay_execution")
    async def execute(self, call: ReplayCall) -> Reply:
        async with heartbeating():
            attempt = self.attempt(call.round, True)
            actor = attempt.token
            receipt = self.store.receipt(call)
            cached = await asyncio.to_thread(
                self.store.claim, call, actor, attempt.number
            )
            if cached is not None:
                return Reply(
                    json.dumps(cached, sort_keys=True), bool(cached.get("is_error"))
                )
            if self.store.phase == call.name + "-before-execution" or (
                self.store.phase in {"partial-batch", "partial-edits"}
                and call.position > 0
            ):
                self.store.event(call, actor, "held-before-execution")
                await self.store.held.wait()
            mirror = TranscriptStore(self.root / (actor.replace(":", "_") + ".db"))
            key = self.store.key(call.session_id)
            mirror._append(key, await self.store.context(call))
            response_cache = FakeMessagesAPI(
                lambda _: [copy.deepcopy(receipt["block"])],
                primary_tools={"Read", "Edit"},
            ).start()
            options = self.options(call.session_id, call.round + 1, mirror)
            options.max_turns = 1
            options.env = engine_env(
                response_cache, str(self.root / ("cache-" + actor.replace(":", "_")))
            )
            failures: list[str] = []

            async def allow(data: Any, tid: str | None, context: Any) -> Any:
                del context
                arguments = dict(data["tool_input"])
                if (
                    call.name == "Edit"
                    and "replace_all" not in call.arguments
                    and arguments.get("replace_all") is False
                ):
                    arguments.pop("replace_all")
                if (tid, data["tool_name"], arguments) != (
                    call.id,
                    call.name,
                    call.arguments,
                ):
                    failures.append("native cached call identity differs")
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                        }
                    }
                self.store.event(call, actor, "permitted")
                if self.store.phase == "parallel-proof":
                    while True:
                        with self.store.connect() as db:
                            permitted = db.execute(
                                "SELECT count(DISTINCT id) FROM native_events WHERE phase='permitted'"
                            ).fetchone()[0]
                        if permitted == len(receipt["calls"]):
                            break
                        await asyncio.sleep(0.01)
                return {}

            options.hooks = {
                "PreToolUse": [HookMatcher(matcher="Read|Edit", hooks=[allow])]
            }
            client = ClaudeSDKClient(options=options)
            result: ResultMessage | None = None
            observed: list[str] = []
            try:
                await client.connect()
                self.record_cli(client)
                await client.query("")
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                if (block.id, block.name, block.input) != (
                                    call.id,
                                    call.name,
                                    call.arguments,
                                ):
                                    failures.append(
                                        "native cached assistant identity differs"
                                    )
                                observed.append(block.id)
                    elif isinstance(message, MirrorErrorMessage):
                        raise RuntimeError(message.error)
                    elif isinstance(message, ResultMessage):
                        result = message
            finally:
                try:
                    await client.disconnect()
                finally:
                    response_cache.stop()
            if (
                failures
                or observed != [call.id]
                or result is None
                or result.subtype != "error_max_turns"
                or len(
                    [
                        b
                        for b in response_cache.requests
                        if any(
                            t["name"] in {"Read", "Edit"} for t in b.get("tools", [])
                        )
                    ]
                )
                != 1
                or response_cache.errors
            ):
                raise RuntimeError(
                    "; ".join(failures)
                    or "native cached execution escaped its bounded turn"
                )
            entries = await mirror.load(key)
            assert entries is not None
            outcomes = result_entries(entries, call.id)
            if len(outcomes) != 1:
                raise RuntimeError("native cached execution lost its real outcome")
            self.store.event(call, actor, "cached-response")
            self.store.event(call, actor, "executed")
            if self.store.phase == call.name + "-after-write":
                await self.store.held.wait()
            result_block = await asyncio.to_thread(
                self.store.publish, call, actor, outcomes[0]
            )
            if self.store.phase == call.name + "-after-commit-before-completion":
                await self.store.held.wait()
            return Reply(
                json.dumps(result_block, sort_keys=True),
                bool(result_block.get("is_error")),
            )
