"""Native Activity executor using verified CLI-owned single-call checkpoints.

Model continuation is refused by a local endpoint. This is a test workaround,
not a supported native execute-tool RPC or a production deployment design.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    MirrorErrorMessage,
    ResultError,
    ResultMessage,
    SessionKey,
    SessionStoreEntry,
    StreamEvent,
    TextBlock,
    ToolUseBlock,
)
from claude_agent_sdk._internal.session_resume import (
    apply_materialized_options,
    materialize_resume_session,
)
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from temporalio import activity
from temporalio.exceptions import ApplicationError
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env
from tests.hybrid.executor_models import NativeDecision, NativeIntent
from tests.hybrid.executor_store import ExecutionStore
from tests.hybrid.models import Attempt, Call, Reply
from tests.hybrid.native_store import NativeStore
from tests.hybrid.store import TranscriptStore

CONTINUATION_REFUSED = "native executor model continuation refused"


class RecordedTransport(SubprocessCLITransport):
    def __init__(
        self, options: ClaudeAgentOptions, record: Callable[[int], None]
    ) -> None:
        super().__init__(prompt="", options=options)
        self.record = record

    async def connect(self) -> None:
        await super().connect()
        assert self._process is not None
        # Log before initialize: deferred auto-execution can start before the
        # SDK installs hooks. Host cleanup must not depend on those hooks.
        self.record(self._process.pid)


@asynccontextmanager
async def heartbeating() -> AsyncIterator[None]:
    async def beat() -> None:
        while True:
            activity.heartbeat({"attempt": activity.info().attempt})
            await asyncio.sleep(0.2)

    task = asyncio.create_task(beat())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class ExecutionMirror(TranscriptStore):
    def __init__(self, path: Path, store: ExecutionStore, intent: NativeIntent) -> None:
        super().__init__(path)
        self.store, self.intent = store, intent
        self.seed = store.frozen(intent)
        self._append(store.key(intent.session_id), self.seed)
        self.published = False

    async def append(self, key: SessionKey, entries: list[SessionStoreEntry]) -> None:
        await super().append(key, entries)
        if self.published:
            return
        for entry in entries:
            message = cast(dict[str, Any], entry.get("message", {}))
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    not isinstance(block, dict)
                    or block.get("type") != "tool_result"
                    or block.get("tool_use_id") != self.intent.id
                ):
                    continue
                self.store.mark(self.intent.id, "executed")
                if self.store.phase == self.intent.name + "-after-write":
                    await self.store.held.wait()
                await asyncio.to_thread(self.store.stage, self.intent.id)
                restored = await self.load(key)
                assert restored is not None and restored[: len(self.seed)] == self.seed
                stop = next(
                    i
                    for i, e in enumerate(restored)
                    if e.get("uuid") == entry.get("uuid")
                )
                suffix = restored[len(self.seed) : stop + 1]
                await asyncio.to_thread(self.store.publish, self.intent, block, suffix)
                self.published = True


StoreType = TypeVar("StoreType", bound=NativeStore)


class NativeCLIActivities(Generic[StoreType]):
    def __init__(self, root: Path, env: dict[str, str], store: StoreType) -> None:
        self.root, self.env, self.store = root, env, store
        self.recorded: set[int] = set()

    def record_cli(self, client: ClaudeSDKClient) -> None:
        transport: Any = client._transport
        process = transport._process
        self.record_pid(process.pid)

    def record_pid(self, pid: int) -> None:
        if pid not in self.recorded:
            self.recorded.add(pid)
            with (self.root / "cli-pids.jsonl").open("a") as log:
                log.write(json.dumps({"pid": pid, "worker": os.getpid()}) + "\n")

    def attempt(self, index: int, execution: bool) -> Attempt:
        info = activity.info()
        return Attempt(
            index * 2 + int(execution),
            info.attempt,
            f"{info.workflow_run_id}:{info.activity_id}:{info.attempt}",
        )

    def options(
        self, session_id: str, index: int, store: TranscriptStore
    ) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            cwd=str(self.store.workspace),
            cli_path=os.environ.get("HYBRID_CLI_PATH"),
            env=self.env,
            tools=["Read", "Edit"],
            allowed_tools=["Read", "Edit"],
            permission_mode="acceptEdits",
            setting_sources=[],
            mcp_servers={},
            strict_mcp_config=True,
            session_id=session_id if index == 0 else None,
            resume=session_id if index else None,
            session_store=cast(Any, store),
            session_store_flush="eager",
            include_partial_messages=True,
        )


class CheckpointActivities(NativeCLIActivities[ExecutionStore]):
    async def publish_checkpoint(
        self, intent: NativeIntent, entries: list[SessionStoreEntry]
    ) -> NativeDecision:
        for position, event in (
            ("before-checkpoint-publication", "checkpoint-unpublished"),
            ("after-checkpoint-before-completion", "checkpointed"),
        ):
            if position.startswith("after"):
                await asyncio.to_thread(self.store.freeze, intent, entries)
            if self.store.phase == intent.name + "-" + position:
                with self.store.connect() as db:
                    db.execute(
                        "INSERT INTO native_events(id,phase,owner) VALUES (?,?,?)",
                        (intent.id, event, self.store.owner),
                    )
                await self.store.held.wait()
        return NativeDecision(
            intent.session_id, intent.index, intent, answer=intent.answer
        )

    @activity.defn(name="native_decision")
    async def decide(self, inp: NativeDecision) -> NativeDecision:
        async with heartbeating():
            cached = self.store.decision(inp.session_id, inp.index)
            if cached is not None:
                self.store.frozen(cached)
                return NativeDecision(
                    inp.session_id, inp.index, cached, answer=cached.answer
                )
            attempt = self.attempt(inp.index, False)
            await asyncio.to_thread(self.store.checkout, attempt)
            key = self.store.key(inp.session_id)
            mirror = TranscriptStore(
                self.root / (attempt.token.replace(":", "_") + ".db")
            )
            mirror._append(key, await self.store.load(key) or [])
            observed: dict[str, tuple[str, dict[str, Any]]] = {}
            complete = asyncio.Event()
            failure: list[str] = []
            invoked: set[str] = set()
            answer = ""

            async def defer(data: Any, tid: str | None, context: Any) -> Any:
                del context
                if tid:
                    invoked.add(tid)
                try:
                    await asyncio.wait_for(complete.wait(), 5)
                    if (
                        len(observed) != 1
                        or not tid
                        or observed.get(tid) != (data["tool_name"], data["tool_input"])
                    ):
                        raise RuntimeError(
                            "native checkpoint requires exactly one verified tool per response"
                        )
                    if (
                        Path(data["tool_input"]["file_path"]).resolve()
                        != self.store.workspace / "note.txt"
                    ):
                        raise RuntimeError(
                            "native checkpoint escaped the managed workspace"
                        )
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "defer",
                        }
                    }
                except Exception as exc:
                    failure.append(str(exc))
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": str(exc),
                        }
                    }

            options = self.options(inp.session_id, inp.index, mirror)
            options.hooks = {
                "PreToolUse": [HookMatcher(matcher="Read|Edit", hooks=[defer])]
            }
            client = ClaudeSDKClient(options=options)
            result: ResultMessage | None = None
            try:
                await client.connect()
                self.record_cli(client)
                await client.query("native" if inp.index == 0 else "")
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        if observed and all(
                            isinstance(b, TextBlock) for b in message.content
                        ):
                            answer = "".join(
                                b.text
                                for b in message.content
                                if isinstance(b, TextBlock)
                            )
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
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
            if failure or result is None or result.is_error:
                raise ApplicationError(
                    "; ".join(failure) or "native decision failed", non_retryable=True
                )
            entries = await mirror.load(key)
            assert entries is not None
            if result.deferred_tool_use is None:
                if observed:
                    if len(observed) != 1:
                        raise ApplicationError(
                            "native validation requires one verified call",
                            non_retryable=True,
                        )
                    tid, (name, arguments) = next(iter(observed.items()))
                    errors = self.store._native_results(entries, tid)
                    if (
                        tid in invoked
                        or name not in {"Read", "Edit"}
                        or Path(arguments["file_path"]).resolve()
                        != self.store.workspace / "note.txt"
                        or len(errors) != 1
                        or not errors[0].get("is_error")
                    ):
                        raise ApplicationError(
                            "native call completed without a verified validation outcome",
                            non_retryable=True,
                        )
                    uid, subpath = await mirror.wait_call(key, tid, name, arguments, 5)
                    assert not subpath
                    head = next(
                        str(e.get("uuid")) for e in reversed(entries) if e.get("uuid")
                    )
                    intent = NativeIntent(
                        tid,
                        name,
                        arguments,
                        inp.session_id,
                        inp.index,
                        self.store.version(),
                        uid,
                        head,
                        outcome_kind="validation",
                        answer=answer,
                    )
                    return await self.publish_checkpoint(intent, entries)
                await asyncio.to_thread(
                    self.store.publish_decision, inp.session_id, entries
                )
                return NativeDecision(
                    inp.session_id, inp.index, answer=result.result or ""
                )
            deferred = result.deferred_tool_use
            if (
                deferred.id not in observed
                or len(observed) != 1
                or observed[deferred.id] != (deferred.name, deferred.input)
            ):
                raise ApplicationError(
                    "native deferred identity differs", non_retryable=True
                )
            name, arguments = observed[deferred.id]
            markers = [
                e
                for e in entries
                if cast(dict[str, Any], e.get("attachment", {})).get("type")
                == "hook_deferred_tool"
                and cast(dict[str, Any], e.get("attachment", {})).get("toolUseID")
                == deferred.id
            ]
            if not markers:
                raise ApplicationError(
                    "engine checkpoint has not reached storage", non_retryable=True
                )
            uid, subpath = await mirror.wait_call(key, deferred.id, name, arguments, 5)
            assert not subpath
            head = next(str(e.get("uuid")) for e in reversed(entries) if e.get("uuid"))
            intent = NativeIntent(
                deferred.id,
                name,
                arguments,
                inp.session_id,
                inp.index,
                self.store.version(),
                uid,
                head,
            )
            return await self.publish_checkpoint(intent, entries)

    @activity.defn(name="native_execution")
    async def execute(self, intent: NativeIntent) -> Reply:
        async with heartbeating():
            self.store.frozen(intent)
            attempt = self.attempt(intent.index, True)
            await asyncio.to_thread(self.store.checkout, attempt)
            cached = self.store.execution(intent.id)
            if cached and cached["result"] is not None:
                if self.store.phase == intent.name + "-after-commit-before-completion":
                    await self.store.held.wait()
                return Reply(
                    json.dumps(cached["result"], sort_keys=True),
                    bool(cached["result"].get("is_error")),
                )
            call = Call(
                intent.id,
                intent.name,
                intent.arguments,
                attempt,
                intent.transcript_uuid,
            )
            await asyncio.to_thread(self.store.claim, call, intent)
            if self.store.phase == intent.name + "-before-execution":
                self.store.mark(intent.id, "held-before-execution")
                await self.store.held.wait()
            mirror = ExecutionMirror(
                self.root / (attempt.token.replace(":", "_") + ".db"),
                self.store,
                intent,
            )
            denied = FakeMessagesAPI(
                lambda _: [], primary_tools={"Read", "Edit"}
            ).start()
            denied.fail_status = 400
            denied.fail_message = CONTINUATION_REFUSED
            options = self.options(intent.session_id, intent.index + 1, mirror)
            options.env = engine_env(denied, str(self.root / "native-executor-config"))
            options.max_turns = 1
            client: ClaudeSDKClient | None = None

            async def allow(data: Any, tid: str | None, context: Any) -> Any:
                del context
                assert client is not None
                self.record_cli(client)
                arguments = dict(data["tool_input"])
                if (
                    intent.name == "Edit"
                    and "replace_all" not in intent.arguments
                    and arguments.get("replace_all") is False
                ):
                    arguments.pop("replace_all")
                if (
                    tid != intent.id
                    or data["tool_name"] != intent.name
                    or arguments != intent.arguments
                ):
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": "native checkpoint identity differs",
                        }
                    }
                self.store.mark(intent.id, "permitted")
                return {}

            options.hooks = {
                "PreToolUse": [
                    HookMatcher(matcher="Read|Edit", hooks=[allow], timeout=120)
                ]
            }
            materialized = None
            try:
                materialized = await materialize_resume_session(options)
                assert materialized is not None
                configured = apply_materialized_options(options, materialized)
                client = ClaudeSDKClient(
                    options=configured,
                    transport=RecordedTransport(configured, self.record_pid),
                )
                try:
                    await client.connect()
                    self.record_cli(client)
                    async for message in client.receive_response():
                        if isinstance(message, MirrorErrorMessage):
                            raise RuntimeError(message.error)
                        if isinstance(message, ResultMessage) and (
                            not message.is_error or message.api_error_status != 400
                        ):
                            raise RuntimeError(
                                "native executor escaped the continuation guard"
                            )
                except ResultError as exc:
                    if CONTINUATION_REFUSED not in str(exc):
                        raise
            finally:
                try:
                    if client is not None:
                        await client.disconnect()
                finally:
                    try:
                        if materialized is not None:
                            await materialized.cleanup()
                    finally:
                        denied.stop()
            with self.store.connect() as db:
                db.executemany(
                    "INSERT INTO native_events(id,phase,owner) VALUES (?,'continuation-refused',?)",
                    [
                        (intent.id, self.store.owner)
                        for body in denied.requests
                        if any(
                            t.get("name") in {"Read", "Edit"}
                            for t in body.get("tools", [])
                        )
                    ],
                )
            row = self.store.execution(intent.id)
            if not mirror.published or row is None or row["result"] is None:
                raise ApplicationError(
                    "native executor has no committed original result",
                    non_retryable=True,
                )
            if self.store.phase == intent.name + "-after-commit-before-completion":
                await self.store.held.wait()
            return Reply(
                json.dumps(row["result"], sort_keys=True),
                bool(row["result"].get("is_error")),
            )
