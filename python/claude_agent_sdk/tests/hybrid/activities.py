"""Long-running CLI Activity and a separately scheduled, idempotent test tool."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from temporalio import activity
from temporalio.client import Client
from temporalio.exceptions import ApplicationError
from tests.hybrid.engine import Burst, PrototypeBlocked
from tests.hybrid.models import (
    Attempt,
    BurstInput,
    Call,
    Checkpoint,
    Reply,
    TurnCheckpoint,
)
from tests.hybrid.store import TranscriptStore
from tests.hybrid.workflows import HybridWorkflow


class HybridActivities:
    def __init__(
        self, client: Client, root: Path, env: dict[str, str], store: TranscriptStore
    ) -> None:
        self.client, self.root, self.env, self.store = client, root, env, store
        self.bursts: list[Burst] = []
        self.subagents = False
        self.recovery = False
        self.started: dict[str, float] = {}
        self.finished: dict[str, float] = {}
        self.tool_calls: list[str] = []
        self.fail_after_effect = False
        self.after_checkpoint: asyncio.Event | None = None
        self.before_request: asyncio.Event | None = None
        self.before_delivery: asyncio.Event | None = None

    @activity.defn(name="hybrid_tool")
    async def tool(self, call: Call) -> Reply:
        self.tool_calls.append(call.id)
        self.started[call.id] = asyncio.get_running_loop().time()
        await asyncio.sleep(float(call.arguments.get("delay", 0)))
        # Stable across CLI retries, Workflow replay and Continue-As-New.
        workflow_id = activity.info().workflow_id
        assert workflow_id is not None
        key = workflow_id + ":tool-" + call.id
        text = await asyncio.to_thread(
            self.store.effect, key, {"n": call.arguments["n"]}
        )
        if self.fail_after_effect:
            self.fail_after_effect = False
            raise RuntimeError("injected failure after external effect")
        self.finished[call.id] = asyncio.get_running_loop().time()
        return Reply(text)

    @activity.defn(name="hybrid_burst")
    async def burst(self, inp: BurstInput) -> list[str]:
        async def beat() -> None:
            while True:
                activity.heartbeat({"attempt": activity.info().attempt})
                await asyncio.sleep(0.2)

        heartbeat = asyncio.create_task(beat())
        try:
            return await self.run_burst(inp)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def run_burst(self, inp: BurstInput) -> list[str]:
        info = activity.info()
        assert info.workflow_id is not None
        handle = self.client.get_workflow_handle(
            info.workflow_id, run_id=info.workflow_run_id
        )
        attempt = Attempt(
            inp.burst,
            info.attempt,
            f"{info.workflow_run_id}:{info.activity_id}:{info.attempt}",
        )
        snap = await handle.execute_update(HybridWorkflow.register, attempt)
        if snap.checkpoints and snap.checkpoints[-1].attempt.burst == inp.burst:
            return snap.checkpoints[-1].answers
        # Completed-turn recovery can resume. Pending-call recovery has no proven protocol.
        if not self.recovery and any(
            e.call.attempt.burst == inp.burst for e in snap.ledger.values()
        ):
            raise ApplicationError(
                "blocked: Activity lost with uncheckpointed native MCP calls; "
                "published engine cannot yet be trusted to restore pending identities",
                non_retryable=True,
            )

        async def execute(call: Call) -> Reply:
            return await handle.execute_update(HybridWorkflow.request, call)

        resume = inp.checkpoint is not None
        if self.recovery and info.attempt > 1:
            # Storage may contain a requested call even if the old Worker died
            # before its request Update reached Temporal.
            # A missing recovery transcript must fail rather than silently
            # starting another model turn with replacement tool IDs.
            resume = True

        burst = Burst(
            self.root,
            self.env,
            self.store,
            inp.session_id,
            attempt,
            execute,
            resume=resume,
            subagents=self.subagents,
            recovery=self.recovery,
            recovery_entries={
                tid: entry
                for tid, entry in snap.ledger.items()
                if entry.call.attempt.burst == inp.burst
            },
        )
        burst.before_request = self.before_request
        burst.before_delivery = self.before_delivery
        self.bursts.append(burst)

        async def run() -> list[str]:
            await burst.open()
            with (self.root / "cli-pids.jsonl").open("a") as log:
                log.write(json.dumps({"pid": burst.pid, "worker": os.getpid()}) + "\n")
            answers = []
            for offset, prompt in enumerate(inp.prompts):
                index = inp.burst + offset
                completed = snap.turns.get(index)
                if completed is not None:
                    answers.append(completed.answer)
                    continue
                answer = (await burst.query(prompt)).result or ""
                uid, _ = await burst.checkpoint()
                await handle.execute_update(
                    HybridWorkflow.finish_turn,
                    TurnCheckpoint(attempt, index, uid, answer),
                )
                answers.append(answer)
            uid, delivered = await burst.checkpoint()
            await handle.execute_update(
                HybridWorkflow.acknowledge,
                Checkpoint(attempt, inp.session_id, uid, delivered, burst.pid, answers),
            )
            if self.after_checkpoint is not None:
                await self.after_checkpoint.wait()
            return answers

        try:
            return await run()
        except PrototypeBlocked as exc:
            raise ApplicationError(str(exc), non_retryable=True) from exc
        finally:
            await burst.close()
