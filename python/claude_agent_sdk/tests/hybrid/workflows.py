"""Test Workflow: approvals, attempt fencing, deduplication and completed checkpoints."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from tests.hybrid.models import (
        Attempt,
        BurstInput,
        Call,
        Checkpoint,
        Entry,
        Reply,
        Snapshot,
        State,
        TurnCheckpoint,
    )


@workflow.defn
class HybridWorkflow:
    def __init__(self) -> None:
        self.state = State("", [])
        self.attempts: dict[int, Attempt] = {}
        self.handles: dict[str, workflow.ActivityHandle[Reply]] = {}
        self.closing = False

    @workflow.run
    async def run(self, state: State) -> State:
        try:
            return await self.drive(state)
        except GeneratorExit:
            # Coroutine disposal can run outside the Workflow runtime; it
            # cannot issue commands or await handler completion there.
            raise
        except BaseException:
            self.closing = True
            await workflow.wait_condition(workflow.all_handlers_finished)
            raise

    async def drive(self, state: State) -> State:
        self.state = state
        size = state.burst_size or len(state.prompts)
        while state.index < len(state.prompts):
            inp = BurstInput(
                state.session_id,
                state.prompts[state.index : state.index + size],
                state.index,
                state.checkpoint,
            )
            answers = await workflow.execute_activity(
                "hybrid_burst",
                inp,
                result_type=list[str],
                start_to_close_timeout=timedelta(minutes=5),
                heartbeat_timeout=timedelta(seconds=3),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1), maximum_attempts=2
                ),
                cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            )
            state.answers.extend(answers)
            state.index += len(inp.prompts)
            await workflow.wait_condition(workflow.all_handlers_finished)
            if state.continue_every and state.index < len(state.prompts):
                if len(state.checkpoints) % state.continue_every == 0:
                    self.closing = True
                    workflow.continue_as_new(state)
        return state

    def check_attempt(self, attempt: Attempt) -> None:
        if self.closing or self.attempts.get(attempt.burst) != attempt:
            raise ApplicationError(
                "superseded CLI Activity attempt", non_retryable=True
            )

    @workflow.update
    async def register(self, attempt: Attempt) -> Snapshot:
        previous = self.attempts.get(attempt.burst)
        if previous and previous != attempt and attempt.number <= previous.number:
            raise ApplicationError("stale CLI registration", non_retryable=True)
        if self.closing:
            raise ApplicationError("checkpoint is closing", non_retryable=True)
        self.attempts[attempt.burst] = attempt
        return self.snapshot()

    @workflow.update
    async def request(self, call: Call) -> Reply:
        self.check_attempt(call.attempt)
        if not call.transcript_uuid:
            raise ApplicationError(
                "missing transcript storage proof", non_retryable=True
            )
        entry = self.state.ledger.get(call.id)
        if entry is not None and entry.call.identity() != call.identity():
            raise ApplicationError("conflicting native call ID", non_retryable=True)
        if entry is None:
            entry = Entry(call)
            self.state.ledger[call.id] = entry
        if entry.outcome is not None:
            return entry.outcome
        if call.arguments.get("approval"):
            await workflow.wait_condition(
                lambda: (
                    entry.approved is not None
                    or self.closing
                    or self.attempts.get(call.attempt.burst) != call.attempt
                )
            )
            self.check_attempt(call.attempt)
            if not entry.approved:
                entry.outcome = Reply("rejected by reviewer", True)
                return entry.outcome
        if call.id not in self.handles:
            entry.scheduled = True
            self.handles[call.id] = workflow.start_activity(
                "hybrid_tool",
                call,
                result_type=Reply,
                activity_id="tool-" + call.id,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
        # Duplicate handlers share the same Activity handle and committed outcome.
        try:
            entry.outcome = await asyncio.shield(self.handles[call.id])
        except Exception as exc:
            entry.outcome = Reply(str(exc), True)
        assert entry.outcome is not None
        return entry.outcome

    @workflow.update
    async def review(self, decision: tuple[str, bool]) -> None:
        tid, approved = decision
        entry = self.state.ledger.get(tid)
        if entry is None or not entry.call.arguments.get("approval"):
            raise ApplicationError("unknown approval call", non_retryable=True)
        if entry.approved is not None and entry.approved != approved:
            raise ApplicationError("conflicting approval", non_retryable=True)
        entry.approved = approved

    @workflow.update
    async def finish_turn(self, turn: TurnCheckpoint) -> None:
        self.check_attempt(turn.attempt)
        if not turn.uuid or not turn.attempt.burst <= turn.index < len(
            self.state.prompts
        ):
            raise ApplicationError("invalid completed turn", non_retryable=True)
        previous = self.state.turns.get(turn.index)
        if previous is not None:
            if (previous.uuid, previous.answer) != (turn.uuid, turn.answer):
                raise ApplicationError("conflicting completed turn", non_retryable=True)
            return
        self.state.turns[turn.index] = turn

    @workflow.update
    async def acknowledge(self, checkpoint: Checkpoint) -> None:
        self.check_attempt(checkpoint.attempt)
        expected = {
            tid
            for tid, e in self.state.ledger.items()
            if e.call.attempt.burst == checkpoint.attempt.burst
        }
        if not checkpoint.uuid or set(checkpoint.delivered) != expected:
            raise ApplicationError("incomplete checkpoint", non_retryable=True)
        if any(self.state.ledger[tid].outcome is None for tid in expected):
            raise ApplicationError("pending checkpoint", non_retryable=True)
        if self.state.checkpoint == checkpoint:
            return
        self.state.checkpoint = checkpoint
        self.state.session_id = checkpoint.session_id
        self.state.checkpoints.append(checkpoint)

    @workflow.query
    def snapshot(self) -> Snapshot:
        return Snapshot(
            self.attempts, self.state.ledger, self.state.checkpoints, self.state.turns
        )
