"""Record the whole native batch before scheduling independently retried calls."""

import asyncio
import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from tests.hybrid.models import Reply
    from tests.hybrid.replay_models import ReplayCall, ReplayDecision, ReplayState


@workflow.defn
class NativeReplayWorkflow:
    def __init__(self) -> None:
        self.state = ReplayState("")

    async def execute(self, call: ReplayCall) -> None:
        result = await workflow.execute_activity(
            "native_replay_execution",
            call,
            result_type=Reply,
            activity_id="tool-" + call.id,
            start_to_close_timeout=timedelta(seconds=30),
            heartbeat_timeout=timedelta(seconds=3),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        self.state.results[call.id] = json.loads(result.text)

    @workflow.run
    async def run(self, state: ReplayState) -> ReplayState:
        self.state = state
        while True:
            decision = await workflow.execute_activity(
                "native_replay_decision",
                ReplayDecision(state.session_id, state.round),
                result_type=ReplayDecision,
                activity_id=f"decision-{state.round}",
                start_to_close_timeout=timedelta(seconds=30),
                heartbeat_timeout=timedelta(seconds=3),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
            if not decision.calls:
                state.answer = decision.answer
                return state
            state.calls.extend(decision.calls)
            if all(call.name == "Read" for call in decision.calls):
                await asyncio.gather(*(self.execute(c) for c in decision.calls))
            else:
                # Edit changes the shared workspace and native Read context.
                for call in decision.calls:
                    await self.execute(call)
            state.round += 1

    @workflow.query
    def snapshot(self) -> ReplayState:
        return self.state
