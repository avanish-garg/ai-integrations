"""Durably accept a native checkpoint before scheduling its native executor."""

import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from tests.hybrid.executor_models import NativeDecision, NativeRun
    from tests.hybrid.models import Reply


@workflow.defn
class NativeExecutionWorkflow:
    def __init__(self) -> None:
        self.state = NativeRun("")

    @workflow.run
    async def run(self, state: NativeRun) -> NativeRun:
        self.state = state
        while True:
            decision = await workflow.execute_activity(
                "native_decision",
                NativeDecision(state.session_id, state.index),
                result_type=NativeDecision,
                activity_id=f"decision-{state.index}",
                start_to_close_timeout=timedelta(seconds=30),
                heartbeat_timeout=timedelta(seconds=3),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
            if decision.intent is None:
                state.answer = decision.answer
                return state
            intent = decision.intent
            state.intents.append(intent)
            reply = await workflow.execute_activity(
                "native_execution",
                intent,
                result_type=Reply,
                activity_id="tool-" + intent.id,
                start_to_close_timeout=timedelta(seconds=30),
                heartbeat_timeout=timedelta(seconds=3),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            # The native block, including its original ID, is the outcome.
            state.results[intent.id] = json.loads(reply.text)
            state.index += 1
            if intent.outcome_kind == "validation" and decision.answer:
                state.answer = decision.answer
                return state

    @workflow.query
    def snapshot(self) -> NativeRun:
        return self.state
