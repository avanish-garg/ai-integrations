"""Workflows for the hardening tests: built-in tools after a cancel, two live agents."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.claude_agent_sdk import DurableClaudeAgent, activity_as_tool
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from tests.endless.activities import count

COUNT = activity_as_tool(count, start_to_close_timeout=timedelta(seconds=30))


@workflow.defn
class WriterWorkflow:
    """An agent that may use Claude Code's Write tool, with a short heartbeat timeout."""

    def __init__(self) -> None:
        self.agent = DurableClaudeAgent(
            tools=[COUNT],
            builtin_tools=["Write"],
            segment_heartbeat_timeout=timedelta(seconds=2),
            segment_retry_policy=RetryPolicy(maximum_attempts=1),
        )

    @workflow.run
    async def run(self, prompt: str) -> str:
        """Run one task."""
        return await self.agent.run(prompt)


@workflow.defn
class TwoLiveAgentsWorkflow:
    """Wrong: two agents in one Workflow both ask for live output."""

    def __init__(self) -> None:
        self.first = DurableClaudeAgent(tools=[COUNT], live_output=True)
        self.initialization_error: str | None = None
        try:
            self.second = DurableClaudeAgent(tools=[COUNT], live_output=True)
        except RuntimeError as err:
            # Temporal allocates its run coroutine before constructing the
            # Workflow. Let construction finish so it awaits that coroutine;
            # an initialization failure otherwise leaves it unawaited on eviction.
            self.initialization_error = str(err)

    @workflow.run
    async def run(self, prompt: str) -> str:
        """Report the constructor's validation failure from the running coroutine."""
        if self.initialization_error is not None:
            raise RuntimeError(self.initialization_error)
        return await self.first.run(prompt)
