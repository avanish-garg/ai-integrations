"""One-line worker setup."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from temporalio.plugin import SimplePlugin

from ._activity import SegmentRunner, make_segment_activity


class ClaudeAgentPlugin(SimplePlugin):
    """Registers the segment Activity that runs Claude on the Worker.

    Example:
        .. code-block:: python

            runner = ClaudeAgentSdkRunner(session_store=FileSessionStore("/shared/sessions"))
            worker = Worker(
                client,
                task_queue="agents",
                workflows=[MyAgentWorkflow],
                activities=[my_tool],
                plugins=[ClaudeAgentPlugin(runner)],
            )
    """

    def __init__(self, runner: SegmentRunner, *, heartbeat_every: float = 5.0) -> None:
        """Create the plugin.

        Args:
            runner: Runs each model segment, for example ``ClaudeAgentSdkRunner``.
            heartbeat_every: Seconds between heartbeats while a segment runs.
        """
        segment_activity = make_segment_activity(
            runner, heartbeat_every=heartbeat_every
        )

        def activities(existing: Sequence[Any] | None) -> list[Any]:
            return [*(existing or []), segment_activity]

        super().__init__("ClaudeAgentPlugin", activities=activities)
