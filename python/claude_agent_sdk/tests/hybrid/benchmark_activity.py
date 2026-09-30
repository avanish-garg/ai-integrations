"""Read-only tool used by the segment benchmark."""

from __future__ import annotations

from typing import Any

from temporalio import activity


@activity.defn(name="echo")
async def segment_echo(arguments: dict[str, Any]) -> dict[str, int]:
    """Echo one number with no external side effects."""
    return {"n": arguments["n"]}
