"""Test helpers: a scripted stand-in for Claude that needs no engine and no API key.

``ScriptedClaude`` is a segment runner driven by a Python policy. It keeps each
session in a folder, like the SDK's session store, so a new Worker process can
continue a session that a crashed Worker started. Like the real runner, each segment
ends at a checkpoint, and a segment that runs again starts over from the checkpoint
it was given, so a retry makes Claude decide again (with a new tool call id).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._events import emit
from ._models import DeferredCall, SegmentInput, SegmentOutput, ToolOutcome


@dataclass
class ToolCall:
    """A policy's decision to call a tool.

    Attributes:
        name: The durable tool to call.
        input: The call's arguments.
    """

    name: str
    input: dict[str, Any]


@dataclass
class Final:
    """A policy's final answer.

    Attributes:
        text: The answer.
    """

    text: str


@dataclass
class HistoryItem:
    """One finished tool call, as the policy sees it.

    Attributes:
        id: The ``tool_use_id``.
        name: The tool's name.
        input: The call's arguments.
        content: The tool's result.
        is_error: Whether the call failed.
    """

    id: str
    name: str
    input: dict[str, Any]
    content: Any
    is_error: bool


Policy = Callable[[str, "list[HistoryItem]"], "ToolCall | Final"]
"""Decides the next step from the latest prompt and every finished tool call so far."""


def _as_outcome(value: Any) -> ToolOutcome:
    if isinstance(value, ToolOutcome):
        return value
    return ToolOutcome(
        content=value.get("content"), is_error=bool(value.get("is_error"))
    )


class ScriptedClaude:
    """A segment runner that plays Claude with a Python policy."""

    def __init__(
        self,
        policy: Policy,
        state_dir: str | os.PathLike[str],
        *,
        cost_per_segment: float = 0.01,
        think_seconds: float = 0.0,
    ) -> None:
        """Create the runner.

        Args:
            policy: Decides each step.
            state_dir: Folder for session state, shared by all Workers in a test.
            cost_per_segment: Cost reported for each segment, in USD.
            think_seconds: Delay per segment, to leave time for a test to crash a Worker.
        """
        self._policy = policy
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cost = cost_per_segment
        self._think = think_seconds

    def _path(self, session_id: str, checkpoint: str) -> Path:
        return self._dir / session_id / f"{checkpoint}.json"

    def _load(self, session_id: str, checkpoint: str) -> dict[str, Any]:
        path = self._path(session_id, checkpoint)
        if not path.exists():
            raise RuntimeError(
                f"Unknown checkpoint {checkpoint} of session {session_id}"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, session_id: str, state: dict[str, Any]) -> str:
        """Store the state as a new checkpoint (never changed afterwards)."""
        checkpoint = f"cp_{uuid.uuid4().hex[:12]}"
        folder = self._dir / session_id
        folder.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(tmp, self._path(session_id, checkpoint))
        return checkpoint

    async def run(self, inp: SegmentInput, attempt: int) -> SegmentOutput:
        """Run one segment of the scripted conversation.

        Args:
            inp: The segment input.
            attempt: The Activity attempt number; a retry gets a new tool call id.

        Returns:
            A pause at the policy's next tool call, or its final answer.
        """
        if inp.checkpoint is None:
            state: dict[str, Any] = {
                "prompt": inp.prompt or "",
                "history": [],
                "pending": None,
                "turn": 0,
            }
        else:
            state = self._load(inp.session_id, inp.checkpoint)
            if inp.prompt is not None:
                state["prompt"] = inp.prompt  # a new task on the same session
        pending = state["pending"]
        if pending is not None:
            raw = inp.injected.get(pending["id"])
            if raw is None:
                return SegmentOutput(
                    session_id=inp.session_id,
                    is_error=True,
                    error=f"The paused call {pending['id']} got no result",
                )
            outcome = _as_outcome(raw)
            state["history"].append(
                {**pending, "content": outcome.content, "is_error": outcome.is_error}
            )
            state["pending"] = None
        if self._think:
            await asyncio.sleep(self._think)
        action = self._policy(
            state["prompt"], [HistoryItem(**h) for h in state["history"]]
        )
        state["turn"] += 1
        if isinstance(action, Final):
            emit({"type": "text", "text": action.text})  # what Claude writes
            return self._out(inp, self._save(inp.session_id, state), result=action.text)
        retry = f"_{attempt}" if attempt > 1 else ""
        call: dict[str, Any] = {
            "id": f"toolu_{inp.session_id[:8]}_{state['turn']:02d}{retry}",
            "name": action.name,
            "input": action.input,
        }
        state["pending"] = call
        checkpoint = self._save(inp.session_id, state)
        return self._out(inp, checkpoint, deferred=DeferredCall(**call))

    def _out(
        self,
        inp: SegmentInput,
        checkpoint: str,
        *,
        result: str | None = None,
        deferred: DeferredCall | None = None,
    ) -> SegmentOutput:
        return SegmentOutput(
            session_id=inp.session_id,
            result=result,
            deferred=deferred,
            checkpoint=checkpoint,
            cost_usd=self._cost,
        )


__all__ = ["Final", "HistoryItem", "Policy", "ScriptedClaude", "ToolCall"]
