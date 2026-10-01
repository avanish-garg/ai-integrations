"""Internal receipts for the checkpointed native execution experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class NativeIntent:
    id: str
    name: str
    arguments: dict[str, Any]
    session_id: str
    index: int
    version: int
    transcript_uuid: str
    checkpoint_uuid: str
    outcome_kind: Literal["deferred", "validation"] = "deferred"
    answer: str = ""


@dataclass
class NativeDecision:
    session_id: str
    index: int
    intent: NativeIntent | None = None
    answer: str = ""


@dataclass
class NativeRun:
    session_id: str
    index: int = 0
    intents: list[NativeIntent] = field(default_factory=list)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    answer: str = ""
