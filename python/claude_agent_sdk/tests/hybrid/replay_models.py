"""Internal receipts for exact native-call replay; no provider rediscovery."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReplayCall:
    id: str
    name: str
    arguments: dict[str, Any]
    session_id: str
    round: int
    position: int
    batch_id: str
    digest: str
    transcript_uuid: str


@dataclass
class ReplayDecision:
    session_id: str
    round: int = 0
    calls: list[ReplayCall] = field(default_factory=list)
    answer: str = ""


@dataclass
class ReplayState:
    session_id: str
    round: int = 0
    calls: list[ReplayCall] = field(default_factory=list)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    answer: str = ""
