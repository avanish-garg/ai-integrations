"""Test-only wire types for the hybrid experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Attempt:
    burst: int
    number: int
    token: str


@dataclass
class Call:
    id: str
    name: str
    arguments: dict[str, Any]
    attempt: Attempt
    transcript_uuid: str
    subpath: str = ""

    def identity(self) -> tuple[str, dict[str, Any], str]:
        return self.name, self.arguments, self.subpath


@dataclass
class Reply:
    text: str
    is_error: bool = False


@dataclass
class Entry:
    call: Call
    approved: bool | None = None
    scheduled: bool = False
    outcome: Reply | None = None


@dataclass
class Checkpoint:
    attempt: Attempt
    session_id: str
    uuid: str
    delivered: list[str]
    pid: int
    answers: list[str] = field(default_factory=list)


@dataclass
class BurstInput:
    session_id: str
    prompts: list[str]
    burst: int = 0
    checkpoint: Checkpoint | None = None


@dataclass
class State:
    session_id: str
    prompts: list[str]
    burst_size: int = 0
    continue_every: int = 0
    index: int = 0
    checkpoint: Checkpoint | None = None
    ledger: dict[str, Entry] = field(default_factory=dict)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)


@dataclass
class Snapshot:
    attempts: dict[int, Attempt]
    ledger: dict[str, Entry]
    checkpoints: list[Checkpoint]
