"""Transactional test store with parent transcripts and discoverable child subkeys."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import SessionKey, SessionStoreEntry


class TranscriptStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.delay = 0.0
        self.fail = False
        self.writing = asyncio.Event()
        self.loads: list[SessionKey] = []
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS entries (seq INTEGER PRIMARY KEY, "
                "project TEXT, session TEXT, subpath TEXT, uuid TEXT, data TEXT, "
                "UNIQUE(project, session, subpath, uuid))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY, data TEXT)"
            )

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def _append(self, key: SessionKey, entries: list[SessionStoreEntry]) -> None:
        with self.connect() as db:
            db.executemany(
                "INSERT OR IGNORE INTO entries(project,session,subpath,uuid,data) "
                "VALUES (?,?,?,?,?)",
                [
                    (
                        key["project_key"],
                        key["session_id"],
                        key.get("subpath", ""),
                        e.get("uuid"),
                        json.dumps(e),
                    )
                    for e in entries
                ],
            )

    async def append(self, key: SessionKey, entries: list[SessionStoreEntry]) -> None:
        self.writing.set()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise OSError("injected transcript write failure")
        await asyncio.to_thread(self._append, key, entries)

    async def append_if_unchanged(
        self,
        key: SessionKey,
        expected_last_uuid: str,
        entries: list[SessionStoreEntry],
    ) -> bool:
        self.writing.set()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise OSError("injected recovery write failure")

        def commit() -> bool:
            with self.connect() as db:
                # Ordinary appends take SQLite's same writer lock. The head
                # check and the whole recovery batch commit together.
                db.execute("BEGIN IMMEDIATE")
                head = db.execute(
                    "SELECT uuid FROM entries WHERE project=? AND session=? "
                    "AND subpath=? AND uuid IS NOT NULL ORDER BY seq DESC LIMIT 1",
                    (key["project_key"], key["session_id"], key.get("subpath", "")),
                ).fetchone()
                if not head or head[0] != expected_last_uuid:
                    return False
                db.executemany(
                    "INSERT INTO entries(project,session,subpath,uuid,data) "
                    "VALUES (?,?,?,?,?)",
                    [
                        (
                            key["project_key"],
                            key["session_id"],
                            key.get("subpath", ""),
                            e.get("uuid"),
                            json.dumps(e),
                        )
                        for e in entries
                    ],
                )
            return True

        return await asyncio.to_thread(commit)

    def _load(self, key: SessionKey) -> list[SessionStoreEntry] | None:
        with self.connect() as db:
            rows = db.execute(
                "SELECT data FROM entries WHERE project=? AND session=? "
                "AND subpath=? ORDER BY seq",
                (key["project_key"], key["session_id"], key.get("subpath", "")),
            ).fetchall()
        return cast("list[SessionStoreEntry]", [json.loads(r[0]) for r in rows]) or None

    async def load(self, key: SessionKey) -> list[SessionStoreEntry] | None:
        self.loads.append(key)
        return await asyncio.to_thread(self._load, key)

    async def list_subkeys(self, key: Any) -> list[str]:
        def read() -> list[str]:
            with self.connect() as db:
                return [
                    r[0]
                    for r in db.execute(
                        "SELECT DISTINCT subpath FROM entries WHERE project=? "
                        "AND session=? AND subpath!=''",
                        (key["project_key"], key["session_id"]),
                    )
                ]

        return await asyncio.to_thread(read)

    async def transcripts(self, key: SessionKey) -> dict[str, list[dict[str, Any]]]:
        paths = ["", *await self.list_subkeys(key)]
        return {
            p: cast(
                "list[dict[str, Any]]",
                await self.load(
                    {
                        "project_key": key["project_key"],
                        "session_id": key["session_id"],
                        "subpath": p,
                    }
                )
                or [],
            )
            for p in paths
        }

    async def wait_call(
        self,
        key: SessionKey,
        tid: str,
        name: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> tuple[str, str]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for subpath, entries in (await self.transcripts(key)).items():
                for entry in entries:
                    content = entry.get("message", {}).get("content", [])
                    if not isinstance(content, list):
                        continue
                    for block in content:
                        if block.get("type") == "tool_use" and block.get("id") == tid:
                            if (block.get("name"), block.get("input")) != (
                                name,
                                arguments,
                            ):
                                raise RuntimeError(
                                    "native ID disagrees with stored assistant call"
                                )
                            return str(entry["uuid"]), subpath
            await asyncio.sleep(0.02)
        raise RuntimeError(f"transcript not stored for native call {tid}")

    def effect(self, key: str, data: dict[str, Any]) -> str:
        """A local idempotent side effect; external services need their own equivalent."""
        text = json.dumps({**data, "idempotency_key": key}, sort_keys=True)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO effects VALUES (?,?)", (key, text))
            row = db.execute("SELECT data FROM effects WHERE key=?", (key,)).fetchone()
        assert row is not None
        return str(row[0])
