"""Immutable engine checkpoints and atomic native result/workspace publication."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import SessionKey, SessionStoreEntry, project_key_for_directory

from tests.hybrid.executor_models import NativeIntent
from tests.hybrid.models import Call
from tests.hybrid.native_store import NativeStore


class ExecutionStore(NativeStore):
    def __init__(self, root: Path, phase: str = "live") -> None:
        super().__init__(root, phase)
        self.fail_publication = False
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS native_checkpoints (id TEXT PRIMARY KEY, "
                "intent TEXT, entries TEXT, digest TEXT)"
            )

    def key(self, session_id: str) -> SessionKey:
        return {
            "project_key": project_key_for_directory(str(self.workspace)),
            "session_id": session_id,
        }

    def decision(self, session_id: str, index: int) -> NativeIntent | None:
        with self.connect() as db:
            rows = db.execute("SELECT intent FROM native_checkpoints").fetchall()
        for row in rows:
            value = json.loads(row[0])
            if (value["session_id"], value["index"]) == (session_id, index):
                return NativeIntent(**value)
        return None

    def version(self) -> int:
        with self.connect() as db:
            return int(
                db.execute("SELECT version FROM workspace WHERE id=1").fetchone()[0]
            )

    def _publish_transcript(
        self,
        db: sqlite3.Connection,
        key: SessionKey,
        entries: list[SessionStoreEntry],
    ) -> None:
        if (
            db.execute("SELECT owner FROM workspace WHERE id=1").fetchone()[0]
            != self.owner
        ):
            raise RuntimeError("stale native decision writer")
        previous = [
            json.loads(row[0])
            for row in db.execute(
                "SELECT data FROM entries WHERE project=? AND session=? AND subpath='' ORDER BY seq",
                (key["project_key"], key["session_id"]),
            )
        ]
        if entries[: len(previous)] != previous:
            raise RuntimeError("canonical session moved during native decision")
        db.executemany(
            "INSERT INTO entries(project,session,subpath,uuid,data) VALUES (?,?,?,?,?)",
            [
                (
                    key["project_key"],
                    key["session_id"],
                    "",
                    e.get("uuid"),
                    json.dumps(e),
                )
                for e in entries[len(previous) :]
            ],
        )

    def publish_decision(
        self, session_id: str, entries: list[SessionStoreEntry]
    ) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._publish_transcript(db, self.key(session_id), entries)

    def freeze(self, intent: NativeIntent, entries: list[SessionStoreEntry]) -> None:
        payload = json.dumps(entries, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute("SELECT owner FROM workspace WHERE id=1").fetchone()[0]
                != self.owner
            ):
                raise RuntimeError("stale native checkpoint writer")
            previous = db.execute(
                "SELECT intent,entries,digest FROM native_checkpoints WHERE id=?",
                (intent.id,),
            ).fetchone()
            value = (json.dumps(asdict(intent), sort_keys=True), payload, digest)
            if previous and tuple(previous) != value:
                raise RuntimeError("conflicting immutable native checkpoint")
            # No canonical pending call exists without its recovery receipt.
            # CLI decision attempts write only to disposable transcript stores.
            self._publish_transcript(db, self.key(intent.session_id), entries)
            db.execute(
                "INSERT OR IGNORE INTO native_checkpoints VALUES (?,?,?,?)",
                (intent.id, *value),
            )

    def frozen(self, intent: NativeIntent) -> list[SessionStoreEntry]:
        with self.connect() as db:
            row = db.execute(
                "SELECT intent,entries,digest FROM native_checkpoints WHERE id=?",
                (intent.id,),
            ).fetchone()
        if row is None or json.loads(row[0]) != asdict(intent):
            raise RuntimeError("missing or conflicting native checkpoint receipt")
        if hashlib.sha256(row[1].encode()).hexdigest() != row[2]:
            raise RuntimeError("native checkpoint digest differs")
        entries = cast(list[SessionStoreEntry], json.loads(row[1]))
        calls = [
            block
            for entry in entries
            if entry.get("uuid") == intent.transcript_uuid
            for block in cast(dict[str, Any], entry.get("message", {})).get(
                "content", []
            )
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        markers = [
            cast(dict[str, Any], entry.get("attachment", {}))
            for entry in entries
            if cast(dict[str, Any], entry.get("attachment", {})).get("type")
            == "hook_deferred_tool"
            and cast(dict[str, Any], entry.get("attachment", {})).get("toolUseID")
            == intent.id
        ]
        if (
            len(calls) != 1
            or (calls[0].get("id"), calls[0].get("name"), calls[0].get("input"))
            != (intent.id, intent.name, intent.arguments)
            or len(markers) != 1
            or (markers[0].get("toolName"), markers[0].get("toolInput"))
            != (intent.name, intent.arguments)
            or intent.name not in {"Read", "Edit"}
            or Path(intent.arguments["file_path"]).resolve()
            != self.workspace / "note.txt"
        ):
            raise RuntimeError("native checkpoint does not authorize the accepted call")
        return entries

    def claim(self, call: Call, intent: NativeIntent) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            owner, version = db.execute(
                "SELECT owner,version FROM workspace WHERE id=1"
            ).fetchone()
            if owner != call.attempt.token or version != intent.version:
                raise RuntimeError("native executor has stale input workspace")
            arguments = json.dumps(call.arguments, sort_keys=True)
            previous = db.execute(
                "SELECT name,arguments,result FROM native_calls WHERE id=?", (call.id,)
            ).fetchone()
            if previous and tuple(previous[:2]) != (call.name, arguments):
                raise RuntimeError("conflicting native execution ID")
            if previous and previous[2] is not None:
                raise RuntimeError("completed native call must use its cached outcome")
            db.execute(
                "INSERT OR IGNORE INTO native_calls VALUES (?,?,?,?,'requested',NULL,NULL,NULL)",
                (call.id, call.name, arguments, call.attempt.token),
            )
            db.execute(
                "UPDATE native_calls SET owner=?,phase='requested',staged=NULL WHERE id=?",
                (call.attempt.token, call.id),
            )

    def publish(
        self,
        intent: NativeIntent,
        block: dict[str, Any],
        suffix: list[SessionStoreEntry],
    ) -> None:
        key = self.key(intent.session_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            owner, version = db.execute(
                "SELECT owner,version FROM workspace WHERE id=1"
            ).fetchone()
            row = db.execute(
                "SELECT owner,staged,result FROM native_calls WHERE id=?", (intent.id,)
            ).fetchone()
            if not row or owner != self.owner or row[0] != self.owner:
                raise RuntimeError("stale native execution publication")
            if version != intent.version or row[1] is None or row[2] is not None:
                raise RuntimeError("native result/workspace commit conflict")
            head = db.execute(
                "SELECT uuid FROM entries WHERE project=? AND session=? AND uuid IS NOT NULL ORDER BY seq DESC LIMIT 1",
                (key["project_key"], key["session_id"]),
            ).fetchone()
            if not head or head[0] != intent.checkpoint_uuid:
                raise RuntimeError("canonical session moved during native execution")
            # Publish actual engine entries, not manufactured transcript markers.
            db.executemany(
                "INSERT OR IGNORE INTO entries(project,session,subpath,uuid,data) VALUES (?,?,?,?,?)",
                [
                    (
                        key["project_key"],
                        key["session_id"],
                        "",
                        e.get("uuid"),
                        json.dumps(e),
                    )
                    for e in suffix
                ],
            )
            db.execute(
                "UPDATE workspace SET files=?,version=? WHERE id=1",
                (row[1], version + 1),
            )
            db.execute(
                "UPDATE native_calls SET result=?,version=?,phase='committed' WHERE id=?",
                (json.dumps(block, sort_keys=True), version + 1, intent.id),
            )
            db.execute(
                "INSERT INTO native_events(id,phase,owner) VALUES (?,'committed',?)",
                (intent.id, self.owner),
            )
            if self.fail_publication and intent.name == "Edit":
                self.fail_publication = False
                # Roll back transcript, result and workspace together, even
                # though the real CLI has already changed its local file.
                raise OSError("injected atomic native publication failure")
