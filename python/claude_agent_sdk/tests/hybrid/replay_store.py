"""Native intent receipts, isolated execution contexts and atomic publication."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import SessionKey, SessionStoreEntry, project_key_for_directory

from tests.hybrid.native_store import NativeStore
from tests.hybrid.replay_models import ReplayCall


class ReplayStore(NativeStore):
    def __init__(self, root: Path, phase: str = "live") -> None:
        super().__init__(root, phase)
        self.fail_publication = False
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS replay_batches (id TEXT PRIMARY KEY, "
                "session TEXT, round INTEGER, calls TEXT, source TEXT, digest TEXT, "
                "owner TEXT, files TEXT, head TEXT, readonly INTEGER, UNIQUE(session,round))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS replay_attempts (id TEXT PRIMARY KEY, "
                "attempt INTEGER, actor TEXT, version INTEGER)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS replay_entries (id TEXT PRIMARY KEY, data TEXT)"
            )

    def key(self, session: str) -> SessionKey:
        return {
            "project_key": project_key_for_directory(str(self.workspace)),
            "session_id": session,
        }

    def files(self) -> dict[str, Any]:
        path = self.workspace / "note.txt"
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("native replay workspace is not a regular file")
        info = path.stat()
        return {
            "note.txt": {
                "data": base64.b64encode(path.read_bytes()).decode(),
                "mode": stat.S_IMODE(info.st_mode),
                "mtime_ns": info.st_mtime_ns,
            }
        }

    def version(self) -> int:
        with self.connect() as db:
            return int(
                db.execute("SELECT version FROM workspace WHERE id=1").fetchone()[0]
            )

    def decision(self, session: str, round: int) -> list[ReplayCall] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT calls FROM replay_batches WHERE session=? AND round=?",
                (session, round),
            ).fetchone()
        return [ReplayCall(**c) for c in json.loads(row[0])] if row else None

    def freeze(self, calls: list[ReplayCall], source: list[SessionStoreEntry]) -> None:
        if not calls:
            raise RuntimeError("empty native batch receipt")
        key = self.key(calls[0].session_id)
        payload = json.dumps(source, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        if any(c.digest != digest for c in calls):
            raise RuntimeError("native batch source digest differs")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            accepted = db.execute(
                "SELECT calls,source FROM replay_batches WHERE session=? AND round=?",
                (calls[0].session_id, calls[0].round),
            ).fetchone()
            serialized_calls = json.dumps([asdict(c) for c in calls], sort_keys=True)
            if accepted:
                if accepted != (serialized_calls, payload):
                    raise RuntimeError("conflicting native batch receipt")
                return
            previous_ids = {
                c["id"]
                for r in db.execute("SELECT calls FROM replay_batches")
                for c in json.loads(r[0])
            }
            if len({c.id for c in calls}) != len(calls) or previous_ids.intersection(
                c.id for c in calls
            ):
                raise RuntimeError("native call ID reused in another accepted batch")
            owner, raw_files = db.execute(
                "SELECT owner,files FROM workspace WHERE id=1"
            ).fetchone()
            if owner != self.owner or self.files() != json.loads(raw_files):
                raise RuntimeError(
                    "native preparation changed its workspace or lost ownership"
                )
            previous = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT data FROM entries WHERE project=? AND session=? AND subpath='' ORDER BY seq",
                    (key["project_key"], key["session_id"]),
                )
            ]
            if source[: len(previous)] != previous:
                raise RuntimeError("native preparation changed its canonical prefix")
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
                    for e in source[len(previous) :]
                ],
            )
            head = next(str(e.get("uuid")) for e in reversed(source) if e.get("uuid"))
            db.execute(
                "INSERT INTO replay_batches VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    calls[0].batch_id,
                    calls[0].session_id,
                    calls[0].round,
                    serialized_calls,
                    payload,
                    digest,
                    owner,
                    raw_files,
                    head,
                    int(all(c.name == "Read" for c in calls)),
                ),
            )

    def terminal(self, session: str, entries: list[SessionStoreEntry]) -> None:
        key = self.key(session)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            owner, files = db.execute(
                "SELECT owner,files FROM workspace WHERE id=1"
            ).fetchone()
            if owner != self.owner or self.files() != json.loads(files):
                raise RuntimeError("native terminal turn changed its workspace")
            previous = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT data FROM entries WHERE project=? AND session=? AND subpath='' ORDER BY seq",
                    (key["project_key"], key["session_id"]),
                )
            ]
            if entries[: len(previous)] != previous:
                raise RuntimeError("native terminal turn changed its canonical prefix")
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

    def receipt(self, call: ReplayCall) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT calls,source,digest,owner,files,readonly FROM replay_batches WHERE id=?",
                (call.batch_id,),
            ).fetchone()
        if row is None or asdict(call) not in json.loads(row[0]):
            raise RuntimeError("native replay receipt differs from its accepted call")
        if (
            call.digest != row[2]
            or hashlib.sha256(row[1].encode()).hexdigest() != row[2]
        ):
            raise RuntimeError("native replay source digest differs")
        source = json.loads(row[1])
        blocks = [
            b
            for e in source
            if e.get("uuid") == call.transcript_uuid
            for b in e.get("message", {}).get("content", [])
            if isinstance(b, dict)
            and b.get("type") == "tool_use"
            and b.get("id") == call.id
        ]
        if len(blocks) != 1 or (blocks[0].get("name"), blocks[0].get("input")) != (
            call.name,
            call.arguments,
        ):
            raise RuntimeError(
                "native replay ID differs from its original assistant call"
            )
        if (
            call.name not in {"Read", "Edit"}
            or Path(call.arguments["file_path"]).resolve()
            != self.workspace / "note.txt"
        ):
            raise RuntimeError(
                "native replay escaped its managed tool set or workspace"
            )
        return {
            "source": source,
            "block": blocks[0],
            "owner": row[3],
            "files": json.loads(row[4]),
            "readonly": bool(row[5]),
            "calls": json.loads(row[0]),
        }

    def claim(
        self, call: ReplayCall, actor: str, attempt: int
    ) -> dict[str, Any] | None:
        receipt = self.receipt(call)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            owner, raw_files, version = db.execute(
                "SELECT owner,files,version FROM workspace WHERE id=1"
            ).fetchone()
            if owner != receipt["owner"]:
                raise RuntimeError("native replay lost its workspace lease")
            cached = db.execute(
                "SELECT result FROM native_calls WHERE id=?", (call.id,)
            ).fetchone()
            if cached and cached[0] is not None:
                return cast(dict[str, Any], json.loads(cached[0]))
            earlier = db.execute(
                "SELECT attempt,actor FROM replay_attempts WHERE id=?", (call.id,)
            ).fetchone()
            if earlier and (
                attempt < earlier[0] or (attempt == earlier[0] and actor != earlier[1])
            ):
                raise RuntimeError("stale native replay Activity attempt")
            files = json.loads(raw_files)
            if receipt["readonly"] and files != receipt["files"]:
                raise RuntimeError("readonly native batch input changed")
            if not receipt["readonly"]:
                for previous in receipt["calls"][: call.position]:
                    done = db.execute(
                        "SELECT result FROM native_calls WHERE id=?", (previous["id"],)
                    ).fetchone()
                    if not done or done[0] is None:
                        raise RuntimeError("mutable native batch executed out of order")
            # Immutable readers share one materialization. Mutating calls run
            # sequentially and reset an uncommitted local write on every retry.
            self.workspace.mkdir(exist_ok=True)
            path = self.workspace / "note.txt"
            if not receipt["readonly"] or not path.exists():
                entry = files["note.txt"]
                path.write_bytes(base64.b64decode(entry["data"]))
                path.chmod(entry["mode"])
                os.utime(path, ns=(entry["mtime_ns"], entry["mtime_ns"]))
            if self.files() != files:
                raise RuntimeError("native replay physical workspace differs")
            db.execute(
                "INSERT OR REPLACE INTO replay_attempts VALUES (?,?,?,?)",
                (call.id, attempt, actor, version),
            )
            db.execute(
                "INSERT OR IGNORE INTO native_calls VALUES (?,?,?,?,'requested',NULL,NULL,NULL)",
                (call.id, call.name, json.dumps(call.arguments, sort_keys=True), actor),
            )
            db.execute(
                "UPDATE native_calls SET owner=?,phase='requested' WHERE id=?",
                (actor, call.id),
            )
        return None

    async def context(self, call: ReplayCall) -> list[SessionStoreEntry]:
        receipt = self.receipt(call)
        entries = await self.load(self.key(call.session_id))
        assert entries is not None
        with self.connect() as db:
            completed = {
                r[0]
                for r in db.execute(
                    "SELECT id FROM native_calls WHERE result IS NOT NULL"
                )
            }
        unresolved = {c["id"] for c in receipt["calls"]} - completed
        context: list[SessionStoreEntry] = []
        parent = None
        for original in entries:
            entry = copy.deepcopy(original)
            message = cast(dict[str, Any], entry.get("message", {}))
            content = message.get("content")
            if isinstance(content, list):
                filtered = [
                    b
                    for b in content
                    if not (
                        isinstance(b, dict)
                        and b.get("type") == "tool_use"
                        and b.get("id") in unresolved
                    )
                ]
                if not filtered:
                    continue
                message["content"] = filtered
            if entry.get("uuid"):
                cast(Any, entry)["parentUuid"] = parent
                parent = entry.get("uuid")
            context.append(entry)
        return context

    def event(self, call: ReplayCall, actor: str, phase: str) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT owner FROM native_calls WHERE id=?", (call.id,)
            ).fetchone()
            if not row or row[0] != actor:
                raise RuntimeError("stale native replay event")
            db.execute("UPDATE native_calls SET phase=? WHERE id=?", (phase, call.id))
            db.execute(
                "INSERT INTO native_events(id,phase,owner) VALUES (?,?,?)",
                (call.id, phase, actor),
            )

    def publish(
        self, call: ReplayCall, actor: str, actual: SessionStoreEntry
    ) -> dict[str, Any]:
        receipt = self.receipt(call)
        blocks = cast(dict[str, Any], actual.get("message", {})).get("content", [])
        results = [
            b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        if (
            not actual.get("uuid")
            or len(results) != 1
            or results[0].get("tool_use_id") != call.id
        ):
            raise RuntimeError("native executor result differs from original ID")
        block = results[0]
        files = self.files()
        key = self.key(call.session_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            owner, raw_files, version = db.execute(
                "SELECT owner,files,version FROM workspace WHERE id=1"
            ).fetchone()
            invocation = db.execute(
                "SELECT actor,version FROM replay_attempts WHERE id=?", (call.id,)
            ).fetchone()
            row = db.execute(
                "SELECT owner,result FROM native_calls WHERE id=?", (call.id,)
            ).fetchone()
            if (
                not row
                or row[0] != actor
                or owner != receipt["owner"]
                or not invocation
                or invocation[0] != actor
            ):
                raise RuntimeError("stale native replay publication")
            if row[1] is not None:
                raise RuntimeError("native outcome already committed")
            if receipt["readonly"]:
                if files != receipt["files"] or json.loads(raw_files) != files:
                    raise RuntimeError("readonly native execution changed workspace")
            elif version != invocation[1]:
                raise RuntimeError("mutable native execution input version changed")
            head = db.execute(
                "SELECT head FROM replay_batches WHERE id=?", (call.batch_id,)
            ).fetchone()[0]
            canonical = db.execute(
                "SELECT uuid FROM entries WHERE project=? AND session=? AND subpath='' ORDER BY seq DESC LIMIT 1",
                (key["project_key"], key["session_id"]),
            ).fetchone()[0]
            if head != canonical:
                raise RuntimeError("canonical native conversation moved")
            entry = copy.deepcopy(actual)
            # Reattach the real executor's result carrier to the original
            # conversation. Its block and native toolUseResult stay unchanged.
            cast(Any, entry)["parentUuid"] = head
            db.execute(
                "INSERT INTO entries(project,session,subpath,uuid,data) VALUES (?,?,?,?,?)",
                (
                    key["project_key"],
                    key["session_id"],
                    "",
                    entry.get("uuid"),
                    json.dumps(entry),
                ),
            )
            db.execute(
                "INSERT INTO replay_entries VALUES (?,?)", (call.id, json.dumps(entry))
            )
            db.execute(
                "UPDATE replay_batches SET head=? WHERE id=?",
                (entry.get("uuid"), call.batch_id),
            )
            db.execute(
                "UPDATE workspace SET files=?,version=? WHERE id=1",
                (json.dumps(files), version + 1),
            )
            db.execute(
                "UPDATE native_calls SET result=?,version=?,phase='committed' WHERE id=?",
                (json.dumps(block, sort_keys=True), version + 1, call.id),
            )
            db.execute(
                "INSERT INTO native_events(id,phase,owner) VALUES (?,'committed',?)",
                (call.id, actor),
            )
            if self.fail_publication and call.name == "Edit":
                self.fail_publication = False
                raise OSError("injected native replay publication failure")
        return cast(dict[str, Any], block)
