"""A SessionStore for the Claude Agent SDK that keeps transcripts in a folder."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import SessionKey, SessionStoreEntry

_MAX_NAME = 240  # file name length, without ".jsonl"
_CACHED_FILES = 256  # transcripts whose entry ids are kept in memory


def _complete_entries(text: str) -> list[Any]:
    """The entries of a JSON Lines text, ignoring a last line still being written."""
    complete = text[: text.rfind("\n") + 1]
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


class FileSessionStore:
    """Keeps Claude session transcripts as JSON Lines files in one folder.

    Made for tests, local development and a single machine: every Worker that may
    resume a session must see the same folder. In production, use a store backed by
    a database or object storage (the Claude Agent SDK repository has example
    stores for S3, Redis and Postgres). It implements the two required
    ``SessionStore`` methods, ``append`` and ``load``, and treats each entry's
    ``uuid`` as an idempotency key, as the SDK asks. File work runs in a thread, so
    it does not block the Worker's event loop.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Create the store.

        Args:
            root: The folder; it is created if missing.
        """
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        # Entry ids per file, with the file size they were read at.
        self._seen: OrderedDict[Path, tuple[int, set[str]]] = OrderedDict()
        self._lock = threading.Lock()  # appends run in threads

    def _path(self, key: SessionKey) -> Path:
        name = f"{key['project_key']}__{key['session_id']}"
        subpath = key.get("subpath")
        if subpath:
            name += "__" + str(subpath)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        if len(safe) > _MAX_NAME:  # keep long keys apart: never just cut them
            digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]
            safe = f"{safe[: _MAX_NAME - len(digest) - 1]}-{digest}"
        return self._root / (safe + ".jsonl")

    def _uuids(self, path: Path) -> set[str]:
        """Entry ids already in ``path`` (read again if another process wrote to it)."""
        size = path.stat().st_size if path.exists() else 0
        cached = self._seen.get(path)
        if cached is not None and cached[0] == size:
            self._seen.move_to_end(path)
            return cached[1]
        found: set[str] = set()
        if size:
            for entry in _complete_entries(path.read_text(encoding="utf-8")):
                uid = entry.get("uuid")
                if uid:
                    found.add(uid)
        self._remember(path, size, found)
        return found

    def _remember(self, path: Path, size: int, uuids: set[str]) -> None:
        self._seen[path] = (size, uuids)
        self._seen.move_to_end(path)
        while len(self._seen) > _CACHED_FILES:
            self._seen.popitem(last=False)

    def _append(self, path: Path, entries: list[SessionStoreEntry]) -> None:
        with self._lock:
            seen = self._uuids(path)
            lines: list[str] = []
            for entry in entries:
                uid = entry.get("uuid")
                if uid and uid in seen:
                    continue
                lines.append(json.dumps(entry, separators=(",", ":")) + "\n")
                if uid:
                    seen.add(uid)
            if lines:  # one write, so another process never sees half of it
                flags = (
                    os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
                )
                fd = os.open(path, flags, 0o644)
                try:
                    os.write(fd, "".join(lines).encode("utf-8"))
                finally:
                    os.close(fd)
            self._remember(path, path.stat().st_size if path.exists() else 0, seen)

    async def append(self, key: SessionKey, entries: list[SessionStoreEntry]) -> None:
        """Append transcript entries, skipping entries whose uuid is already stored.

        Args:
            key: The session key (``project_key``, ``session_id``, optional ``subpath``).
            entries: The entries to add.
        """
        await asyncio.to_thread(self._append, self._path(key), entries)

    def _load(self, path: Path) -> list[SessionStoreEntry] | None:
        with self._lock:  # never while this process appends to it
            if not path.exists():
                return None
            text = path.read_text(encoding="utf-8")
        return cast("list[SessionStoreEntry]", _complete_entries(text))

    async def load(self, key: SessionKey) -> list[SessionStoreEntry] | None:
        """Load a session's entries.

        Args:
            key: The session key.

        Returns:
            The entries in order, or None if the session is unknown.
        """
        return await asyncio.to_thread(self._load, self._path(key))
