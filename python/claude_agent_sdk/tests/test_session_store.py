"""FileSessionStore follows the SessionStore contracts, checked with the SDK's own suite."""

from __future__ import annotations

import itertools
from pathlib import Path

from claude_agent_sdk import SessionStore
from claude_agent_sdk.testing import run_session_store_conformance

from temporalio.claude_agent_sdk import FileSessionStore


async def test_file_session_store_passes_the_sdk_conformance_suite(
    tmp_path: Path,
) -> None:
    fresh = itertools.count()

    def make_store() -> SessionStore:
        # SessionStore's type includes the optional methods (list, delete), which
        # FileSessionStore does not implement; the suite skips their contracts.
        return FileSessionStore(tmp_path / f"store-{next(fresh)}")  # type: ignore[return-value]

    await run_session_store_conformance(make_store)
