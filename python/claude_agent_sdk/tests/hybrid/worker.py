"""A disposable Worker process for the hybrid crash tests."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from temporalio.client import Client
from temporalio.worker import Worker
from tests.hybrid.activities import HybridActivities
from tests.hybrid.store import TranscriptStore
from tests.hybrid.workflows import HybridWorkflow


async def main() -> None:
    client = await Client.connect(os.environ["HYBRID_ADDRESS"])
    root = Path(os.environ["HYBRID_ROOT"])
    env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("ANTHROPIC", "CLAUDE", "DISABLE_", "NO_PROXY", "no_proxy"))
    }
    acts = HybridActivities(client, root, env, TranscriptStore(root / "store.db"))
    acts.recovery = os.environ.get("HYBRID_MAIN_RECOVERY") == "1"
    if os.environ.get("HYBRID_HOLD_DELIVERY"):
        acts.before_delivery = asyncio.Event()
    if os.environ.get("HYBRID_HOLD_REQUEST"):
        acts.before_request = asyncio.Event()
    if os.environ.get("HYBRID_HOLD_CHECKPOINT"):
        acts.after_checkpoint = asyncio.Event()
    async with Worker(
        client,
        task_queue=os.environ["HYBRID_QUEUE"],
        workflows=[HybridWorkflow],
        activities=[acts.burst, acts.tool],
    ):
        print("worker ready", flush=True)
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
