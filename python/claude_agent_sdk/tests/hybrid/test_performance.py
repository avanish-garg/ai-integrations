"""Opt-in measurements of three runs of twenty tool rounds, without timing assertions."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import time
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pytest

from temporalio.claude_agent_sdk import ClaudeAgentPlugin, ClaudeAgentSdkRunner
from temporalio.client import Client
from temporalio.worker import Worker
from tests.helpers.fake_messages_api import engine_env
from tests.hybrid.benchmark_activity import segment_echo
from tests.hybrid.benchmark_workflow import SegmentBenchmarkWorkflow
from tests.hybrid.models import State
from tests.hybrid.policy import model_requests, rounds
from tests.hybrid.store import TranscriptStore
from tests.hybrid.test_workflow import runtime, worker
from tests.hybrid.workflows import HybridWorkflow

pytestmark = [
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        os.environ.get("HYBRID_BENCHMARK") != "1",
        reason="opt-in performance measurement",
    ),
]


class CountingRunner(ClaudeAgentSdkRunner):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.starts = 0

    async def _run_engine(self, *args: Any, **kwargs: Any) -> Any:
        self.starts += 1
        return await super()._run_engine(*args, **kwargs)


async def test_twenty_rounds_three_runs(client: Client, tmp_path: Path) -> None:
    measurements: list[dict[str, Any]] = []
    for trial in range(3):
        for mode in ("segment", "hybrid"):
            root = tmp_path / f"{mode}-{trial}"
            root.mkdir()
            api = rounds(20)
            times: list[float] = []
            decide = api.decide

            def timed(body: dict[str, Any]) -> list[dict[str, Any]]:
                times.append(time.monotonic())
                return decide(body)

            api.decide = timed
            queue = f"bench-{mode}-{uuid.uuid4().hex}"
            started = time.monotonic()
            try:
                if mode == "segment":
                    runner = CountingRunner(
                        session_store=cast(Any, TranscriptStore(root / "store.db")),
                        cwd=str(root),
                        env=engine_env(api, str(root / "cfg")),
                    )
                    async with Worker(
                        client,
                        task_queue=queue,
                        workflows=[SegmentBenchmarkWorkflow],
                        activities=[segment_echo],
                        plugins=[ClaudeAgentPlugin(runner)],
                    ):
                        answer = await client.execute_workflow(
                            SegmentBenchmarkWorkflow.run,
                            "work",
                            id=queue,
                            task_queue=queue,
                        )
                    starts = runner.starts
                    cli = await runner._engine_version()
                else:
                    acts = runtime(client, root, api)
                    async with worker(client, queue, acts):
                        state = await client.execute_workflow(
                            HybridWorkflow.run,
                            State(str(uuid.uuid4()), ["work"]),
                            id=queue,
                            task_queue=queue,
                        )
                    answer = state.answers[0]
                    starts = len(acts.bursts)
                    cli = acts.bursts[0].version
                elapsed = time.monotonic() - started
                assert (
                    answer == "DONE 20" and len(times) == len(model_requests(api)) == 21
                )
                latencies = [1000 * (b - a) for a, b in zip(times, times[1:])]
                measurements.append(
                    {
                        "mode": mode,
                        "trial": trial + 1,
                        "rounds": 20,
                        "process_starts": starts,
                        "model_requests": len(times),
                        "wall_seconds": elapsed,
                        "round_ms": latencies,
                        "median_ms": statistics.median(latencies),
                        "p95_ms": sorted(latencies)[
                            math.ceil(0.95 * len(latencies)) - 1
                        ],
                        "cli": cli,
                    }
                )
                assert api.errors == []
            finally:
                api.stop()
    report = {
        "sdk": version("claude-agent-sdk"),
        "temporalio": version("temporalio"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "measurements": measurements,
    }
    target = Path(
        os.environ.get("HYBRID_BENCHMARK_OUT", str(tmp_path / "benchmark.json"))
    )
    target.write_text(json.dumps(report, indent=2) + "\n")
    print("HYBRID_BENCHMARK " + str(target), flush=True)
