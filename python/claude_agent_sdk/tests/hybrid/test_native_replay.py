"""Recover batches and native failures without invoking a model in tool Activities."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest

from temporalio.client import Client, WorkflowFailureError
from temporalio.worker import Replayer, Worker
from tests.helpers.fake_messages_api import FakeMessagesAPI, engine_env, history_of
from tests.hybrid.native_replay import ReplayActivities
from tests.hybrid.replay_models import ReplayState
from tests.hybrid.replay_store import ReplayStore
from tests.hybrid.replay_workflow import NativeReplayWorkflow
from tests.hybrid.test_native_workspace import native_api, native_requests, stop_worker
from tests.hybrid.test_worker_loss import launch
from tests.hybrid.test_workflow import until

pytestmark = pytest.mark.timeout(120)


def batch_api(
    root: Path, mixed: bool = False
) -> tuple[FakeMessagesAPI, dict[str, str]]:
    emitted = (
        {"original-read": "Read", "original-edit": "Edit"}
        if mixed
        else {f"original-read-{i}": "Read" for i in range(3)}
    )

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, results = history_of(body)
        if results:
            assert {r.id for r in results} == set(emitted)
            assert not any(r.is_error for r in results)
            return [{"type": "text", "text": "BATCH DONE"}]
        calls = []
        for tid, name in emitted.items():
            arguments: dict[str, Any] = {
                "file_path": str(root / "workspace" / "note.txt")
            }
            if name == "Edit":
                arguments.update(old_string="BEFORE", new_string="AFTER")
            calls.append(
                {"type": "tool_use", "id": tid, "name": name, "input": arguments}
            )
        return calls

    return FakeMessagesAPI(policy, primary_tools={"Read", "Edit"}).start(), emitted


async def verify(
    handle: Any,
    state: ReplayState,
    store: ReplayStore,
    api: FakeMessagesAPI,
    emitted: dict[str, str],
    model_requests: int,
    failed: bool = False,
) -> None:
    assert set(state.results) == set(emitted) == {c.id for c in state.calls}
    assert len(state.calls) == len(emitted) == store.version()
    assert len(native_requests(api)) == model_requests and not api.errors
    _, _, results = history_of(native_requests(api)[-1])
    assert {r.id for r in results} == set(emitted)
    assert any(r.is_error for r in results) == failed
    with store.connect() as db:
        committed = [
            r[0]
            for r in db.execute("SELECT id FROM native_events WHERE phase='committed'")
        ]
        assert sorted(committed) == sorted(emitted)
        assert not db.execute(
            "SELECT 1 FROM native_events WHERE phase='continuation-refused'"
        ).fetchone()
        for call in state.calls:
            entry = json.loads(
                db.execute(
                    "SELECT data FROM replay_entries WHERE id=?", (call.id,)
                ).fetchone()[0]
            )
            assert entry["message"]["content"][0] == state.results[call.id]
            assert "toolUseResult" in entry
            store.receipt(call)
    history = await handle.fetch_history()
    schedules = [
        e.activity_task_scheduled_event_attributes.activity_id
        for e in history.events
        if e.HasField("activity_task_scheduled_event_attributes")
        and e.activity_task_scheduled_event_attributes.activity_type.name
        == "native_replay_execution"
    ]
    assert sorted(schedules) == sorted("tool-" + tid for tid in emitted)
    await Replayer(workflows=[NativeReplayWorkflow]).replay_workflow(history)


def edits_api(root: Path) -> tuple[FakeMessagesAPI, dict[str, str]]:
    emitted = {
        "original-read": "Read",
        "original-edit-1": "Edit",
        "original-edit-2": "Edit",
    }

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, results = history_of(body)
        path = str(root / "workspace" / "note.txt")
        if not results:
            return [
                {
                    "type": "tool_use",
                    "id": "original-read",
                    "name": "Read",
                    "input": {"file_path": path},
                }
            ]
        if len(results) == 1:
            return [
                {
                    "type": "tool_use",
                    "id": tid,
                    "name": "Edit",
                    "input": {"file_path": path, "old_string": old, "new_string": new},
                }
                for tid, old, new in [
                    ("original-edit-1", "BEFORE", "AFTER"),
                    ("original-edit-2", "AFTER", "FINAL"),
                ]
            ]
        assert {r.id for r in results} == set(emitted)
        assert not any(r.is_error for r in results)
        return [{"type": "text", "text": "EDIT DONE"}]

    return FakeMessagesAPI(policy, primary_tools={"Read", "Edit"}).start(), emitted


async def test_preparation_hook_failure_denies_native_edit(
    client: Client, tmp_path: Path
) -> None:
    store = ReplayStore(tmp_path, "Edit-preparation-hook-failure")
    store.initialize("BEFORE\n")
    api, _ = native_api(tmp_path)
    acts = ReplayActivities(tmp_path, engine_env(api, str(tmp_path / "cfg")), store)
    queue = "native-preparation-error-" + uuid.uuid4().hex
    session = str(uuid.uuid4())
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeReplayWorkflow],
            activities=[acts.decide, acts.execute],
        ):
            handle = await client.start_workflow(
                NativeReplayWorkflow.run,
                ReplayState(session),
                id=queue,
                task_queue=queue,
            )
            with pytest.raises(WorkflowFailureError):
                await asyncio.wait_for(handle.result(), 30)
            state = await handle.query(NativeReplayWorkflow.snapshot)
        assert len(state.calls) == len(state.results) == store.version() == 1
        assert state.calls[0].name == "Read"
        assert store.decision(session, 1) is None
        assert (
            store.durable_text()
            == (store.workspace / "note.txt").read_text()
            == "BEFORE\n"
        )
        assert len(native_requests(api)) == 3 and not api.errors
        assert {row["name"] for row in store.executions().values()} == {"Read"}
        await Replayer(workflows=[NativeReplayWorkflow]).replay_workflow(
            await handle.fetch_history()
        )
    finally:
        api.stop()


@pytest.mark.parametrize(
    "scenario", ["single", "validation", "parallel", "mixed", "publication", "edits"]
)
async def test_native_replay_boundaries(
    client: Client, tmp_path: Path, scenario: str
) -> None:
    store = ReplayStore(
        tmp_path, "parallel-proof" if scenario == "parallel" else "live"
    )
    store.initialize("BEFORE\n")
    store.fail_publication = scenario == "publication"
    api, emitted = (
        batch_api(tmp_path, scenario == "mixed")
        if scenario in {"parallel", "mixed"}
        else native_api(tmp_path)
    )
    if scenario == "edits":
        api.stop()
        api, emitted = edits_api(tmp_path)
    if scenario == "validation":
        original = api.decide

        def invalid(body: dict[str, Any]) -> list[dict[str, Any]]:
            blocks = original(body)
            for block in blocks:
                if block.get("name") == "Edit":
                    block["input"]["old_string"] = "MISSING"
            return blocks

        api.decide = invalid
    acts = ReplayActivities(tmp_path, engine_env(api, str(tmp_path / "cfg")), store)
    queue = "native-replay-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeReplayWorkflow],
            activities=[acts.decide, acts.execute],
        ):
            handle = await client.start_workflow(
                NativeReplayWorkflow.run,
                ReplayState(str(uuid.uuid4())),
                id=queue,
                task_queue=queue,
            )
            state = await asyncio.wait_for(handle.result(), 45)
        expected = (
            "BEFORE\n"
            if scenario in {"parallel", "validation"}
            else "FINAL\n"
            if scenario == "edits"
            else "AFTER\n"
        )
        assert (
            store.durable_text()
            == (store.workspace / "note.txt").read_text()
            == expected
        )
        assert state.answer == (
            "BATCH DONE"
            if scenario in {"parallel", "mixed"}
            else "NATIVE ERROR"
            if scenario == "validation"
            else "EDIT DONE"
        )
        await verify(
            handle,
            state,
            store,
            api,
            emitted,
            2 if scenario in {"parallel", "mixed"} else 3,
            scenario == "validation",
        )
        with store.connect() as db:
            cached = db.execute(
                "SELECT count(*) FROM native_events WHERE phase='cached-response'"
            ).fetchone()[0]
            assert cached == len(emitted) + int(scenario == "publication")
            assert (
                len((tmp_path / "cli-pids.jsonl").read_text().splitlines())
                == len(native_requests(api)) + cached
            )
            if scenario == "parallel":
                phases = [
                    r[0]
                    for r in db.execute("SELECT phase FROM native_events ORDER BY seq")
                ]
                assert phases[:3] == ["permitted"] * 3
        assert (store.workspace / "note.txt").stat().st_mode & 0o777 == 0o640
    finally:
        api.stop()


@pytest.mark.parametrize(
    "phase,invalid",
    [
        ("Read-before-execution", False),
        ("Edit-after-write", False),
        ("Edit-after-commit-before-completion", False),
        ("Read-after-checkpoint-before-completion", False),
        ("partial-batch", False),
        ("partial-edits", False),
        ("Edit-after-commit-before-completion", True),
    ],
)
async def test_native_replay_worker_loss(
    client: Client, address: str, tmp_path: Path, phase: str, invalid: bool
) -> None:
    store = ReplayStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = (
        batch_api(tmp_path) if phase == "partial-batch" else native_api(tmp_path)
    )
    if phase == "partial-edits":
        api.stop()
        api, emitted = edits_api(tmp_path)
    if invalid:
        original = api.decide

        def validation_error(body: dict[str, Any]) -> list[dict[str, Any]]:
            blocks = original(body)
            for block in blocks:
                if block.get("name") == "Edit":
                    block["input"]["old_string"] = "MISSING"
            return blocks

        api.decide = validation_error
    queue = "native-replay-loss-" + uuid.uuid4().hex
    session = str(uuid.uuid4())
    procs: list[subprocess.Popen[bytes]] = []
    try:
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                1,
                False,
                native_phase=phase,
                native_replay=True,
            )
        )
        handle = await client.start_workflow(
            NativeReplayWorkflow.run, ReplayState(session), id=queue, task_queue=queue
        )

        async def reached() -> Any:
            calls = store.decision(session, 0)
            if phase == "Read-after-checkpoint-before-completion":
                return calls
            snapshot = await handle.query(NativeReplayWorkflow.snapshot)
            if phase == "partial-batch":
                return snapshot if len(snapshot.results) == 1 else None
            if phase == "partial-edits":
                return snapshot if len(snapshot.results) == 2 else None
            wanted = (
                "executed"
                if phase.endswith("after-write")
                else "committed"
                if "after-commit" in phase
                else "held-before-execution"
            )
            return (
                snapshot
                if any(
                    c.name == phase.split("-")[0]
                    and (e := store.execution(c.id))
                    and e["phase"] == wanted
                    for c in snapshot.calls
                )
                else None
            )

        await until(reached)
        accepted = store.decision(session, 0)
        assert accepted is not None
        if phase.endswith("after-write"):
            assert (store.workspace / "note.txt").read_text() == "AFTER\n"
            assert store.durable_text() == "BEFORE\n"
        stop_worker(procs[0], tmp_path)
        shutil.rmtree(store.workspace)
        shutil.rmtree(tmp_path / "machine-1", ignore_errors=True)
        for path in tmp_path.glob("*.db"):
            if path != store.path:
                path.unlink()
        for path in tmp_path.glob("cache-*"):
            shutil.rmtree(path)
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                2,
                False,
                native_phase="replacement",
                native_replay=True,
            )
        )
        state = await asyncio.wait_for(handle.result(), 45)
        assert store.decision(session, 0) == accepted
        await verify(
            handle,
            state,
            store,
            api,
            emitted,
            2 if phase == "partial-batch" else 3,
            failed=invalid,
        )
        assert store.durable_text() == (
            "BEFORE\n"
            if phase == "partial-batch" or invalid
            else "FINAL\n"
            if phase == "partial-edits"
            else "AFTER\n"
        )
        with store.connect() as db:
            executions = db.execute(
                "SELECT count(*) FROM native_events WHERE phase='executed'"
            ).fetchone()[0]
            assert executions == len(emitted) + int(phase.endswith("after-write"))
    finally:
        for proc in procs:
            stop_worker(proc, tmp_path)
        api.stop()
