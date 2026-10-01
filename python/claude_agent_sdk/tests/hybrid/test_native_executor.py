"""Checkpointed original native tool execution across Worker and workspace loss."""

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
from tests.hybrid.executor_models import NativeRun
from tests.hybrid.executor_store import ExecutionStore
from tests.hybrid.executor_workflow import NativeExecutionWorkflow
from tests.hybrid.native_executor import CheckpointActivities
from tests.hybrid.test_native_workspace import native_api, native_requests, stop_worker
from tests.hybrid.test_worker_loss import launch
from tests.hybrid.test_workflow import until

pytestmark = pytest.mark.timeout(120)


async def verify(
    handle: Any,
    state: NativeRun,
    api: Any,
    emitted: dict[str, str],
    store: ExecutionStore,
    model_requests: int = 3,
) -> None:
    assert state.answer == "EDIT DONE"
    assert set(state.results) == set(emitted) == {i.id for i in state.intents}
    assert sorted(emitted.values()) == ["Edit", "Read"]
    assert (
        store.durable_text() == (store.workspace / "note.txt").read_text() == "AFTER\n"
    )
    assert (store.workspace / "note.txt").stat().st_mode & 0o777 == 0o640
    assert store.version() == 2
    assert len(native_requests(api)) == model_requests
    _, _, results = history_of(native_requests(api)[-1])
    assert {h.id for h in results} == set(emitted)
    assert not any(h.is_error for h in results)
    for tid, block in state.results.items():
        assert store.execution(tid)["result"] == block  # type: ignore[index]
        assert "file had been modified on disk" not in block["content"]
    with store.connect() as db:
        guarded = {
            r[0]
            for r in db.execute(
                "SELECT id FROM native_events WHERE phase='continuation-refused'"
            )
        }
    # A Worker can die after publication but before logging the refused
    # continuation. Its cached retry needs no executor or gateway.
    assert guarded and guarded.issubset(state.results)
    assert not api.errors
    history = await handle.fetch_history()
    schedules = [
        e.activity_task_scheduled_event_attributes.activity_id
        for e in history.events
        if e.HasField("activity_task_scheduled_event_attributes")
        and e.activity_task_scheduled_event_attributes.activity_type.name
        == "native_execution"
    ]
    assert set(schedules) == {"tool-" + tid for tid in emitted}
    assert len(schedules) == 2
    await Replayer(workflows=[NativeExecutionWorkflow]).replay_workflow(history)


async def test_checkpointed_native_execution(client: Client, tmp_path: Path) -> None:
    store = ExecutionStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = native_api(tmp_path)
    acts = CheckpointActivities(tmp_path, engine_env(api, str(tmp_path / "cfg")), store)
    queue = "native-executor-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeExecutionWorkflow],
            activities=[acts.decide, acts.execute],
        ):
            handle = await client.start_workflow(
                NativeExecutionWorkflow.run,
                NativeRun(str(uuid.uuid4())),
                id=queue,
                task_queue=queue,
            )
            state = await asyncio.wait_for(handle.result(), 45)
        await verify(handle, state, api, emitted, store)
        assert len((tmp_path / "cli-pids.jsonl").read_text().splitlines()) == 5
    finally:
        api.stop()


async def test_native_validation_before_hook_is_durable(
    client: Client, tmp_path: Path
) -> None:
    store = ExecutionStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = native_api(tmp_path)
    policy = api.decide

    def fail_edit(body: dict[str, Any]) -> list[dict[str, Any]]:
        blocks = policy(body)
        for block in blocks:
            if block.get("name") == "Edit":
                block["input"]["old_string"] = "MISSING"
        return blocks

    api.decide = fail_edit
    acts = CheckpointActivities(tmp_path, engine_env(api, str(tmp_path / "cfg")), store)
    queue = "native-error-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeExecutionWorkflow],
            activities=[acts.decide, acts.execute],
        ):
            handle = await client.start_workflow(
                NativeExecutionWorkflow.run,
                NativeRun(str(uuid.uuid4())),
                id=queue,
                task_queue=queue,
            )
            state = await asyncio.wait_for(handle.result(), 45)
        assert state.answer == "NATIVE ERROR"
        assert set(state.results) == set(emitted)
        assert len(state.results) == 2
        failed = next(i for i in state.intents if i.outcome_kind == "validation")
        assert failed.name == "Edit" and state.results[failed.id]["is_error"]
        assert store.execution(failed.id)["result"] == state.results[failed.id]  # type: ignore[index]
        assert store.frozen(failed) and store.version() == 2
        store.freeze(failed, store.frozen(failed))
        assert store.version() == 2
        assert (
            store.durable_text()
            == (store.workspace / "note.txt").read_text()
            == "BEFORE\n"
        )
        assert len(native_requests(api)) == 3 and not api.errors
        await Replayer(workflows=[NativeExecutionWorkflow]).replay_workflow(
            await handle.fetch_history()
        )
    finally:
        api.stop()


@pytest.mark.parametrize(
    "phase",
    ["Edit-after-checkpoint-before-completion", "Edit-after-commit-before-completion"],
)
async def test_native_validation_outcome_survives_worker_loss(
    client: Client, address: str, tmp_path: Path, phase: str
) -> None:
    store = ExecutionStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = native_api(tmp_path)
    policy = api.decide

    def fail_edit(body: dict[str, Any]) -> list[dict[str, Any]]:
        blocks = policy(body)
        for block in blocks:
            if block.get("name") == "Edit":
                block["input"]["old_string"] = "MISSING"
        return blocks

    api.decide = fail_edit
    queue = "native-validation-loss-" + uuid.uuid4().hex
    procs: list[subprocess.Popen[bytes]] = []
    session = str(uuid.uuid4())
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
                native_executor=True,
            )
        )
        handle = await client.start_workflow(
            NativeExecutionWorkflow.run, NativeRun(session), id=queue, task_queue=queue
        )

        async def reached() -> Any:
            intent = store.decision(session, 1)
            if intent is None:
                return None
            snapshot = await handle.query(NativeExecutionWorkflow.snapshot)
            if "after-checkpoint" in phase:
                return intent if len(snapshot.intents) == 1 else None
            return intent if len(snapshot.intents) == 2 else None

        intent = await until(reached)
        frozen = store.frozen(intent)
        outcome = store.execution(intent.id)
        assert outcome is not None and outcome["result"]["is_error"]
        stop_worker(procs[0], tmp_path)
        shutil.rmtree(store.workspace)
        shutil.rmtree(tmp_path / "machine-1", ignore_errors=True)
        for path in tmp_path.glob("*.db"):
            if path != store.path:
                path.unlink()
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                2,
                False,
                native_phase="replacement",
                native_executor=True,
            )
        )
        state = await asyncio.wait_for(handle.result(), 45)
        assert state.answer == "NATIVE ERROR"
        assert set(state.results) == set(emitted)
        assert state.intents[-1] == intent
        assert state.results[intent.id] == outcome["result"]
        assert store.frozen(intent) == frozen and store.version() == 2
        assert (
            store.durable_text()
            == (store.workspace / "note.txt").read_text()
            == "BEFORE\n"
        )
        assert len(native_requests(api)) == 3 and not api.errors
        with store.connect() as db:
            assert (
                db.execute(
                    "SELECT count(*) FROM native_events WHERE id=? AND phase='committed'",
                    (intent.id,),
                ).fetchone()[0]
                == 1
            )
        await Replayer(workflows=[NativeExecutionWorkflow]).replay_workflow(
            await handle.fetch_history()
        )
    finally:
        for proc in procs:
            stop_worker(proc, tmp_path)
        api.stop()


async def test_failed_native_publication_retries_from_checkpoint(
    client: Client, tmp_path: Path
) -> None:
    store = ExecutionStore(tmp_path)
    store.initialize("BEFORE\n")
    store.fail_publication = True
    api, emitted = native_api(tmp_path)
    acts = CheckpointActivities(tmp_path, engine_env(api, str(tmp_path / "cfg")), store)
    queue = "native-publication-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeExecutionWorkflow],
            activities=[acts.decide, acts.execute],
        ):
            handle = await client.start_workflow(
                NativeExecutionWorkflow.run,
                NativeRun(str(uuid.uuid4())),
                id=queue,
                task_queue=queue,
            )
            state = await asyncio.wait_for(handle.result(), 45)
        await verify(handle, state, api, emitted, store)
        edit_id = next(tid for tid, name in emitted.items() if name == "Edit")
        with store.connect() as db:
            assert (
                db.execute(
                    "SELECT count(*) FROM native_events WHERE id=? AND phase='executed'",
                    (edit_id,),
                ).fetchone()[0]
                == 2
            )
            assert (
                db.execute(
                    "SELECT count(*) FROM native_events WHERE id=? AND phase='committed'",
                    (edit_id,),
                ).fetchone()[0]
                == 1
            )
    finally:
        api.stop()


async def test_native_checkpoint_rejects_batches_before_execution(
    client: Client, tmp_path: Path
) -> None:
    store = ExecutionStore(tmp_path)
    store.initialize("BEFORE\n")

    def policy(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if history:
            return [{"type": "text", "text": "BATCH DENIED"}]
        path = str(store.workspace / "note.txt")
        return [
            {
                "type": "tool_use",
                "id": "original-read",
                "name": "Read",
                "input": {"file_path": path},
            },
            {
                "type": "tool_use",
                "id": "original-edit",
                "name": "Edit",
                "input": {
                    "file_path": path,
                    "old_string": "BEFORE",
                    "new_string": "AFTER",
                },
            },
        ]

    api = FakeMessagesAPI(policy, primary_tools={"Read", "Edit"}).start()
    acts = CheckpointActivities(tmp_path, engine_env(api, str(tmp_path / "cfg")), store)
    queue = "native-batch-" + uuid.uuid4().hex
    try:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[NativeExecutionWorkflow],
            activities=[acts.decide, acts.execute],
        ):
            handle = await client.start_workflow(
                NativeExecutionWorkflow.run,
                NativeRun(str(uuid.uuid4())),
                id=queue,
                task_queue=queue,
            )
            with pytest.raises(WorkflowFailureError):
                await asyncio.wait_for(handle.result(), 30)
            assert (await handle.query(NativeExecutionWorkflow.snapshot)).intents == []
        assert (
            store.durable_text()
            == (store.workspace / "note.txt").read_text()
            == "BEFORE\n"
        )
        assert store.executions() == {}
        with store.connect() as db:
            assert (
                db.execute("SELECT count(*) FROM native_checkpoints").fetchone()[0] == 0
            )
        assert not api.errors
        await Replayer(workflows=[NativeExecutionWorkflow]).replay_workflow(
            await handle.fetch_history()
        )
    finally:
        api.stop()


async def test_unpublished_native_checkpoint_is_discarded_after_worker_loss(
    client: Client, address: str, tmp_path: Path
) -> None:
    store = ExecutionStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = native_api(tmp_path)
    queue = "native-unpublished-" + uuid.uuid4().hex
    procs: list[subprocess.Popen[bytes]] = []
    session = str(uuid.uuid4())
    try:
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                1,
                False,
                native_phase="Read-before-checkpoint-publication",
                native_executor=True,
            )
        )
        handle = await client.start_workflow(
            NativeExecutionWorkflow.run, NativeRun(session), id=queue, task_queue=queue
        )

        async def unpublished() -> Any:
            with store.connect() as db:
                row = db.execute(
                    "SELECT id FROM native_events WHERE phase='checkpoint-unpublished'"
                ).fetchone()
            return row[0] if row else None

        lost_id = await until(unpublished)
        assert (await handle.query(NativeExecutionWorkflow.snapshot)).intents == []
        assert store.decision(session, 0) is None
        assert await store.load(store.key(session)) is None
        stop_worker(procs[0], tmp_path)
        shutil.rmtree(store.workspace)
        shutil.rmtree(tmp_path / "machine-1", ignore_errors=True)
        for path in tmp_path.glob("*.db"):
            if path != store.path:
                path.unlink()
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                2,
                False,
                native_phase="replacement",
                native_executor=True,
            )
        )
        state = await asyncio.wait_for(handle.result(), 45)
        assert lost_id not in state.results
        assert store.execution(lost_id) is None
        await verify(
            handle,
            state,
            api,
            {i.id: emitted[i.id] for i in state.intents},
            store,
            model_requests=4,
        )
    finally:
        for proc in procs:
            stop_worker(proc, tmp_path)
        api.stop()


async def test_native_checkpoint_receipt_survives_worker_loss(
    client: Client, address: str, tmp_path: Path
) -> None:
    store = ExecutionStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = native_api(tmp_path)
    queue = "native-checkpoint-loss-" + uuid.uuid4().hex
    procs: list[subprocess.Popen[bytes]] = []
    session = str(uuid.uuid4())
    try:
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                1,
                False,
                native_phase="Read-after-checkpoint-before-completion",
                native_executor=True,
            )
        )
        handle = await client.start_workflow(
            NativeExecutionWorkflow.run, NativeRun(session), id=queue, task_queue=queue
        )

        async def checkpointed() -> Any:
            return store.decision(session, 0)

        intent = await until(checkpointed)
        assert (await handle.query(NativeExecutionWorkflow.snapshot)).intents == []
        stop_worker(procs[0], tmp_path)
        shutil.rmtree(store.workspace)
        shutil.rmtree(tmp_path / "machine-1", ignore_errors=True)
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                2,
                False,
                native_phase="replacement",
                native_executor=True,
            )
        )
        state = await asyncio.wait_for(handle.result(), 45)
        assert state.intents[0] == intent
        await verify(handle, state, api, emitted, store)
    finally:
        for proc in procs:
            stop_worker(proc, tmp_path)
        api.stop()


@pytest.mark.parametrize(
    "phase",
    [
        "Read-before-execution",
        "Edit-before-execution",
        "Edit-after-write",
        "Read-after-commit-before-completion",
        "Edit-after-commit-before-completion",
    ],
)
async def test_checkpointed_native_executor_survives_worker_loss(
    client: Client, address: str, tmp_path: Path, phase: str
) -> None:
    store = ExecutionStore(tmp_path)
    store.initialize("BEFORE\n")
    api, emitted = native_api(tmp_path)
    queue = "native-executor-loss-" + uuid.uuid4().hex
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
                native_executor=True,
            )
        )
        handle = await client.start_workflow(
            NativeExecutionWorkflow.run,
            NativeRun(str(uuid.uuid4())),
            id=queue,
            task_queue=queue,
        )

        async def held() -> Any:
            snapshot = await handle.query(NativeExecutionWorkflow.snapshot)
            wanted = (
                "executed"
                if phase.endswith("after-write")
                else "committed"
                if "after-commit" in phase
                else "held-before-execution"
            )
            for intent in snapshot.intents:
                row = store.execution(intent.id)
                if (
                    intent.name == phase.split("-")[0]
                    and row
                    and row["phase"] == wanted
                ):
                    return snapshot, intent
            return None

        before, intent = await until(held)
        frozen = store.frozen(intent)
        if phase.endswith("after-write"):
            assert (store.workspace / "note.txt").read_text() == "AFTER\n"
            assert store.durable_text() == "BEFORE\n"
        stop_worker(procs[0], tmp_path)
        shutil.rmtree(store.workspace)
        shutil.rmtree(tmp_path / "machine-1", ignore_errors=True)
        # Delete all partial attempt transcripts; only the canonical service
        # and its immutable engine-generated checkpoint survive.
        for path in tmp_path.glob("*.db"):
            if path != store.path:
                path.unlink()
        shutil.rmtree(tmp_path / "native-executor-config", ignore_errors=True)
        procs.append(
            await launch(
                address,
                queue,
                tmp_path,
                api,
                2,
                False,
                native_phase="replacement",
                native_executor=True,
            )
        )
        state = await asyncio.wait_for(handle.result(), 45)
        assert {i.id for i in before.intents}.issubset(state.results)
        assert store.frozen(intent) == frozen
        await verify(handle, state, api, emitted, store)
        with store.connect() as db:
            committed = [
                r[0]
                for r in db.execute(
                    "SELECT id FROM native_events WHERE phase='committed'"
                )
            ]
            executed = [
                r[0]
                for r in db.execute(
                    "SELECT id FROM native_events WHERE phase='executed'"
                )
            ]
        assert sorted(committed) == sorted(emitted)
        assert executed.count(intent.id) == (2 if phase.endswith("after-write") else 1)
        print(
            "NATIVE_EXECUTOR_RECOVERY "
            + json.dumps(
                {
                    "phase": phase,
                    "original_id": intent.id,
                    "committed_versions": store.version(),
                }
            ),
            flush=True,
        )
    finally:
        for proc in procs:
            stop_worker(proc, tmp_path)
        api.stop()
