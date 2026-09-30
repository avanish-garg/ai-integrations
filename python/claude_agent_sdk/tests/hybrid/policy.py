"""Scripted real-engine responses used by capability and performance probes."""

from __future__ import annotations

from typing import Any

from tests.helpers.fake_messages_api import FakeMessagesAPI, history_of


def model_requests(api: FakeMessagesAPI) -> list[dict[str, Any]]:
    return [
        r
        for r in api.requests
        if any(t["name"].startswith("mcp__durable__") for t in r.get("tools", []))
    ]


def rounds(
    round_count: int, batch: int = 1, *, approval: bool = False, delay: float = 0
) -> FakeMessagesAPI:
    holder: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, history = history_of(body)
        if len(history) >= round_count * batch:
            return [{"type": "text", "text": f"DONE {len(history)}"}]
        n = len(history)
        return [
            holder[0].tool_use(
                "echo", {"n": n + i, "approval": approval, "delay": delay}
            )
            for i in range(batch)
        ]

    api = FakeMessagesAPI(decide)
    holder.append(api)
    return api.start()


def chat() -> FakeMessagesAPI:
    holder: list[FakeMessagesAPI] = []

    def decide(body: dict[str, Any]) -> list[dict[str, Any]]:
        _, texts, history = history_of(body)
        wanted = next(
            (
                int(t.removeprefix("round "))
                for t in reversed(texts)
                if t.startswith("round ")
            ),
            0,
        )
        if any(h.input.get("n") == wanted and not h.is_error for h in history):
            return [
                {"type": "text", "text": f"DONE {wanted}; remembered {len(history)}"}
            ]
        return [holder[0].tool_use("echo", {"n": wanted})]

    api = FakeMessagesAPI(decide)
    holder.append(api)
    return api.start()
