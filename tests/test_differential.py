"""Differential check: when a private event moves to another day, nothing that leaves the device may change.

The leak test looks for private values in what is sent; this looks for what a cloud model or an MCP client
could infer by comparison. Each question is asked against the demo persona and against a copy where the two
private appointments are on other days (and one later in the day). The payloads sent to the fake model
(compared line by line, since full-text ranking order may shift), the drafts that may be passed on, the
delivered text, the guard summary and the MCP output must be identical. Found by an independent review on
27 October 2026: before the fix, 18 of 30 question pairs differed through which placeholders the context held.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from confidant import Assistant, Budget, Guard, MemoryStore, ReceiptLog
from confidant.assistant import delivery_text
from confidant.mcp_server import MCPServer

from conftest import MODELS, PERSONA, FakeLLM, FakeTavily

MOVED = {
    7: "Dermatologist appointment on Monday 2 Nov 2026 at 16:30.",
    11: "Confidential: final-round interview for a lead designer role on Friday 6 Nov 2026 at 11:00 (video call); "
        "not to be mentioned at work.",
}
QUESTIONS = [
    # availability, checked on the device
    "Am I free Tuesday morning?", "Am I free on Thursday at 11:00?", "Is the 1030 slot on Tuesday still open?",
    "Am I still free on Tuesday?", "Got anything on Thursday?", "Is the afternoon free on Monday?",
    "When am I free next week?", "Suis-je libre mardi matin ?", "Am I free between Monday and Friday?",
    # questions for the model that name a time
    "What do I have on Tuesday?", "What's on Tuesday morning?", "What's on Thursday?", "Any plans for Monday?",
    "What's on my plate next week?", "What happens on the 3rd?", "Anything on 5/11?",
]


def _store(tmp_path: Path, moves: dict[int, str]) -> MemoryStore:
    data = json.loads(PERSONA.read_text(encoding="utf-8"))
    for memory in data["memories"]:
        memory["text"] = moves.get(memory["id"], memory["text"])
    path = tmp_path / f"persona-{len(moves)}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = MemoryStore()
    store.load_seed(path)
    return store


def _lines(sent: str) -> list[list[tuple[str, int]]]:
    """Each call's user text as a multiset of lines."""
    calls = json.loads(sent or "[]")
    return [sorted(Counter(line for message in call if message["role"] == "user"
                           for line in message["content"].splitlines()).items()) for call in calls]


def _observe(store: MemoryStore, question: str) -> dict:
    llm, tavily = FakeLLM(), FakeTavily()
    assistant = Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), llm, search=tavily, models=MODELS)
    turn = assistant.handle(question, session_id="d")
    chat = {"intent": turn.intent, "draft": turn.draft, "delivery": delivery_text(turn, store),
            "guard": turn.guard.summary(), "sent": _lines(llm.sent_text()), "tavily": tavily.queries}
    mcp_llm = FakeLLM()
    mcp_assistant = Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), mcp_llm, models=MODELS)
    response = MCPServer(mcp_assistant).handle_message({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "guarded_answer", "arguments": {"question": question}},
    })
    result = response["result"]
    mcp = result.get("structuredContent", result)
    return {"chat": chat, "mcp": mcp, "mcp_sent": _lines(mcp_llm.sent_text())}


@pytest.mark.parametrize("question", QUESTIONS)
def test_moving_a_private_event_changes_nothing_that_leaves(tmp_path: Path, question: str) -> None:
    original = _observe(_store(tmp_path, {}), question)
    moved = _observe(_store(tmp_path, MOVED), question)
    assert original == moved
