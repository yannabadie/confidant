"""Round 7: a second outside code review of 28 October 2026 (a Claude agent with no history of the project) read
the round-6 changes and found what they left open. Each finding is reproduced here before its fix.
"""
from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from confidant import Assistant, Budget, Guard, MemoryStore, ReceiptLog, ReplayBook
from confidant.mcp_server import MCPServer

from conftest import MODELS, PERSONA, FakeLLM, Rig

MOVED = {  # the two private appointments on other days, as in tests/test_differential.py
    7: "Dermatologist appointment on Monday 2 Nov 2026 at 16:30.",
    11: "Confidential: final-round interview for a lead designer role on Friday 6 Nov 2026 at 11:00 (video call); "
        "not to be mentioned at work.",
}


def seeded(tmp_path: Path, moves: dict[int, str] | None = None) -> MemoryStore:
    data = json.loads(PERSONA.read_text(encoding="utf-8"))
    for memory in data["memories"]:
        memory["text"] = (moves or {}).get(memory["id"], memory["text"])
    path = tmp_path / f"persona-{len(moves or {})}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = MemoryStore()
    store.load_seed(path)
    return store


def server(store: MemoryStore, llm: FakeLLM | None = None, *, live: bool = True) -> MCPServer:
    return MCPServer(Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), (llm or FakeLLM()) if live else None,
                               replay=None if live else ReplayBook(), models=MODELS if live else None))


def tool(mcp: MCPServer, name: str, **arguments: Any) -> dict[str, Any]:
    response = mcp.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": name, "arguments": arguments}})
    result = response["result"]
    assert result["isError"] is False, result
    return result["structuredContent"]


def nebius_requests(llm: FakeLLM) -> list[str]:
    return [message["content"] for call in llm.calls for message in call["messages"] if message["role"] == "user"]


# --- Critical 1: private matches used up the agent's search slots -------------------------------------------

FILLER = "every dinner product"
PROBE_WORDS = ["dermatologist", "interview", "sabbatical", "rent", "dentist", "podcast", "salary"]


def test_an_agents_answer_lists_the_same_memories_whatever_words_private_memories_hold(tmp_path: Path) -> None:
    listed = {}
    for word in PROBE_WORDS:
        answer = tool(server(seeded(tmp_path), live=False), "guarded_answer",
                      question=f"Any notes on {word}, {FILLER}?")["answer"]
        listed[word] = [line for line in answer.splitlines() if line.startswith("- ")]
    assert len({len(lines) for lines in listed.values()}) == 1, {word: len(lines) for word, lines in listed.items()}


def test_the_context_sent_for_an_agent_does_not_depend_on_which_words_private_memories_hold(tmp_path: Path) -> None:
    contexts = set()
    for word in ("dermatologist", "dentist"):
        llm = FakeLLM()
        tool(server(seeded(tmp_path), llm), "guarded_answer", question=f"Any notes on {word}, {FILLER}?")
        context = nebius_requests(llm)[-1]
        contexts.add(tuple(line.split("]")[0] for line in context.splitlines() if line.startswith("[m")))
    assert len(contexts) == 1, contexts


def test_an_agent_cannot_tell_a_private_date_by_how_its_question_is_sent(tmp_path: Path) -> None:
    """5 Nov holds the private interview, 12 Nov nothing private: the request sent for each must differ only
    in the period the agent named."""
    sent = {}
    for day in ("Thursday 5 Nov", "Thursday 12 Nov"):
        llm = FakeLLM()
        tool(server(seeded(tmp_path), llm), "guarded_answer", question=f"What's on my plate on {day}?")
        sent[day] = nebius_requests(llm)[-1]
    assert "[withheld]" not in sent["Thursday 5 Nov"]
    assert all("Period asked about" in text for text in sent.values()), sent
    normalised = {text.replace("5 November", "<day>").replace("12 November", "<day>") for text in sent.values()}
    assert len(normalised) == 1, sent


# --- Critical 2: memory_search ranked shared results with statistics that include private rows --------------

RANK_WORDS = ["tuesday", "thursday", "friday", "monday", "nov", "every", "dinner", "design", "lead", "work",
              "video", "role", "call", "appointment", "review", "team"]


def test_moving_private_events_changes_no_ranking_of_shared_results(tmp_path: Path) -> None:
    original, moved = server(seeded(tmp_path)), server(seeded(tmp_path, MOVED))
    differing = []
    for first, second in itertools.combinations(RANK_WORDS, 2):
        for limit in (1, 8):
            query = f"{first} {second}"
            a = [m["cite"] for m in tool(original, "memory_search", query=query, limit=limit)["memories"]]
            b = [m["cite"] for m in tool(moved, "memory_search", query=query, limit=limit)["memories"]]
            if a != b:
                differing.append((query, limit, a, b))
    assert differing == []


def test_the_top_hit_does_not_say_which_weekday_a_private_memory_names(tmp_path: Path) -> None:
    a = tool(server(seeded(tmp_path)), "memory_search", query="tuesday friday", limit=1)["memories"]
    b = tool(server(seeded(tmp_path, MOVED)), "memory_search", query="tuesday friday", limit=1)["memories"]
    assert [m["cite"] for m in a] == [m["cite"] for m in b]


# --- Important 3: a private search said its words stayed on the device, then sent them to Nebius -----------

PRIVATE_SEARCHES = ["Search for tips for my final-round lead designer interview on Thursday.",
                    "Look up what to ask the dermatologist on Tuesday about a mole that changed."]


@pytest.mark.parametrize("message", PRIVATE_SEARCHES)
def test_a_private_search_says_what_left_and_masks_its_values(make_rig: Callable[..., Rig], message: str) -> None:
    rig = make_rig()
    turn = rig.ask(message)
    assert rig.tavily.queries == []
    said = " ".join([*turn.actions, turn.notice])
    assert "stay on this device" not in said and "Nemotron" in turn.notice
    request = nebius_requests(rig.llm)[-1].splitlines()[-1]
    assert not any(day in request for day in ("Tuesday", "Thursday")), request


# --- Important 4: a correction made over MCP never applied the "never less private" rule -------------------

DIABETES = "Partner Sam is vegetarian and was just diagnosed with diabetes; he starts insulin next week."


def test_a_correction_over_mcp_that_looks_private_makes_the_memory_private(tmp_path: Path) -> None:
    llm = FakeLLM()
    mcp = server(seeded(tmp_path), llm)
    view = tool(mcp, "memory_correct", memory_id=12, new_text=DIABETES, expected_version=1)["memory"]
    memory = mcp.assistant.store.get(12)
    assert memory.sensitivity == "private" and memory.label == "health"
    assert "diabetes" not in view["text"] and view["withheld"] is True
    mcp.assistant.handle("What should I cook for Sam this weekend?", session_id="s2")
    assert all("diabetes" not in text for text in nebius_requests(llm))


def test_a_harmless_correction_over_mcp_keeps_the_level_and_label(tmp_path: Path) -> None:
    mcp = server(seeded(tmp_path))
    tool(mcp, "memory_correct", memory_id=12, new_text="Partner Sam is vegan now, so dinner plans need a vegan option.",
         expected_version=1)
    memory = mcp.assistant.store.get(12)
    assert (memory.sensitivity, memory.label) == ("personal", "partner's diet")


# --- Important 5: new everyday words made sensitive facts shareable again -----------------------------------

@pytest.mark.parametrize("fact", [
    "Brunch with my sponsor after Sunday's meeting.", "Cooking class at the women's shelter on Monday.",
    "I enjoy a bottle of wine every night.", "Wedding is off; Sam moved out.",
])
def test_a_fact_that_was_private_before_round_6_is_private_again(fact: str) -> None:
    from confidant.rules import classify_new_fact
    assert classify_new_fact(fact) == "private"
