"""Round 6: findings of two independent reviews of 28 October 2026 (a judge's read and a five-axis code review),
each reproduced here before its fix. The earlier rounds looked at what the chat sends to Nebius; these look at
the MCP tools, at how new facts and corrections are classified, at the web query of a private message, at the
schedule rule, and at what the user reads after late binding.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from confidant import Assistant, Budget, Guard, MemoryStore, ReceiptLog
from confidant.assistant import delivery_text
from confidant.mcp_server import MCPServer
from confidant.rules import classify_new_fact

from conftest import MODELS, PERSONA, FakeLLM, FakeTavily, Rig

CATEGORY_LABELS = {
    "health", "health appointment", "finances", "career", "career appointment", "legal or official matter",
    "personal matter", "private matter",
}


def tool(server: MCPServer, name: str, **arguments: Any) -> dict[str, Any]:
    response = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                      "params": {"name": name, "arguments": arguments}})
    result = response["result"]
    assert result["isError"] is False, result
    return result["structuredContent"]


def seeded(tmp_path: Path, moves: dict[int, str] | None = None) -> MemoryStore:
    data = json.loads(PERSONA.read_text(encoding="utf-8"))
    for memory in data["memories"]:
        memory["text"] = (moves or {}).get(memory["id"], memory["text"])
    path = tmp_path / f"persona-{len(moves or {})}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = MemoryStore()
    store.load_seed(path)
    return store


def mcp(store: MemoryStore, llm: FakeLLM | None = None) -> MCPServer:
    return MCPServer(Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), llm or FakeLLM(),
                               models=MODELS))


MOVED = {  # the two private appointments on other days, as in tests/test_differential.py
    7: "Dermatologist appointment on Monday 2 Nov 2026 at 16:30.",
    11: "Confidential: final-round interview for a lead designer role on Friday 6 Nov 2026 at 11:00 (video call); "
        "not to be mentioned at work.",
}


# --- MCP: memory_search was an oracle for the words of private memories ------------------------------------

PROBES = ["interview", "designer", "lead", "confidential", "video", "work", "dermatologist", "tuesday", "rent",
          "sabbatical", "june", "monday", "thursday", "friday"]


@pytest.mark.parametrize("query", PROBES)
def test_memory_search_never_tells_an_agent_which_words_a_private_memory_holds(tmp_path: Path, query: str) -> None:
    found = tool(mcp(seeded(tmp_path)), "memory_search", query=query)["memories"]
    assert all(memory["sensitivity"] != "private" and not memory.get("withheld") for memory in found), found


@pytest.mark.parametrize("query", PROBES)
def test_moving_a_private_event_changes_no_search_result(tmp_path: Path, query: str) -> None:
    original = tool(mcp(seeded(tmp_path)), "memory_search", query=query)
    moved = tool(mcp(seeded(tmp_path, MOVED)), "memory_search", query=query)
    strip = lambda result: [(m["cite"], m["text"]) for m in result["memories"]]  # noqa: E731
    assert strip(original) == strip(moved)


def test_memory_search_still_finds_private_memories_when_the_server_reveals_them(tmp_path: Path) -> None:
    server = MCPServer(mcp(seeded(tmp_path)).assistant, reveal_private=True)
    found = tool(server, "memory_search", query="dermatologist")["memories"]
    assert [memory["cite"] for memory in found] == ["m7v1"]


# --- MCP: guarded_answer said whether a private memory matched the question --------------------------------

def test_guarded_answer_does_not_tell_an_agent_whether_its_words_match_a_private_memory(tmp_path: Path) -> None:
    llms = {"dermatologist": FakeLLM(), "dentist": FakeLLM()}
    results = {word: tool(mcp(seeded(tmp_path), llm), "guarded_answer", question=f"Any advice about my {word}?")
               for word, llm in llms.items()}
    assert results["dermatologist"]["guard"] == results["dentist"]["guard"]
    assert results["dermatologist"]["placeholders"] == results["dentist"]["placeholders"] == {}
    context = "\n".join(message["content"] for call in llms["dermatologist"].calls for message in call["messages"]
                        if message["role"] == "user")
    assert "⟦m7⟧" not in context and "health appointment" not in context  # its words never pick a private memory


# --- MCP: memory_record labelled private memories with their own words --------------------------------------

@pytest.mark.parametrize(("arguments", "secret_words"), [
    ({"text": "Marc is in prison for fraud until March."}, {"marc", "prison", "fraud"}),
    ({"text": "My brother Marc is in prison for fraud until March.", "sensitivity": "private"},
     {"brother", "prison", "fraud"}),
    ({"text": "HIV clinic visit with Dr Rossi on 12 Nov at 9:00.", "label": "HIV clinic Dr Rossi 12 Nov"},
     {"hiv", "clinic", "rossi", "12"}),
    ({"text": "I have lupus; flare review on Monday 9 Nov at 8:15."}, {"lupus", "flare", "review"}),
])
def test_a_private_memory_recorded_over_mcp_gets_a_category_label(
    tmp_path: Path, arguments: dict[str, str], secret_words: set[str]
) -> None:
    server = mcp(seeded(tmp_path))
    memory = tool(server, "memory_record", **arguments)["memory"]
    stored = server.assistant.store.get(int(memory["id"]))
    assert stored.sensitivity == "private"
    assert stored.label in CATEGORY_LABELS, stored.label
    assert not secret_words & set(stored.label.lower().split())


def test_an_agent_cannot_lower_the_sensitivity_the_device_chose(tmp_path: Path) -> None:
    server = mcp(seeded(tmp_path))
    memory = tool(server, "memory_record", text="My HIV test came back positive.", sensitivity="public")["memory"]
    assert server.assistant.store.get(int(memory["id"])).sensitivity == "private"


def test_a_label_an_agent_gives_a_shared_memory_carries_no_value(tmp_path: Path) -> None:
    server = mcp(seeded(tmp_path))
    memory = tool(server, "memory_record", text="Prefers oat milk in tea.", label="oat milk since 12 Nov")["memory"]
    stored = server.assistant.store.get(int(memory["id"]))
    assert stored.sensitivity == "personal" and "12" not in stored.label and "nov" not in stored.label.lower()


# --- New facts: an everyday word no longer makes a sensitive fact shareable --------------------------------

SENSITIVE_FACTS = [
    "My mother has Alzheimer's.",
    "My mother has Alzheimer's and is moving to a care home in December.",
    "My brother was arrested last night.",
    "My brother is in prison until March.",
    "My sister is gay but has not told our parents.",
    "My sister is bisexual.",
    "My partner is cheating on me.",
    "My partner and I are separating.",
    "My wife had a stroke on Sunday.",
    "My son has autism.",
    "My father is an alcoholic.",
    "I hate going to the AA meetings on weekends.",
    "My dad filed for bankruptcy last month.",
    "I go to Friday prayers at the mosque with my father.",
    "Flying to Berlin on Thursday to meet a hiring manager; my boss must not know.",
    "We are trying for a baby.",
    "My ex has custody of the kids on weekends.",
    "Don't tell anyone: I'm leaving the company in January.",
]
EVERYDAY_FACTS = [
    "My sister Léa loves peonies.",
    "Dinner with friends at the Lebanese place on Friday.",
    "My partner prefers window seats on trains.",
    "I run every Sunday morning.",
    "My brother's favourite band is Air.",
    "Léa's birthday party is on Saturday 14 Nov.",
    "My dad likes crosswords.",
    "Prefers oat milk in tea.",
]


@pytest.mark.parametrize("fact", SENSITIVE_FACTS)
def test_a_sensitive_new_fact_is_private_even_next_to_an_everyday_word(fact: str) -> None:
    assert classify_new_fact(fact) == "private"


@pytest.mark.parametrize("fact", EVERYDAY_FACTS)
def test_an_everyday_fact_stays_personal(fact: str) -> None:
    assert classify_new_fact(fact) == "personal"


def test_a_remembered_sensitive_family_fact_never_reaches_nebius(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    saved = rig.ask("Remember that my mother has Alzheimer's and forgets my name.")
    assert "private" in saved.answer
    rig.ask("What should I bring when I visit my mother?", session_id="s2")
    assert "Alzheimer" not in rig.llm.sent_text()


# --- Corrections: a correction the rules find private makes the memory private -----------------------------

def test_a_correction_that_looks_private_makes_the_memory_private(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    rig.ask("My track session is now replaced by physiotherapy for my knee injury every Tuesday at 19:00.")
    corrected = rig.store.get(16)
    assert corrected.version == 2 and corrected.sensitivity == "private"
    assert corrected.label in CATEGORY_LABELS, corrected.label
    rig.ask("What should I pack for the track session?", session_id="s2")
    sent = rig.llm.sent_text()
    assert "physiotherapy" not in sent and "knee" not in sent


# --- Web search: a message judged private is not sent to the search provider ------------------------------

@pytest.mark.parametrize("message", [
    "Search for tips for my final-round lead designer interview on Thursday.",
    "Look up what to ask the dermatologist on Tuesday about a mole that changed.",
])
def test_a_private_message_is_not_searched_on_the_web(make_rig: Callable[..., Rig], message: str) -> None:
    rig = make_rig()
    turn = rig.ask(message)
    assert rig.tavily.queries == []
    assert "web" in turn.notice.lower()


# --- Schedule questions: no trailing "?" needed to stay away from the cloud router -------------------------

@pytest.mark.parametrize("message", [
    "What's on my plate next week", "Show me my plate for next week.", "what is on my agenda next week",
])
def test_a_question_about_my_own_week_does_not_reach_the_router(make_rig: Callable[..., Rig], message: str) -> None:
    rig = make_rig()
    rig.ask(message)
    assert "route" not in rig.llm.purposes()


def test_a_statement_about_next_week_still_reaches_the_router(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    rig.ask("Next week I'm in Paris for a workshop")
    assert "route" in rig.llm.purposes()


# --- Late binding: the label the model copied is not shown after the filled-in value -----------------------

ECHOING = {
    "route": '{"intent": "answer"}',
    "answer": "Next week:\n- ⟦m7⟧ (private: health appointment, value withheld)\n"
              "- Quarterly product review, Wed 4 Nov at 15:00 [m6v1]",
}


def test_the_label_after_a_filled_in_value_is_dropped(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies=ECHOING)
    turn = rig.ask("What's on my plate next week?")
    assert "Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30." in turn.answer
    assert "value withheld" not in turn.answer
    delivered = delivery_text(turn, rig.store)
    assert delivered.count("health appointment") == 1 and "value withheld" not in delivered


# --- Late binding: on the device, the week reads in date order ----------------------------------------------

OUT_OF_ORDER = {
    "route": '{"intent": "answer"}',
    "answer": "Your week:\n- Quarterly product review, Wed 4 Nov at 15:00 [m6v1]\n- ⟦m7⟧\n"
              "- Dinner with friends, Fri 6 Nov at 20:00 [m9v1]\n- ⟦m11⟧\nEnjoy it.",
}


def test_on_the_device_a_list_of_dated_items_reads_in_date_order(make_rig: Callable[..., Rig]) -> None:
    """Withheld items carry no date, so the model lists them apart; the device knows their dates and puts
    them back in order for the user. The draft that may leave keeps the model's order, which dates nothing."""
    rig = make_rig(replies=OUT_OF_ORDER)
    turn = rig.ask("What's on my plate next week?")
    lines = [line for line in turn.answer.splitlines() if line.startswith("- ")]
    assert [line[:16] for line in lines] == ["- Dermatologist ", "- Quarterly prod", "- Confidential: ",
                                             "- Dinner with fr"]
    assert turn.draft == OUT_OF_ORDER["answer"]
    delivered = delivery_text(turn, rig.store).splitlines()
    assert delivered[1:5] == ["- Quarterly product review, Wed 4 Nov at 15:00 [m6v1]", "- [private: health appointment]",
                              "- Dinner with friends, Fri 6 Nov at 20:00 [m9v1]", "- [private: private appointment]"]


def test_a_list_with_an_undated_item_keeps_the_models_order(make_rig: Callable[..., Rig]) -> None:
    replies = {"route": '{"intent": "answer"}',
               "answer": "Ideas:\n- Dinner with friends [m9v1]\n- Coffee: oat flat white [m19v1]\n- ⟦m7⟧"}
    turn = make_rig(replies=replies).ask("Any ideas for next week?")
    assert [line[:9] for line in turn.answer.splitlines() if line.startswith("- ")] == ["- Dinner ", "- Coffee:",
                                                                                         "- Dermato"]
