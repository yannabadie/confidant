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


# --- Important 6: one odd memory broke every schedule feature ---------------------------------------------

from datetime import date  # noqa: E402

from confidant.memory import Memory  # noqa: E402
from confidant.rules import event_days, item_span, period_asked  # noqa: E402

TODAY = date(2026, 10, 30)


def event(text: str) -> Memory:
    return Memory(99, 1, text, "event", "personal", "current", "user", "", "u", "", None, "x")


@pytest.mark.parametrize("text", ["Sprint 0 - 4 Nov 2026 kickoff at 10:00.", "Day 0 – 15 Nov: race day.",
                                  "Week 0 to 2 Nov: onboarding."])
def test_a_day_zero_before_a_date_is_not_a_range(text: str) -> None:
    first, last = event_days(event(text), TODAY)
    assert first == last


def test_an_absurd_duration_is_ignored_not_a_crash() -> None:
    span = item_span(event("Fast on Tuesday 3 Nov 2026 at 10:00 for " + "1" * 400 + " hours."))
    assert span is not None and span[0] == 600


def test_the_brief_and_the_week_survive_a_sprint_zero(tmp_path: Path) -> None:
    mcp = server(seeded(tmp_path))
    tool(mcp, "memory_record", text="Sprint 0 - 4 Nov 2026 kickoff at 10:00.", kind="event")
    assert tool(mcp, "daily_brief")["brief"]
    assert tool(mcp, "guarded_answer", question="Am I free next week?")["answer"]
    assert mcp.assistant.handle("What's on my plate next week?", session_id="chat").answer


# --- Important 7: availability read "and", a later "is now" and "moved from 3 to 5 Nov" as ranges ---------

def test_two_days_joined_by_and_are_not_a_range() -> None:
    assert event_days(event("Physio on Wednesday 11 Nov and Friday 13 Nov 2026 at 8:00."), TODAY) == (
        date(2026, 11, 11), date(2026, 11, 11))
    # a question naming two days still checks both: every day between them is listed day by day
    assert period_asked("Am I free on 9 Nov and 11 Nov?", TODAY) == (date(2026, 11, 9), date(2026, 11, 11))


def test_between_and_is_a_range() -> None:
    assert event_days(event("Conference between 9 Nov and 11 Nov 2026."), TODAY) == (date(2026, 11, 9),
                                                                                 date(2026, 11, 11))


def test_a_reschedule_word_after_the_dates_keeps_the_range() -> None:
    assert event_days(event("The offsite runs from 9 Nov to 11 Nov 2026; the venue is now Annecy."), TODAY) == (
        date(2026, 11, 9), date(2026, 11, 11))


def test_moved_from_one_day_number_to_a_date_is_the_new_date_only() -> None:
    assert event_days(event("My dentist appointment moved from 3 to 5 Nov at 9:00."), TODAY) == (
        date(2026, 11, 5), date(2026, 11, 5))


# --- Important 8: the wider credential patterns refused ordinary messages, and still missed some codes -------

from confidant.scan import detect_secret  # noqa: E402

NOT_CREDENTIALS = [
    "Remember the IT password training on Tuesday at 14:00.",
    "Password tips: use three random words.",
    "Remember to pin the meeting notes at 9.30.",
    "Pin the 2.1.0 release notes in the design channel.",
    "The phone code for Morocco is 212.",
    "Remember my social security appointment on 12 Nov 2026 at 10:30.",
    "Remember the tax id deadline is 15 May 2027.",
    "Remember PRs must pass code review for PR 1234.",
    "Remember the numbers on the back of Léa's race bib are 1234.",
    "Remember the sign-in desk is open 9:00 / 17:00.",
    "Remember the wifi password is written on the back of the router.",
]
MORE_CREDENTIALS = [
    "Remember the code for the front door is 4512.",
    "Remember the entry code is 4512.",
    "Remember the keypad code at Léa's is 1987.",
    "Remember my SIM PUK is 12345678.",
    "Remember my debit card code is 4821.",
]


@pytest.mark.parametrize("message", NOT_CREDENTIALS)
def test_an_ordinary_message_is_not_refused_as_a_credential(message: str) -> None:
    assert detect_secret(message) is None


@pytest.mark.parametrize("message", MORE_CREDENTIALS)
def test_more_codes_are_refused(message: str) -> None:
    assert detect_secret(message) is not None


# --- Important 9: requests to remember something skipped the router because they start like a question -----

ROUTER_SAYS_REMEMBER = {"route": '{"intent": "remember", "text": "x"}', "answer": "Noted."}


@pytest.mark.parametrize("message", ["Remind me to book the restaurant for Léa tomorrow",
                                     "Could you remember that I'm in Paris next week",
                                     "Do keep in mind that my parents visit next weekend"])
def test_a_request_to_keep_something_still_reaches_the_router(make_rig: Callable[..., Rig], message: str) -> None:
    rig = make_rig(replies=ROUTER_SAYS_REMEMBER)
    before = len(rig.store.all_current())
    rig.ask(message)
    assert "route" in rig.llm.purposes() and len(rig.store.all_current()) == before + 1


@pytest.mark.parametrize("message", ["What's on my plate next week", "Show me my plate for next week.",
                                     "what is on my agenda next week", "Do I have anything on next week"])
def test_a_question_about_my_week_still_stays_on_the_device(make_rig: Callable[..., Rig], message: str) -> None:
    rig = make_rig()
    rig.ask(message)
    assert "route" not in rig.llm.purposes()


# --- Suggestions ----------------------------------------------------------------------------------------------

import httpx  # noqa: E402

from confidant.llm import TokenFactory  # noqa: E402
from confidant.search import Tavily  # noqa: E402


def test_a_proxy_that_refused_the_connection_sent_nothing_and_leaves_no_receipt(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ProxyError("proxy said no")

    llm = TokenFactory("test-key", http=httpx.Client(transport=httpx.MockTransport(refuse)), retry_delay=0)
    assistant = Assistant(seeded(tmp_path), Guard(), ReceiptLog(), Budget(1_000_000, 100), llm, models=MODELS)
    turn = assistant.handle("Draft a birthday message for Léa.", session_id="s")
    assert turn.receipts == []


@pytest.mark.parametrize("body", [{"results": ["x"]}, {"results": [{"url": "https://e.org", "score": "high"}]}])
def test_a_malformed_search_reply_is_a_failed_search_with_its_receipt(make_rig: Callable[..., Rig],
                                                                       body: dict[str, Any]) -> None:
    rig = make_rig()
    rig.assistant.search = Tavily("test-key", retry_delay=0, http=httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))))
    turn = rig.ask("Find a 10-day taper plan for my half marathon.")
    web = [r for r in turn.receipts if r.purpose == "web_search"]
    assert [r.status for r in web] == ["no answer (unexpected response)"]
    assert "Web search failed" in turn.notice
