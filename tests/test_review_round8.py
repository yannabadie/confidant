"""Round 8: a third outside code review of 28 October 2026 (a Claude agent with no history of the project) read
the round-7 changes. It found three oracles left over MCP and Nebius, regressions in credentials, routing and
robustness, and a few smaller faults. Each finding is reproduced here before its fix.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from confidant import Assistant, Budget, Guard, MemoryStore, ReceiptLog, WebResult
from confidant.assistant import masked_private
from confidant.llm import TokenFactory
from confidant.mcp_server import MCPServer
from confidant.memory import Memory
from confidant.rules import event_days, period_asked
from confidant.scan import detect_secret

from conftest import MODELS, PERSONA, FakeLLM, FakeTavily, Rig

TODAY = date(2026, 10, 30)  # the persona's as_of, a Friday
MOVED = {  # the two private appointments on other days, as in tests/test_differential.py
    7: "Dermatologist appointment on Monday 2 Nov 2026 at 16:30.",
    11: "Confidential: final-round interview for a lead designer role on Friday 6 Nov 2026 at 11:00 (video call); "
        "not to be mentioned at work.",
}


def seeded(tmp_path: Path, changes: dict[int, str] | None = None, extra: Sequence[dict[str, Any]] = ()) -> MemoryStore:
    data = json.loads(PERSONA.read_text(encoding="utf-8"))
    for memory in data["memories"]:
        memory["text"] = (changes or {}).get(memory["id"], memory["text"])
    data["memories"] += list(extra)
    path = tmp_path / "persona.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = MemoryStore()
    store.load_seed(path)
    return store


def assistant(store: MemoryStore, llm: FakeLLM | None = None, **options: Any) -> Assistant:
    return Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), llm or FakeLLM(), models=MODELS, **options)


def server(store: MemoryStore, llm: FakeLLM | None = None, **options: Any) -> MCPServer:
    return MCPServer(assistant(store, llm, **options))


def call(mcp: MCPServer, name: str, **arguments: Any) -> dict[str, Any]:
    response = mcp.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": name, "arguments": arguments}})
    assert "error" not in response, response
    result = response["result"]
    assert result["isError"] is False, result
    return result["structuredContent"]


def nebius_requests(llm: FakeLLM) -> list[str]:
    return [message["content"] for call_ in llm.calls for message in call_["messages"] if message["role"] == "user"]


def echo(messages: list[dict[str, str]]) -> str:
    """A hostile model: repeats what it got of the request and the period, which an answer can reflect."""
    content = messages[-1]["content"]
    return "\n".join(line for line in content.splitlines() if line.startswith(("Request:", "Period")))


ECHO = {"route": '{"intent": "answer"}', "answer": echo, "brief": echo}


def agent_view(store: MemoryStore, question: str) -> tuple[list[str], str]:
    """What Nebius receives for an agent's question, and the answer the agent gets back."""
    llm = FakeLLM(ECHO)
    answer = call(server(store, llm), "guarded_answer", question=question)["answer"]
    return nebius_requests(llm), answer


# --- Critical 1: an agent's request was redacted where its words matched a private value ---------------------

@pytest.mark.parametrize("question", [
    "Anything on 11月3日?", "Anything on 11月5日?", "Anything on 11月12日?", "Anything at half ten?",
    "Anything at eleven sharp?", "Anything on the third of November?", "Is anything on at 10.30?",
])
def test_an_agents_request_does_not_depend_on_when_private_items_are(tmp_path: Path, question: str) -> None:
    original, moved = agent_view(seeded(tmp_path), question), agent_view(seeded(tmp_path, MOVED), question)
    assert original == moved
    assert not any("[withheld]" in line for text in original[0] for line in text.splitlines()
                   if line.startswith("Request:"))


def test_an_agents_request_does_not_tell_which_names_a_private_memory_holds(tmp_path: Path) -> None:
    rossi = seeded(tmp_path, {7: "Dermatologist appointment with Dr Rossi on Tuesday 3 Nov 2026 at 10:30."})
    rossi_view = [agent_view(rossi, q) for q in ("Any notes about rossi?", "Rossi: anything I should know?")]
    bianchi = seeded(tmp_path, {7: "Dermatologist appointment with Dr Bianchi on Tuesday 3 Nov 2026 at 10:30."})
    bianchi_view = [agent_view(bianchi, q) for q in ("Any notes about rossi?", "Rossi: anything I should know?")]
    assert rossi_view == bianchi_view


def test_an_agent_cannot_confirm_the_whole_text_of_a_private_memory(tmp_path: Path) -> None:
    question = "Is it true that I am saving quietly for a sabbatical month away?"
    sabbatical = agent_view(seeded(tmp_path, {21: "I am saving quietly for a sabbatical month away."}), question)
    bike = agent_view(seeded(tmp_path, {21: "I am saving quietly for a new road bike."}), question)
    assert sabbatical == bike


def test_an_agents_request_still_refuses_a_credential(tmp_path: Path) -> None:
    sent, _ = agent_view(seeded(tmp_path), "My wifi password is hunter2: is that strong enough?")
    assert not any("hunter2" in text for text in sent)


# --- Critical 2: the brief picked private memories by their words and by today's weekday --------------------

def placeholders(texts: list[str]) -> list[str]:
    return sorted({line.split(" ")[0] for text in texts for line in text.splitlines() if line.startswith("⟦")})


@pytest.mark.parametrize("before, after", [
    ({21: "Savings goal: €6,000 by June 2027 for a sabbatical month."},
     {21: "Savings goal: €6,000 by June 2027 for a sabbatical month; need to open the account."}),
    ({10: "Monthly rent is €1,180; therapy session every Monday at 18:00."},
     {10: "Monthly rent is €1,180; therapy session every Friday at 18:00."}),
])
def test_the_brief_does_not_pick_private_memories_by_their_words_or_weekday(
        tmp_path: Path, before: dict[int, str], after: dict[int, str]) -> None:
    seen = []
    for changes in (before, after):
        llm = FakeLLM()
        brief = call(server(seeded(tmp_path, changes), llm), "daily_brief")
        seen.append((brief["guard"], brief["brief"].splitlines()[-1], placeholders(nebius_requests(llm))))
    assert seen[0] == seen[1]


def test_the_brief_still_lists_a_private_routine_on_the_device(tmp_path: Path) -> None:
    store = seeded(tmp_path, {10: "Monthly rent is €1,180; therapy session every Friday at 18:00."})
    brief = assistant(store).daily_brief("s")
    assert 10 in {memory.id for memory in brief.guard.prepared.withheld}


# --- Critical 3: public web text was redacted where it happened to equal a private value ---------------------

CLUB = [WebResult("Lyon running club calendar", "https://club.example.org/calendar",
                  "Group taper run on Tuesday 3 Nov at 10:30 from Parc de la Tête d'Or; recovery jog Thursday 5 Nov.",
                  0.9)]


def club_lines(texts: list[str]) -> list[str]:
    return [line for text in texts for line in text.splitlines() if line.startswith("[m") and "club calendar" in line]


def test_public_web_text_reaches_nebius_whole_whatever_private_memories_hold(tmp_path: Path) -> None:
    seen = []
    for changes in (None, MOVED):
        llm = FakeLLM(ECHO)
        rig = assistant(seeded(tmp_path, changes), llm, search=FakeTavily(CLUB))
        rig.handle("Find a 10-day taper plan for my half marathon.", session_id="s")
        rig.handle("What does the running club calendar say?", session_id="s")
        seen.append(club_lines(nebius_requests(llm)))
    assert seen[0] == seen[1]
    assert all("3 Nov at 10:30" in line and "Thursday 5 Nov" in line for line in seen[0]), seen[0]


# --- Important 4: a placeholder dated itself by leaving the brief once its day had passed --------------------

def test_the_brief_sends_the_same_placeholders_every_day(tmp_path: Path) -> None:
    store = seeded(tmp_path)
    sent = set()
    for offset in range(8):  # 30 Oct to 6 Nov: both private appointments fall in between
        llm = FakeLLM()
        call(server(store, llm, today=lambda d=TODAY + timedelta(days=offset): d), "daily_brief")
        sent.add(tuple(placeholders(nebius_requests(llm))))
    assert len(sent) == 1, sent


def test_a_private_appointment_that_is_over_is_not_shown_as_coming_up(tmp_path: Path) -> None:
    # A question with no period: since the judge's review (tests/test_review_judge.py) the device also leaves out
    # private items outside a period asked, and on 4 Nov "next week" holds neither 3 nor 5 Nov.
    store = seeded(tmp_path)
    reply = {"answer": "Coming up: [m9v1].\nWithheld:\n- ⟦m7⟧\n- ⟦m11⟧"}
    turn = assistant(store, FakeLLM(reply), today=lambda: date(2026, 11, 4)).handle(
        "What's coming up?", session_id="s")
    assert "Dermatologist" not in turn.answer  # 3 Nov is over
    assert "final-round interview" in turn.answer  # 5 Nov is not
    assert "⟦m7⟧" in turn.draft  # what may leave the device does not depend on the date
    assert "unknown placeholder" not in turn.notice


# --- Important 5: credentials the round-7 patterns no longer refused ----------------------------------------

@pytest.mark.parametrize("message", [
    "Remember my PIN is 4.8.2.1.", "Alarm code is 45.12.", "Remember my sign-in is alex@example.com / Sunflower2026",
    "Remember the 3 digits on the back of my Visa are 123.", "Remember my passport number is 12AB34567.",
    "My passport no. is 19FV12345.", "The digicode is 12A45.", "The garage opener code is 7731.",
])
def test_these_credentials_are_refused(message: str) -> None:
    assert detect_secret(message) is not None


@pytest.mark.parametrize("message", [
    "My PIN is 4.8.2.1, is that a weak one?", "My sign-in is alex@example.com / Sunflower2026, is that safe?",
    "The 3 digits on the back of my Visa are 123, should I worry?", "My passport no. is 19FV12345, when does it expire?",
])
def test_a_credential_asked_about_is_not_sent(tmp_path: Path, message: str) -> None:
    llm = FakeLLM(ECHO)
    assistant(seeded(tmp_path), llm).handle(message, session_id="s")
    assert nebius_requests(llm) == []


@pytest.mark.parametrize("message", [
    "PIN the slides for 2026 review: 3 decks.", "Remember the code for the alarm company is ACME-2026.",
    "Pin the 2.1.0 release notes in the design channel.", "Remember Léa's PIN-up poster costs €120.",
    "Remember the sign-in desk is open 9:00 / 17:00.", "Remember my login is at 9:00 / 17:00 shifts.",
    "My passport number renewal appointment is on 12 Nov 2026 at 10:30.", "The building code requires 2 exits.",
])
def test_these_ordinary_messages_are_not_refused(message: str) -> None:
    assert detect_secret(message) is None


# --- Important 6: masked_private missed abbreviations and other languages -----------------------------------

@pytest.mark.parametrize("message, kept", [
    ("Look up what to ask the dermatologist on Tue about a mole that changed.", "Tue"),
    ("Tips for my final-round interview on Thurs morning?", "Thurs"),
    ("My therapist moved me to Wed next week, what should I bring?", "Wed"),
    ("Search tips for my interview in Nov or early Dec.", "Nov"),
    ("Search tips for my interview in Nov or early Dec.", "Dec"),
    ("Tips for my interview el martes por la mañana?", "martes"),
    ("Was soll ich zum Hautarzt am Dienstag mitnehmen?", "Dienstag"),
    ("What should I ask my oncologist in May?", "May"),
])
def test_a_private_message_has_its_abbreviated_and_foreign_days_and_months_masked(message: str, kept: str) -> None:
    assert kept not in masked_private(message).split() and kept not in masked_private(message)


def test_may_as_a_verb_is_kept() -> None:
    assert masked_private("May I ask my dermatologist about it?").startswith("May I ask")


def test_the_private_search_notice_says_which_names_are_masked(make_rig: Callable[..., Rig]) -> None:
    turn = make_rig().ask("Look up what to ask the dermatologist on Tue about a mole that changed.")
    assert "capitalised names" in turn.notice


# --- Important 7: a correction did not apply "private unless clearly everyday" to the words it adds ----------

ADDED = ["Brunch with my sponsor after Sunday's meeting.", "Wedding is off; Sam moved out.",
         "Cooking class at the women's shelter on Monday."]


@pytest.mark.parametrize("text", ADDED)
def test_a_correction_over_mcp_that_adds_new_words_is_classed_like_a_new_fact(tmp_path: Path, text: str) -> None:
    mcp = server(seeded(tmp_path))
    recorded = call(mcp, "memory_record", text=text)["memory"]["sensitivity"]
    corrected = call(mcp, "memory_correct", memory_id=9, new_text=text, expected_version=1)["memory"]["sensitivity"]
    assert recorded == corrected == "private"


def test_a_chat_correction_that_adds_new_words_is_classed_like_a_new_fact(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    rig.ask("My dinner with friends changed to brunch with my sponsor after Sunday's meeting.")
    assert rig.store.get(9).sensitivity == "private"


@pytest.mark.parametrize("text", ["Dinner with friends at the Lebanese place near Bellecour on Saturday 7 Nov 2026 "
                                  "at 20:00.", "Dinner with friends near Bellecour on Friday 6 Nov 2026 at 20:30."])
def test_a_correction_of_the_date_or_time_keeps_the_level(tmp_path: Path, text: str) -> None:
    memory = call(server(seeded(tmp_path)), "memory_correct", memory_id=9, new_text=text, expected_version=1)["memory"]
    assert (memory["sensitivity"], memory["label"]) == ("personal", "dinner with friends")


def test_the_store_never_lowers_a_memory_in_a_correction(tmp_path: Path) -> None:
    store = seeded(tmp_path)
    corrected = store.correct(7, "Dermatologist appointment on Wednesday 4 Nov 2026 at 10:30.", author="t",
                              source_ref="t", expected_version=1, sensitivity="public", label="clinic")
    assert corrected.sensitivity == "private"


# --- Important 8: an odd year crashed schedule reading -------------------------------------------------------

@pytest.mark.parametrize("text", ["Kickoff on 0000-11-03 at 10:00.", "Archive review on 3 Nov 0000 at 10:00.",
                                  "Time capsule opens from 30 Dec 9999 to 31 Dec 9999.", "Deadline 31/12/9999 for the vault."])
def test_an_odd_year_breaks_nothing(tmp_path: Path, text: str) -> None:
    mcp = server(seeded(tmp_path))
    call(mcp, "memory_record", text=text, kind="event")
    call(mcp, "daily_brief")
    call(mcp, "guarded_answer", question="Am I free next week?")
    call(mcp, "guarded_answer", question="What's on my plate next week?")
    mcp.assistant.handle("What's on my plate next week?", session_id="chat")


def test_an_odd_year_in_a_private_memory_is_not_an_oracle(tmp_path: Path) -> None:
    odd = agent_view(seeded(tmp_path, {7: "Dermatologist appointment on 3 Nov 0000 at 10:30."}),
                     "Anything planned next week?")
    assert odd[1]  # answered, no internal error


# --- Important 9: memory_history showed the text a now-private memory had before its correction --------------

def test_the_history_of_a_memory_now_private_withholds_every_version(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    rig.ask("Correction: partner Sam is vegetarian and was just diagnosed with diabetes; he starts insulin next week.")
    assert rig.store.get(12).sensitivity == "private"
    versions = call(MCPServer(rig.assistant), "memory_history", memory_id=12)["versions"]
    assert len(versions) == 2
    assert not any("vegetarian" in version["text"] or "diabetes" in version["text"] for version in versions)


# --- Important 10: first-person period requests reached the cloud router again ------------------------------

SEARCH = {"route": '{"intent": "search", "query": "weekly plan"}', "answer": "ok"}


@pytest.mark.parametrize("message", ["Tell me what's on my plate next week", "Give me my agenda for next week",
                                     "Can you list my plans for next week", "Remind me what I have next week",
                                     "Could you tell me what I have on Tuesday", "Would you show me what I have this week"])
def test_a_request_about_my_week_stays_on_the_device(make_rig: Callable[..., Rig], message: str) -> None:
    rig = make_rig(replies=SEARCH)
    rig.ask(message)
    assert "route" not in rig.llm.purposes() and rig.tavily.queries == []


def test_remind_me_to_still_reaches_the_router(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies={"route": '{"intent": "remember", "text": "x"}', "answer": "Noted."})
    rig.ask("Remind me to book the restaurant for Léa tomorrow")
    assert "route" in rig.llm.purposes()


# --- Suggestions ---------------------------------------------------------------------------------------------

def event(text: str) -> Memory:
    return Memory(99, 1, text, "event", "personal", "current", "user", "", "u", "", None, "x")


@pytest.mark.parametrize("text, days", [
    ("Sprint 1 - 4 Nov 2026 kickoff at 10:00.", (date(2026, 11, 4), date(2026, 11, 4))),
    ("Phase 2 - 9 Nov 2026: go-live.", (date(2026, 11, 9), date(2026, 11, 9))),
    ("Offsite moved to 9-11 Nov 2026.", (date(2026, 11, 9), date(2026, 11, 11))),
    ("Offsite on 9-11 Nov 2026.", (date(2026, 11, 9), date(2026, 11, 11))),
    ("Moved from 3 to 4 Nov 2026.", (date(2026, 11, 4), date(2026, 11, 4))),
])
def test_a_label_number_is_not_the_start_of_a_range(text: str, days: tuple[date, date]) -> None:
    assert event_days(event(text), TODAY) == days


def test_two_far_apart_days_in_a_question_are_both_checked(make_rig: Callable[..., Rig]) -> None:
    assert period_asked("Am I free on 9 Nov and 11 Nov?", TODAY) == (date(2026, 11, 9), date(2026, 11, 11))
    turn = make_rig().ask("Am I free on 2 Nov and 30 Nov?")
    assert "30 November" in turn.answer or "30 Nov" in turn.answer


@pytest.mark.parametrize("usage", ["n/a", {"prompt_tokens": "12k"}, [1, 2]])
def test_an_odd_usage_field_does_not_break_the_turn_or_lose_its_receipt(tmp_path: Path, usage: Any) -> None:
    body = {"choices": [{"message": {"content": "Happy birthday Léa! [m3v1]"}}], "usage": usage}
    llm = TokenFactory("test-key", http=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=body))), retry_delay=0)
    turn = Assistant(seeded(tmp_path), Guard(), ReceiptLog(), Budget(1_000_000, 100), llm, models=MODELS).handle(
        "Draft a birthday message for Léa.", session_id="s")
    assert turn.receipts and all(receipt.status == "answered" for receipt in turn.receipts)
