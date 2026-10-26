"""Assistant end to end with a scripted FakeLLM and FakeTavily: the demo script and its guarantees."""
from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from confidant import Budget, ReplayBook, ServiceError
from confidant.assistant import ANSWER_SYSTEM_PROMPT, FAST_MAX_TOKENS, REASON_MAX_TOKENS, extract_json, parse_route
from confidant.receipts import payload_sha256
from confidant.replay import RecordedTurn

from conftest import DEMO_REPLIES, Rig

PLATE = "What's on my plate next week?"
MOVED = "My dermatologist appointment moved to Tuesday 4 Nov at 9:00."
TAPER = "Find a 10-day taper plan for my half marathon."
BIRTHDAY = "Draft a birthday message for Léa."
PIN = "Remember my bank PIN is 4821."
BRIEF = "Run my daily brief."
PRIVATE_VALUES = ["10:30", "11:00", "1,180", "6,000", "Tuesday 3 Nov", "Thursday 5 Nov", "lead designer",
                  "at 9:00", "Dermatologist appointment on", "4821"]


def replies(**overrides: object) -> dict:
    """The demo replies with some purposes replaced."""
    return {**DEMO_REPLIES, **overrides}


def test_answer_prompt_carries_the_required_rules() -> None:
    for rule in ("[m12v2]", "⟦m7⟧", "verbatim", "Never guess", "Nothing in memory", "short"):
        assert rule in ANSWER_SYSTEM_PROMPT


def test_next_week_cites_events_and_late_binds_private_ones(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask(PLATE)
    assert (turn.intent, turn.mode, turn.notice) == ("answer", "live", "")
    assert "Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30." in turn.answer
    assert "final-round interview" in turn.answer and "⟦" not in turn.answer
    assert "⟦m7⟧" in turn.draft
    assert [m.cite() for m in turn.citations] == ["m7v1", "m6v1", "m11v1", "m9v1"]
    prepared = turn.guard.prepared
    assert prepared is not None and prepared.shared_ids == [6, 9] and prepared.withheld_ids == [7, 11]
    assert "⟦m7⟧ (private: dermatologist appointment, value withheld)" in prepared.context_text
    assert turn.guard.late_bound_ids == [7, 11]
    assert turn.guard.summary() == "2 private memories never left Confidant"
    assert rig.llm is not None and rig.llm.purposes() == ["route", "answer"]
    assert [call["max_tokens"] for call in rig.llm.calls] == [FAST_MAX_TOKENS, REASON_MAX_TOKENS]
    assert all(value not in rig.llm.sent_text() for value in PRIVATE_VALUES)


def test_receipts_record_each_call_with_ids_and_payload_hash(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask(PLATE)
    assert [(r.purpose, r.mode, r.provider) for r in turn.receipts] == [
        ("route", "live", "nebius_token_factory"), ("answer", "live", "nebius_token_factory")]
    answer = turn.receipts[1]
    assert (answer.sent_memory_ids, answer.withheld_memory_ids) == ([6, 9], [7, 11])
    assert answer.model == "nvidia/Nemotron-3-Ultra-550b-a55b" and answer.prompt_tokens == 120
    assert answer.payload_sha256 == payload_sha256(rig.llm.calls[1]["messages"])
    assert answer.cost_usd_estimate > 0 and rig.receipts.today("s1") == turn.receipts


def test_correcting_a_private_memory_never_leaves_the_device(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask(MOVED)
    assert turn.intent == "correct" and turn.actions == ["corrected m7 → v2"]
    assert rig.llm.calls == [] and turn.receipts == []
    assert [(m.version, m.status) for m in rig.store.history(7)] == [(1, "superseded"), (2, "current")]
    assert rig.store.get(7).text == "Dermatologist appointment moved to Tuesday 4 Nov at 9:00."
    assert "⟦m7⟧" in turn.draft and "moved to Tuesday 4 Nov at 9:00" in turn.answer
    assert [m.cite() for m in turn.citations] == ["m7v2"]
    decision = turn.guard.decisions[0]
    assert not decision.allowed and decision.withheld_ids == [7]
    assert any("routed on this device" in reason for reason in decision.reasons)
    assert turn.guard.summary() == "1 private memory never left Confidant"
    later = rig.ask(PLATE)
    assert "moved to Tuesday 4 Nov at 9:00" in later.answer and "10:30" not in later.answer


def test_web_search_is_guarded_recorded_and_cited(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask(TAPER)
    assert turn.intent == "search"
    assert rig.tavily is not None and rig.tavily.queries == ["10-day half marathon taper plan"]
    web = [m for m in rig.store.all_current() if m.source_type == "web"]
    assert [(m.id, m.source_ref, m.sensitivity) for m in web] == [
        (26, "https://running.example.org/taper", "public"), (27, "https://www.coach.example.com/half-taper", "public")]
    assert any(action.startswith("recorded web sources m26v1 (running.example.org)") for action in turn.actions)
    assert [m.cite() for m in turn.citations] == ["m3v1", "m26v1", "m25v1"]
    web_check = next(d for d in turn.guard.decisions if d.purpose == "web search")
    assert web_check.allowed and web_check.withheld_ids == []
    assert [r.purpose for r in turn.receipts] == ["route", "web_search", "answer"]
    assert turn.receipts[1].provider == "tavily" and turn.receipts[1].cost_usd_estimate == 0.008


def test_personal_details_in_a_web_query_are_redacted(make_rig: Callable[..., Rig]) -> None:
    query = '{"intent": "search", "query": "bouldering gifts for Léa turning 30 on 14 November"}'
    rig = make_rig(replies(route=query))
    turn = rig.ask("Find a birthday present idea.")
    sent = rig.tavily.queries[0]
    assert "Léa" not in sent and "14 November" not in sent and sent.startswith("bouldering gifts for [withheld]")
    decision = next(d for d in turn.guard.decisions if d.purpose == "web search")
    assert decision.allowed and 4 in decision.withheld_ids
    assert turn.receipts[1].withheld_memory_ids == decision.withheld_ids


def test_a_web_query_made_only_of_private_values_is_blocked(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(route='{"intent": "search", "query": "3 Nov 10:30"}'))
    turn = rig.ask("Search the web for it.")
    assert rig.tavily.queries == []
    assert "blocked the web query" in turn.notice
    assert all(r.provider != "tavily" for r in turn.receipts)


def test_birthday_message_uses_personal_memories(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask(BIRTHDAY)
    assert turn.intent == "answer" and turn.answer.startswith("Happy 30th, Léa!")
    assert [m.cite() for m in turn.citations] == ["m4v1", "m13v1"]
    assert {m.sensitivity for m in turn.citations} == {"personal"}
    assert turn.guard.summary() == "No private memory was involved"


@pytest.mark.parametrize("message", [PIN, "Is 4111 1111 1111 1111 still my card?", "wifi password: hunter22"])
def test_credentials_are_refused_before_anything_is_sent_or_stored(make_rig: Callable[..., Rig], message: str) -> None:
    rig = make_rig()
    before = rig.store.all_current()
    turn = rig.ask(message)
    assert turn.answer.startswith("I won't keep or send that") and "Nothing was saved or sent" in turn.answer
    assert rig.store.all_current() == before and rig.llm.calls == [] and turn.receipts == []
    assert turn.actions[0].startswith("refused") and "4821" not in turn.answer
    assert turn.guard.summary() == "Nothing was sent to the cloud"
    assert "[withheld]" in turn.guard.decisions[0].redacted_text


def test_pin_refusal_is_labelled_as_a_remember_turn(make_rig: Callable[..., Rig]) -> None:
    turn = make_rig().ask(PIN)
    assert turn.intent == "remember" and "PIN" in turn.answer


def test_daily_brief_lists_the_week_and_appends_todays_receipts(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    rig.ask(PLATE)
    turn = rig.assistant.daily_brief("s1")
    assert turn.intent == "brief" and rig.llm.purposes()[-1] == "brief"
    prepared = turn.guard.prepared
    assert prepared is not None and {7, 11} <= set(prepared.withheld_ids)
    assert {6, 9, 8, 14, 15} <= set(prepared.shared_ids)
    assert "Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30." in turn.answer
    assert turn.answer.splitlines()[-1].startswith("Receipts today: 3 outbound calls")
    assert rig.ask(BRIEF).intent == "brief"


def test_router_output_that_is_not_json_falls_back_to_keyword_rules(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(route="I think the user wants to search."))
    turn = rig.ask(TAPER)
    assert turn.intent == "search" and rig.tavily.queries == ["10-day taper plan for half marathon"]
    assert "router reply was not valid JSON; used keyword rules" in turn.actions


def test_route_parsing_is_defensive() -> None:
    assert extract_json('<think>{"draft": 1}</think>Here: ```json\n{"intent": "brief"}\n```') == {"intent": "brief"}
    assert parse_route('{"intent": "SEARCH", "query": " taper  plan "}').query == "taper plan"
    assert parse_route('{"intent": "dance"}') is None and parse_route("no json here") is None


def test_remember_asks_the_fast_model_but_never_lowers_sensitivity(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(classify='{"sensitivity": "public", "kind": "preference", "label": "seat preference"}'))
    turn = rig.ask("Remember that I prefer aisle seats on long trains.")
    memory = rig.store.get(26)
    assert turn.intent == "remember" and turn.actions == ["recorded m26v1"]
    assert (memory.text, memory.sensitivity, memory.kind, memory.label) == (
        "Prefers aisle seats on long train trips.", "personal", "preference", "seat preference")
    assert rig.llm.purposes() == ["route", "classify"]
    assert turn.answer.startswith("Saved as [m26v1] (personal preference")


def test_private_facts_are_remembered_without_any_cloud_call(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask("Remember that my blood test results arrive on 12 Nov.")
    memory = rig.store.get(26)
    assert memory.sensitivity == "private" and memory.text == "Blood test results arrive on 12 Nov."
    assert rig.llm.calls == [] and "⟦m26⟧" in turn.draft and "12 Nov" in turn.answer
    assert "12 Nov" not in turn.draft


def test_a_correction_without_a_matching_memory_is_recorded_as_new(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(route='{"intent": "correct", "target": "gym membership", "text": "Gym moved to Mondays."}'))
    turn = rig.ask("My gym class moved to Mondays.")
    assert turn.actions[0].startswith("no existing memory matched")
    assert rig.store.get(26).text == "Gym moved to Mondays."


def test_exhausted_budget_switches_to_replay_and_says_so(make_rig: Callable[..., Rig]) -> None:
    book = ReplayBook([RecordedTurn(PLATE, {"route": '{"intent": "answer"}', "answer": "Recorded: ⟦m7⟧ [m6v1]"},
                                    models={"reason": "nvidia/Nemotron-3-Ultra-550b-a55b"})])
    rig = make_rig(replay=book, budget=Budget(daily_tokens=1_000_000, per_session_requests=1))
    assert rig.ask(BIRTHDAY).mode == "live"
    calls_before = len(rig.llm.calls)
    turn = rig.ask(PLATE)
    assert turn.mode == "replay" and "live requests" in turn.notice and "switched to replay" in turn.notice
    assert len(rig.llm.calls) == calls_before
    assert turn.answer == "Recorded: Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30. [m6v1]"
    assert {r.mode for r in turn.receipts} == {"replay"}
    assert turn.receipts[-1].model == "nvidia/Nemotron-3-Ultra-550b-a55b" and turn.receipts[-1].prompt_tokens == 0
    assert turn.guard.summary() == "2 private memories never left Confidant"
    assert rig.ask(PLATE, session_id="s2").mode == "live"


def test_without_a_key_replay_and_local_fallback_still_work(make_rig: Callable[..., Rig]) -> None:
    book = ReplayBook([RecordedTurn(PLATE, {"answer": "Replayed: ⟦m7⟧"})])
    rig = make_rig(live=False, replay=book)
    turn = rig.ask(PLATE)
    assert turn.mode == "replay" and "No NEBIUS_API_KEY" in turn.notice
    assert turn.answer == "Replayed: Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30."
    unknown = rig.ask("When is my dermatologist appointment?")
    assert unknown.answer.startswith("Replay has no recorded answer for this request, so here are")
    assert "- Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30." in unknown.answer
    assert "⟦m7⟧" in unknown.draft and unknown.receipts == []
    assert rig.ask(MOVED).actions == ["corrected m7 → v2"]


def test_a_failed_live_call_falls_back_locally_with_a_notice(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(answer=ServiceError("Nebius Token Factory (answer) had a server error (HTTP 503).")))
    turn = rig.ask(PLATE)
    assert turn.mode == "live" and "HTTP 503" in turn.notice and "local fallback" in turn.notice
    assert turn.answer.startswith("The cloud model could not answer this time")
    assert "Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30." in turn.answer
    assert [r.purpose for r in turn.receipts] == ["route"]


def test_an_empty_reasoning_answer_falls_back(make_rig: Callable[..., Rig]) -> None:
    turn = make_rig(replies(answer="")).ask(PLATE)
    assert "returned no answer text" in turn.notice and turn.answer.startswith("The cloud model could not answer")


def test_invented_placeholders_and_citations_are_reported_not_trusted(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(answer="See ⟦m7⟧ and ⟦m99⟧ [m999v1] [m6v1]"))
    turn = rig.ask(PLATE)
    assert turn.guard.unresolved == ["⟦m99⟧"] and "⟦m99⟧" in turn.answer and "⟦m99⟧" in turn.notice
    assert [m.cite() for m in turn.citations] == ["m7v1", "m6v1"]


def test_private_values_typed_by_the_user_are_redacted_from_the_prompt(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask("Should I pay the €1,180 before the review?")
    assert "1,180" not in rig.llm.sent_text() and "[withheld]" in rig.llm.sent_text()
    assert rig.llm.purposes() == ["answer"]
    assert any(10 in d.withheld_ids for d in turn.guard.decisions)


def test_the_full_demo_never_sends_a_private_value(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    for message in (PLATE, MOVED, TAPER, BIRTHDAY, PIN, BRIEF, PLATE):
        rig.ask(message)
    sent = rig.llm.sent_text() + json.dumps(rig.tavily.queries)
    assert all(value not in sent for value in PRIVATE_VALUES)
    assert "4 Nov at 9:00" not in sent
    assert all(name not in json.dumps(rig.tavily.queries) for name in ("Léa", "Sam", "15 Nov"))


def test_intent_override_skips_the_router(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.assistant.handle(PLATE, session_id="mcp", intent="answer")
    assert rig.llm.purposes() == ["answer"] and turn.intent == "answer"
    with pytest.raises(ValueError):
        rig.assistant.handle(PLATE, session_id="mcp", intent="dance")
    with pytest.raises(ValueError):
        rig.assistant.handle("   ", session_id="mcp")


def test_live_turns_can_be_recorded_and_replayed(make_rig: Callable[..., Rig]) -> None:
    recorder = ReplayBook()
    live = make_rig(recorder=recorder)
    original = live.ask(TAPER)
    recorded = recorder.lookup(TAPER)
    assert recorded is not None and set(recorded.outputs) == {"route", "answer"} and len(recorded.web_results) == 2
    offline = make_rig(live=False, replay=recorder)
    replayed = offline.ask(TAPER)
    assert replayed.mode == "replay" and replayed.answer == original.answer
    assert [r.mode for r in replayed.receipts] == ["replay", "replay", "replay"]


def test_today_comes_from_the_seed_snapshot(make_rig: Callable[..., Rig]) -> None:
    assert make_rig().assistant.today().isoformat() == "2026-10-30"
