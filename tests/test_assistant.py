"""Assistant end to end with a scripted FakeLLM and FakeTavily: the demo script and its guarantees."""
from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from confidant.guard import WITHHELD_HEADING

from confidant import Budget, ReplayBook, ServiceError
from confidant.assistant import ANSWER_SYSTEM_PROMPT, FAST_MAX_TOKENS, REASON_MAX_TOKENS, extract_json, parse_route
from confidant.llm import ChatResult
from confidant.receipts import payload_sha256
from confidant.replay import RecordedTurn

from conftest import DEMO_REPLIES, MODELS, Rig

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


def test_answer_prompt_asks_for_no_tables() -> None:
    # Live, Nemotron Ultra answered the taper question with a Markdown table, which neither the chat nor the
    # Telegram delivery renders: the pipes showed as text.
    assert "no tables" in ANSWER_SYSTEM_PROMPT


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
    assert "⟦m7⟧ (private: health appointment, value withheld)" in prepared.context_text
    assert turn.guard.late_bound_ids == [7, 11]
    assert turn.guard.summary() == "2 private memories never left Confidant"
    assert rig.llm is not None and rig.llm.purposes() == ["answer"]  # a question about next week: no router
    assert [call["max_tokens"] for call in rig.llm.calls] == [REASON_MAX_TOKENS]
    assert all(value not in rig.llm.sent_text() for value in PRIVATE_VALUES)


def test_receipts_record_each_call_with_ids_and_payload_hash(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask(PLATE)
    assert [(r.purpose, r.mode, r.provider) for r in turn.receipts] == [("answer", "live", "nebius_token_factory")]
    answer = turn.receipts[0]
    assert (answer.sent_memory_ids, answer.withheld_memory_ids) == ([6, 9], [7, 11])
    assert answer.model == "nvidia/Nemotron-3-Ultra-550b-a55b" and answer.prompt_tokens == 120
    assert answer.payload_sha256 == payload_sha256(rig.llm.calls[0]["messages"])
    assert answer.cost_usd_estimate > 0 and rig.receipts.today("s1") == turn.receipts
    routed = rig.ask(BIRTHDAY)  # a turn with two calls has two receipts, in order
    assert [(r.purpose, r.mode) for r in routed.receipts] == [("route", "live"), ("answer", "live")]
    assert [r.payload_sha256 for r in routed.receipts] == [payload_sha256(c["messages"]) for c in rig.llm.calls[1:]]


def test_each_outbound_call_is_kept_with_the_hash_its_receipt_stores(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask(PLATE)
    assert [o.sha256 for o in turn.outbound] == [r.payload_sha256 for r in turn.receipts]
    assert all(o.sha256 == payload_sha256(o.payload) for o in turn.outbound)
    assert turn.outbound[-1].payload == rig.llm.calls[-1]["messages"]
    assert not any(value in json.dumps(turn.outbound[-1].payload, ensure_ascii=False) for value in PRIVATE_VALUES)


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
    assert not decision.allowed and any("looks private" in reason for reason in decision.reasons)
    assert any("routed on this device" in reason for reason in decision.reasons)
    assert turn.guard.summary() == "1 private memory never left Confidant"
    later = rig.ask(PLATE)
    assert "moved to Tuesday 4 Nov at 9:00" in later.answer and "10:30" not in later.answer


def test_web_search_is_guarded_recorded_and_cited(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask(TAPER)
    assert turn.intent == "search"
    assert rig.tavily is not None and rig.tavily.queries == ["10-day taper plan for half marathon"]
    web = [m for m in rig.store.all_current() if m.source_type == "web"]
    assert [(m.id, m.source_ref, m.sensitivity) for m in web] == [
        (26, "https://running.example.org/taper", "public"), (27, "https://www.coach.example.com/half-taper", "public")]
    assert any(action.startswith("recorded web sources m26v1 (running.example.org)") for action in turn.actions)
    assert [m.cite() for m in turn.citations] == ["m3v1", "m26v1", "m25v1"]
    web_check = next(d for d in turn.guard.decisions if d.purpose == "web search")
    assert web_check.allowed and web_check.withheld_ids == []
    assert [r.purpose for r in turn.receipts] == ["web_search", "answer"]  # the rules saw a search: no router
    assert turn.receipts[0].provider == "tavily" and turn.receipts[0].cost_usd_estimate == 0.008


def test_live_without_tavily_guards_the_query_but_sends_nothing(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(web=False)
    turn = rig.ask(TAPER)
    assert "Web search is off" in turn.notice and "10-day taper plan" in turn.notice
    web_check = next(d for d in turn.guard.decisions if d.purpose == "web search")
    assert web_check.allowed and web_check.provider == "tavily"
    assert all(r.provider != "tavily" for r in turn.receipts)
    assert all(o.provider != "tavily" for o in turn.outbound)


def test_personal_details_in_a_web_query_are_redacted(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask("Search for bouldering gifts for Léa turning 30 on 14 November.")
    sent = rig.tavily.queries[0]
    assert "Léa" not in sent and "14 November" not in sent and sent.startswith("bouldering gifts for [withheld]")
    decision = next(d for d in turn.guard.decisions if d.purpose == "web search")
    assert decision.allowed and 4 in decision.withheld_ids
    assert turn.receipts[0].withheld_memory_ids == decision.withheld_ids


def test_a_search_the_router_chose_carries_none_of_the_values_typed(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(route='{"intent": "search", "query": "bouldering gifts for Léa"}'))
    rig.ask("Any bouldering gifts for Léa?")  # no search verb and no date or number: the router decides
    assert rig.llm.calls[0]["messages"][1]["content"] == "Any bouldering gifts for [name]?"
    assert rig.tavily.queries == ["Any bouldering gifts for"]


def test_a_message_that_may_carry_a_value_is_routed_on_the_device(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    for message in ("Is anything on the third of November?", "Am I free on Tuesday morning?", "Is 45,000 a good offer?"):
        rig.ask(message)
    assert "route" not in rig.llm.purposes()


def test_a_web_query_made_only_of_private_values_is_blocked(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(route='{"intent": "search", "query": "anything"}'))
    turn = rig.ask("Léa?")  # a personal name only: the web may see none of it
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
    assert turn.answer.splitlines()[-1].startswith("Receipts today: 2 outbound calls")  # the answer, the brief
    assert rig.ask(BRIEF).intent == "brief"


def test_router_output_that_is_not_json_falls_back_to_keyword_rules(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(route="I think the user wants to search."))
    turn = rig.ask("How should I taper for the half marathon?")
    assert turn.intent == "answer" and rig.llm.purposes() == ["route", "answer"]
    assert "router reply was not valid JSON; used keyword rules" in turn.actions


def test_messages_the_rules_can_route_are_never_shown_to_a_router(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    for message in (TAPER, "Remember that I prefer aisle seats on long trains.", "My gym class moved to Mondays."):
        rig.ask(message)
    assert "route" not in rig.llm.purposes()


def test_route_parsing_is_defensive() -> None:
    assert extract_json('<think>{"draft": 1}</think>Here: ```json\n{"intent": "brief"}\n```') == {"intent": "brief"}
    assert parse_route('{"intent": "SEARCH", "query": " taper  plan "}').query == "taper plan"
    assert parse_route('{"intent": "dance"}') is None and parse_route("no json here") is None


def test_remember_classifies_on_the_device_without_asking_a_model(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(classify='{"sensitivity": "public", "kind": "preference", "label": "seat preference"}'))
    turn = rig.ask("Remember that I prefer aisle seats on long trains.")
    memory = rig.store.get(26)
    assert turn.intent == "remember" and turn.actions == ["recorded m26v1"]
    assert (memory.text, memory.sensitivity, memory.kind) == (
        "I prefer aisle seats on long trains.", "personal", "preference")
    assert rig.llm.calls == []  # neither a router nor a classifier saw the fact
    assert turn.answer.startswith("Saved as [m26v1] (personal preference")


def test_new_private_facts_get_a_category_label_and_no_cloud_call(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    rig.ask("Remember that I have a colonoscopy on 9 Dec at 8:00.")
    rig.ask("Remember that I was diagnosed with type 2 diabetes.")
    assert [(rig.store.get(i).sensitivity, rig.store.get(i).label) for i in (26, 27)] == [
        ("private", "health appointment"), ("private", "health")]
    assert rig.llm.calls == []


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
    assert rig.store.get(26).text == "Gym class moved to Mondays."


def test_a_cloud_router_cannot_point_a_correction_at_a_private_memory(make_rig: Callable[..., Rig]) -> None:
    hostile = '{"intent": "correct", "target": "health appointment", "text": "⟦m7⟧ leaked"}'
    rig = make_rig(replies(route=hostile))
    rig.ask("Please update that for me.")
    assert rig.store.get(7).version == 1 and rig.store.get(7).text.startswith("Dermatologist appointment on")
    assert all("leaked" not in memory.text for memory in rig.store.all_current())  # the router's text is not stored


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
    assert turn.receipts == []  # the one call failed, so no receipt: nothing was received


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
    assert recorded is not None and set(recorded.outputs) == {"answer"} and len(recorded.web_results) == 2
    offline = make_rig(live=False, replay=recorder)
    replayed = offline.ask(TAPER)
    assert replayed.mode == "replay" and replayed.answer == original.answer
    assert [r.mode for r in replayed.receipts] == ["replay", "replay"]


def test_today_comes_from_the_seed_snapshot(make_rig: Callable[..., Rig]) -> None:
    assert make_rig().assistant.today().isoformat() == "2026-10-30"


def test_withheld_events_are_listed_apart_for_any_period_so_none_is_dated(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    rig.ask("What's coming up in the next 3 days?")
    rig.ask("What's coming up in the next 4 days?")
    rig.ask(PLATE)
    answers = [call["messages"][1]["content"] for call in rig.llm.calls if call["purpose"] == "answer"]
    blocks = [prompt.split(WITHHELD_HEADING, 1)[1].split("\n\nRequest:")[0] for prompt in answers]
    assert len(set(blocks)) == 1 and "⟦m7⟧" in blocks[0] and "⟦m11⟧" in blocks[0]
    assert "the 5 days from today" in answers[1] and "3 November" not in answers[1] and "[withheld]" not in answers[1]


def test_a_message_carrying_a_private_value_never_reaches_the_router(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    rig.ask("Is the 1030 slot on 2026/11/03 still free?")
    assert "route" not in rig.llm.purposes()
    assert "1030" not in rig.llm.sent_text() and "2026/11/03" not in rig.llm.sent_text()


# --- availability: checked on the device ---------------------------------------------

def test_availability_is_checked_on_the_device_with_the_private_dates(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask("Am I free Tuesday morning?")
    assert turn.intent == "availability" and rig.llm.calls == [] and rig.tavily.queries == []
    assert turn.receipts == [] and turn.outbound == [] and turn.notice == ""
    assert turn.answer.startswith("Tuesday 3 November 2026, morning: not free.")
    assert "Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30." in turn.answer
    assert "Running club track session every Tuesday at 19:00" in turn.answer.split("Also that day:")[1]
    assert [m.cite() for m in turn.citations] == ["m7v1", "m16v1"]
    assert turn.guard.summary() == "2 private memories never left Confidant"
    assert turn.actions == ["checked availability on this device: nothing sent"]


def test_the_availability_draft_says_the_same_whatever_private_items_the_period_holds(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    busy, free = rig.ask("Am I free Tuesday morning?"), rig.ask("Am I free Monday morning?")
    assert free.answer.startswith("Monday 2 November 2026, morning: free, nothing in memory.")
    assert busy.draft.replace("Tuesday 3", "Monday 2") == free.draft
    asked = "Tuesday 3 Nov"  # the period asked about is in the draft: the user named it
    assert not any(value in busy.draft for value in [*PRIVATE_VALUES, "⟦m7⟧", "not free"] if value != asked)
    assert busy.guard.prepared.withheld_ids == free.guard.prepared.withheld_ids == [7, 11]
    assert [m.id for m in busy.guard.kept_local] == [m.id for m in free.guard.kept_local] == [7, 11]


def test_availability_over_several_days_lists_each_day(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask("When am I free next week?")
    assert turn.answer.splitlines()[0] == (
        "The 7 days from Monday 2 November 2026: free on 2 of 7 days (Mon 2 Nov, Sat 7 Nov).")
    assert "- Thu 5 Nov: Confidential: final-round interview" in turn.answer
    assert "- Wed 4 Nov: Quarterly product review" in turn.draft and "Thu 5 Nov" not in turn.draft
    assert rig.llm.calls == []


def test_an_availability_question_carrying_a_private_value_still_sends_nothing(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask("Is the 1030 slot on 2026/11/03 still free?")
    assert turn.intent == "availability" and rig.llm.calls == []
    assert "not free" in turn.answer and "Dermatologist appointment" in turn.answer


def test_availability_in_replay_needs_no_recording(make_rig: Callable[..., Rig]) -> None:
    turn = make_rig(live=False).ask("Is Wednesday afternoon free?")
    assert turn.mode == "replay" and turn.receipts == [] and turn.notice == ""
    assert turn.answer.startswith("Wednesday 4 November 2026, afternoon: not free.")
    assert "Design critique happens every Wednesday at 14:00" in turn.answer


def test_availability_chosen_by_the_router_does_not_claim_that_nothing_left(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(route='{"intent": "availability"}'))
    turn = rig.ask("Do I have anything on?")
    assert turn.intent == "availability" and rig.llm.purposes() == ["route"] and len(turn.receipts) == 1
    assert "Nothing was sent" not in turn.answer and "Only the routing request left" in turn.answer
    assert turn.actions == ["checked availability on this device"]


def test_a_day_it_cannot_read_is_asked_for_not_replaced_by_today(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    turn = rig.ask("Am I free next month?")
    assert turn.answer.startswith("Which day should I check?") and rig.llm.calls == []


def test_a_question_naming_a_day_never_selects_private_items_by_their_words(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    for question in ("What do I have on Tuesday?", "What do I have on Monday?", "What's on Thursday?"):
        rig.ask(question)
    answers = [call["messages"][1]["content"] for call in rig.llm.calls if call["purpose"] == "answer"]
    blocks = [prompt.split(WITHHELD_HEADING, 1)[1].split("\n\nRequest:")[0] for prompt in answers]
    assert len(answers) == 3 and len(set(blocks)) == 1 and "⟦m7⟧" in blocks[0] and "⟦m11⟧" in blocks[0]


# --- token limits and reasoning, as seen live on 28 Oct 2026 (scripts/probe_reasoning.py) -----------------

def cut_off(text: str, tokens: int, model: str = MODELS["reason"]) -> ChatResult:
    """A reply the model did not finish: it reached max_tokens (finish_reason "length")."""
    return ChatResult(text=text, prompt_tokens=504, completion_tokens=tokens, latency_ms=2908, model=model,
                      finish_reason="length")


UNFINISHED_TRACE = (
    "The user wants a daily brief for the next 7 days from today (Friday 30 October 2026).\n\n"
    "We have memories with dates:\n- m6v1: Quarterly product review on Wednesday 4 Nov 2026 at 15:00\n"
    "Withheld memories: we need to list them after the events. We must not guess"
)


def test_a_brief_cut_off_at_the_token_limit_is_never_shown(make_rig: Callable[..., Rig]) -> None:
    # Live, Nemotron Ultra spent the brief's whole budget reasoning; the unfinished trace came back as the
    # answer text, with no tags, and was shown (and recorded for replay) as the brief.
    recorder = ReplayBook()
    rig = make_rig(replies(brief=cut_off(UNFINISHED_TRACE, REASON_MAX_TOKENS)), recorder=recorder)
    turn = rig.assistant.daily_brief("s1")
    assert "The user wants" not in turn.answer and "We must not guess" not in turn.answer
    assert turn.answer.startswith("The cloud model could not answer this time")
    assert "token limit" in turn.notice and "local fallback" in turn.notice
    assert [r.purpose for r in turn.receipts] == ["brief"]  # the call happened and its tokens are counted
    assert len(recorder) == 0  # an unfinished reply is never kept as a replay


def test_a_router_reply_cut_off_at_the_token_limit_is_not_read_for_an_intent(make_rig: Callable[..., Rig]) -> None:
    # An unfinished trace can hold a draft JSON object the model had not settled on.
    trace = 'The user asks for a birthday message. Maybe {"intent": "search", "query": "birthday wishes"}? But'
    rig = make_rig(replies(route=cut_off(trace, FAST_MAX_TOKENS, MODELS["fast"])))
    turn = rig.ask(BIRTHDAY)
    assert turn.intent == "answer" and rig.llm.purposes() == ["route", "answer"]
    assert "token limit" in turn.notice


def test_the_router_runs_with_reasoning_off_and_the_answers_with_it_on(make_rig: Callable[..., Rig]) -> None:
    # With reasoning on, Nano spent its 256 tokens thinking and returned no route twice out of two.
    rig = make_rig()
    rig.ask(BIRTHDAY)
    rig.assistant.daily_brief("s1")
    assert [(call["purpose"], call["thinking"]) for call in rig.llm.calls] == [
        ("route", False), ("answer", True), ("brief", True)]


def test_the_reasoning_budget_fits_the_longest_trace_seen_live() -> None:
    # docs/token-factory-probe.md: with its trace, the brief took up to 2,146 tokens (the final recording of
    # 28 Oct); 1,024 cut it off.
    assert REASON_MAX_TOKENS >= 2 * 2146


def test_the_router_prompt_makes_answer_the_default_for_the_users_own_life() -> None:
    # With reasoning off and "answer" described last, as "anything else", Nano sent "What's on my plate next
    # week?" and two other questions about the user's own life to a web search (one run, recorded in the
    # history of docs/router-check.md). The new wording did not measurably help (same file).
    from confidant.assistant import ROUTER_SYSTEM_PROMPT

    rules = [line for line in ROUTER_SYSTEM_PROMPT.splitlines() if line.startswith("- ")]
    assert rules[0].startswith("- answer, the default:") and "own plans, schedule" in rules[0]
    assert next(rule for rule in rules if rule.startswith("- search")).startswith("- search: only when")


def test_a_question_about_a_period_of_the_users_own_time_is_never_left_to_the_router(
    make_rig: Callable[..., Rig],
) -> None:
    # Live, with reasoning off and the final prompt, Nano sent "What's on my plate next week?" to a web search in
    # five calls out of seven (docs/token-factory-probe.md): a first-person question naming a period is about the
    # user's own schedule, so the rules route it and nothing reaches a router.
    rig = make_rig(replies(route='{"intent": "search", "query": "my schedule next week"}'))
    turn = rig.ask(PLATE)
    assert turn.intent == "answer" and rig.llm.purposes() == ["answer"]
    assert rig.tavily is not None and rig.tavily.queries == []


def test_a_statement_that_names_a_period_still_reaches_the_router(make_rig: Callable[..., Rig]) -> None:
    # Only a question about a period stays away from the router: "Next week I'm in Paris for work." is news to
    # keep, which the local rules do not see, so the router must still be asked.
    rig = make_rig(replies(route='{"intent": "remember", "text": "In Paris for work next week."}'))
    turn = rig.ask("Next week I'm in Paris for work.")
    assert rig.llm.purposes()[0] == "route" and turn.intent == "remember"


# --- findings of the zero-context review of 28 Oct (period rule, recorder, web search state) ---------------

def test_a_question_about_a_period_with_no_first_person_still_reaches_the_router(
    make_rig: Callable[..., Rig],
) -> None:
    # "Is there a public holiday next week?" asks about the world, not about the user's own time.
    rig = make_rig(replies(route='{"intent": "search", "query": "public holiday next week"}'))
    turn = rig.ask("Is there a public holiday next week?")
    assert rig.llm.purposes()[0] == "route" and turn.intent == "search"


def test_a_first_person_question_about_a_period_is_routed_on_the_device(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig(replies(route='{"intent": "search", "query": "my week"}'))
    turn = rig.ask("What do I have next week?")
    assert turn.intent == "answer" and rig.llm.purposes() == ["answer"]


def test_a_routed_turn_whose_answer_was_cut_off_is_never_recorded(make_rig: Callable[..., Rig]) -> None:
    # The route succeeded, so the turn had outputs; recording them alone would replay a turn with no answer.
    recorder = ReplayBook()
    rig = make_rig(replies(answer=cut_off("The user wants a birthday message", REASON_MAX_TOKENS)), recorder=recorder)
    rig.ask(BIRTHDAY)
    assert rig.llm.purposes() == ["route", "answer"] and len(recorder) == 0


def test_a_routed_turn_whose_answer_call_failed_is_never_recorded(make_rig: Callable[..., Rig]) -> None:
    recorder = ReplayBook()
    rig = make_rig(replies(answer=ServiceError("Nebius Token Factory (answer) had a server error (HTTP 503).")),
                   recorder=recorder)
    rig.ask(BIRTHDAY)
    assert len(recorder) == 0


def web_decision(turn):
    """The guard's decision on the turn's web query."""
    return next(d for d in turn.guard.decisions if d.purpose == "web search")


def test_a_web_query_with_web_search_off_is_marked_as_not_sent(make_rig: Callable[..., Rig]) -> None:
    turn = make_rig(web=False).ask(TAPER)
    decision = web_decision(turn)
    assert decision.allowed and "web search is off" in decision.not_sent
    assert not [o for o in turn.outbound if o.provider == "tavily"]


def test_a_web_search_that_failed_is_not_marked_as_not_sent(make_rig: Callable[..., Rig]) -> None:
    # The query left the device and the search engine failed: saying it was not sent would be false.
    class FailingTavily:
        def search(self, query: str, *, max_results: int = 5):
            raise ServiceError("Tavily did not answer within 15 s.")

    rig = make_rig()
    rig.assistant.search = FailingTavily()
    turn = rig.ask(TAPER)
    assert web_decision(turn).not_sent == "" and "Web search failed" in turn.notice


def test_the_recorder_keeps_whether_web_search_ran(make_rig: Callable[..., Rig]) -> None:
    off, on = ReplayBook(), ReplayBook()
    make_rig(web=False, recorder=off).ask(TAPER)
    make_rig(recorder=on).ask(TAPER)
    assert off.lookup(TAPER).web_search == "off" and on.lookup(TAPER).web_search == "ran"


def test_replaying_a_turn_recorded_with_web_search_off_says_so(make_rig: Callable[..., Rig]) -> None:
    book = ReplayBook([RecordedTurn(TAPER, {"answer": "Race day is Sunday 15 November [m3v1]."},
                                    recorded_at="2026-10-28T06:53:46Z", web_search="off")])
    turn = make_rig(live=False, replay=book).ask(TAPER)
    assert "web search was off when this turn was recorded" in turn.notice.lower()
    assert "web search was off" in web_decision(turn).not_sent


def test_replaying_a_search_that_found_nothing_still_shows_its_request(make_rig: Callable[..., Rig]) -> None:
    book = ReplayBook([RecordedTurn(TAPER, {"answer": "Nothing in memory about that."},
                                    recorded_at="2026-10-28T06:53:46Z", web_search="ran")])
    turn = make_rig(live=False, replay=book).ask(TAPER)
    assert [o.provider for o in turn.outbound].count("tavily") == 1 and web_decision(turn).not_sent == ""
