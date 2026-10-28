"""Round 10, 29 October 2026: a second zero-context review of the submission (three judges: a Devpost reader, a
technical skeptic with the code and no network, and a judge who knows the other entries of the track).

The technical judge ran the Prove it check on questions a visitor could type and found that 7 of 16 changed when
the private appointments moved. A date or time the user typed in a form the scanners do not read ("half ten",
"3 de novembro", "11月3日", "03NOV", "eleven sharp", "halb elf") was masked only when it equalled a private value,
because only the private values' written forms were looked for: "Anything at half ten?" reached Nebius as
"Anything at [withheld]?" while the dermatologist appointment was at 10:30, and as typed otherwise. A shared
memory written that way ("Call with the landlord ... at half ten") lost its time the same way. The fix looks for
the written forms of every date and of every time of day on a five-minute grid, whatever the memories hold.

The same judge stored "dinner with the Acme hiring panel" and found it personal (an everyday word won), so it was
sent with its date; likewise a sponsor, leaving a husband, a pregnancy and a performance improvement plan.
"""
from __future__ import annotations

import pytest

from confidant import Assistant, Budget, Guard, MemoryStore, ReceiptLog
from confidant.assistant import masked, masked_calendar
from confidant.forms import calendar_form_spans
from confidant.proof import prove
from confidant.rules import classify_new_fact, make_label

from conftest import MODELS, PERSONA, FakeLLM

# The questions of tests/test_review_round8.py that an agent sends, typed in the chat instead, and the forms the
# technical judge added.
TYPED = [
    "Anything at half ten?",
    "Tenho alguma coisa marcada para 3 de novembro?",
    "11月3日は何がありますか?",
    "Anything on 03NOV?",
    "Am I free at eleven sharp?",
    "Habe ich um halb elf etwas?",
    "Anything on 11月4日?",
]


def seed() -> MemoryStore:
    store = MemoryStore()
    store.load_seed(PERSONA)
    return store


def sent_requests(store: MemoryStore, message: str) -> list[str]:
    llm = FakeLLM({"route": '{"intent": "answer"}', "answer": "ok"})
    Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), llm, models=MODELS).handle(message, session_id="s")
    return [line for call in llm.calls for m in call["messages"] if m["role"] == "user"
            for line in m["content"].splitlines() if line.startswith("Request:")]


@pytest.mark.parametrize("question", TYPED)
def test_a_typed_date_or_time_in_any_form_does_not_depend_on_private_items(question: str) -> None:
    proof = prove(seed(), question)
    assert proof.identical, (proof.original, proof.moved)


def test_half_ten_is_masked_as_a_time_whether_or_not_a_private_item_is_then() -> None:
    assert sent_requests(seed(), "Anything at half ten?") == ["Request: Anything at [time]?"]
    empty = MemoryStore()  # no private memory at all
    assert sent_requests(empty, "Anything at half ten?") == ["Request: Anything at [time]?"]


def test_written_forms_of_any_date_or_time_are_found_without_private_data() -> None:
    kinds = [(text[start:end], kind) for text in ["Anything at half ten?", "marcada para 3 de novembro",
                                                     "11月3日は", "on 03NOV", "um halb elf", "at eleven sharp"]
             for start, end, kind in calendar_form_spans(text)]
    assert kinds == [("half ten", "time"), ("3 de novembro", "date"), ("11月3日", "date"), ("03NOV", "date"),
                     ("halb elf", "time"), ("eleven sharp", "time")]
    assert calendar_form_spans("Draft a birthday message for Léa, she loves peonies.") == []


def test_the_masks_cover_written_forms_for_users_and_agents_alike() -> None:
    assert masked_calendar("Anything at half ten on 3 de novembro?") == "Anything at [time] on [date]?"
    assert masked("Anything at half ten on 3 de novembro?") == "Anything at [time] on [date]?"


def test_the_answer_shown_here_gets_the_typed_time_back() -> None:
    def echo(messages):
        return [line for line in messages[-1]["content"].splitlines() if line.startswith("Request:")][0][9:]

    llm = FakeLLM({"route": '{"intent": "answer"}', "answer": echo})
    turn = Assistant(seed(), Guard(), ReceiptLog(), Budget(1_000_000, 100), llm, models=MODELS).handle(
        "Draft a note: the retro moves to half ten.", session_id="s")
    assert "half ten" in turn.answer


def test_a_shared_memory_keeps_its_own_time_written_in_words() -> None:
    store = seed()
    store.record("Call with the landlord on Monday 2 Nov 2026 at half ten.", kind="event", sensitivity="personal",
                 source_type="user", source_ref="chat", author="user", label="call")
    proof = prove(store, "What's on my plate next week?")
    assert proof.identical, (proof.original, proof.moved)
    llm = FakeLLM({"route": '{"intent": "answer"}', "answer": "ok"})
    Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), llm, models=MODELS).handle(
        "What's on my plate next week?", session_id="s")
    sent = "\n".join(m["content"] for call in llm.calls for m in call["messages"])
    assert "Call with the landlord on Monday 2 Nov 2026 at half ten." in sent


@pytest.mark.parametrize("fact, category", [
    ("Remember that I have dinner with the Acme hiring panel on Thursday 5 Nov at 20:00.", "career"),
    ("Remember that HR put me on a performance improvement plan.", "career"),
    ("Remember that I'm having brunch with my sponsor after Sunday's meeting.", "personal matter"),
    ("Remember that I'm leaving my husband after Christmas.", "personal matter"),
    ("Remember that I'm 8 weeks along, and we have not told anyone.", "health"),
])
def test_sensitive_facts_with_an_everyday_word_stay_private(fact: str, category: str) -> None:
    assert classify_new_fact(fact) == "private"
    assert make_label(fact, "private").startswith(category)


def test_everyday_facts_stay_personal() -> None:
    assert classify_new_fact("Remember that I have dinner with Léa on Friday at 20:00.") == "personal"
    assert classify_new_fact("Remember that our team uses the new design system checklist.") == "public"


@pytest.mark.parametrize("typed, sent", [
    ("Remind me what 20261103 holds.", "Remind me what [date] holds."),
    ("Remind me, wasn't it first set for 1030?", "Remind me, wasn't it first set for [time]?"),
    ("Anything on 3.XI.2026?", "Anything on [date]?"),
    ("Confirming 3-11, 1030. Still good?", "Confirming [date], [time]. Still good?"),
    ("I ran 1030 km this year.", "I ran 1030 km this year."),
    ("It is for 1030 euros.", "It is for 1030 euros."),
    ("Room 1030 is on the left.", "Room 1030 is on the left."),
])
def test_dates_and_times_in_bare_digits_are_masked_by_their_shape_alone(typed: str, sent: str) -> None:
    # Caught before only by the digit signature of a private value, which showed that they equalled one.
    assert masked_calendar(typed) == sent
