"""The Guard tab's "Prove it" check (confidant/proof.py): the requests a turn sends are the same, byte for byte,
when every private appointment moves to another week, weekday and hour. A scout comparing the other entries
of the hackathon on 28 October 2026 found that placeholders and receipts are common, and that hiding *when*
a private item happens is what sets Confidant apart, but nothing let a visitor see it. The check rebuilds the
turn offline, with no model call, so it also works in the replayed demo.
"""
from __future__ import annotations

from datetime import date

import pytest

from confidant import MemoryStore, ReceiptLog
from confidant.proof import moved_text, prove
from confidant.replay import ReplayBook
from confidant.guard import Guard

from conftest import PERSONA
from test_replay_exact import REPLAY

QUESTIONS = ["What's on my plate next week?", "What do I have on Tuesday?", "Anything on Thursday at 11:00?",
             "Plan my weekend", "What's on 5 Nov?"]


def seed() -> MemoryStore:
    store = MemoryStore()
    store.load_seed(PERSONA)
    return store


def test_the_private_appointments_move_to_another_week_and_hour() -> None:
    store = seed()
    texts = {memory.id: moved_text(memory, date(2026, 10, 30), Guard()) for memory in store.all_current()}
    assert texts[7] == "Dermatologist appointment on Thursday 12 Nov 2026 at 11:45."
    assert texts[11].startswith("Confidential: final-round interview for a lead designer role on Saturday 14 Nov 2026 at 12:15")
    assert texts[6] == store.get(6).text  # a shared event stays as it is


@pytest.mark.parametrize("question", QUESTIONS)
def test_what_leaves_does_not_depend_on_when_they_happen(question: str) -> None:
    proof = prove(seed(), question)
    assert {move.memory_id for move in proof.moves} == {7, 11}
    assert proof.identical, (proof.original, proof.moved)


def test_the_brief_does_not_either() -> None:
    proof = prove(seed(), "Daily brief", intent="brief")
    assert proof.identical and [purpose for purpose, _ in proof.original] == ["brief"]


def test_the_original_request_is_the_one_recorded_live() -> None:
    proof = prove(seed(), "What's on my plate next week?")
    recorded = ReplayBook.load(REPLAY).lookup("What's on my plate next week?")
    assert recorded is not None and dict(proof.original)["answer"] == recorded.requests["answer"]


def test_a_date_typed_back_is_masked_for_nebius_and_shown_here() -> None:
    from conftest import FakeLLM, MODELS
    from confidant import Assistant, Budget

    def echo(messages):  # a model that writes the request back, as a draft does
        return [line for line in messages[-1]["content"].splitlines() if line.startswith("Request:")][0][9:]

    store = seed()
    llm = FakeLLM({"route": '{"intent": "answer"}', "answer": echo})
    turn = Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), llm, models=MODELS).handle(
        "Draft a note for the team: the retro moves to Thursday 5 Nov at 11:00.", session_id="s")
    sent = [line for call in llm.calls for m in call["messages"] if m["role"] == "user"
            for line in m["content"].splitlines() if line.startswith("Request:")]
    assert sent == ["Request: Draft a note for the team: the retro moves to [date] at [time]."]
    assert "Thursday 5 Nov" in turn.answer and "11:00" in turn.answer  # put back on this device


def test_moving_a_shared_event_changes_the_request() -> None:
    # The check is not vacuous: a memory Nemotron does receive changes what leaves when it changes.
    def move_the_review(memory, today):
        return memory.text.replace("Wednesday 4 Nov", "Thursday 12 Nov") if memory.id == 6 else memory.text

    proof = prove(seed(), "What's on my plate next week?", rewrite=move_the_review)
    assert proof.original and not proof.identical


def test_the_session_memory_and_receipts_are_untouched() -> None:
    store, receipts = seed(), ReceiptLog()
    before = [memory.to_dict() for memory in store.all_versions()]
    prove(store, "What's on my plate next week?")
    assert [memory.to_dict() for memory in store.all_versions()] == before
    assert receipts.today() == []


@pytest.mark.parametrize("question, target", [
    ("What do I have on Thursday 12 Nov?", "Thursday 12 Nov 2026"),
    ("Anything on 11月4日?", "Wednesday 4 Nov 2026"),  # a judge's counter-example of 29 Oct (round 10)
    ("Anything at half ten?", "10:30"),
])
def test_the_appointments_also_move_onto_the_date_or_time_asked_about(question: str, target: str) -> None:
    proof = prove(seed(), question)
    assert proof.aimed_at == target and proof.aimed_moves
    assert all(target.split(" at ")[0] in move.after for move in proof.aimed_moves)
    assert proof.identical, (proof.original, proof.moved, proof.aimed)


def test_a_question_that_names_no_date_or_time_moves_them_once() -> None:
    proof = prove(seed(), "What's on my plate next week?")
    assert proof.aimed_at == "" and proof.aimed_moves == [] and proof.identical
