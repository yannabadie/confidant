"""Replay answers only the exact request that was recorded live.

The final judge of 28 October 2026 stored a memory in the hosted demo, then ran step 1: the Guard tab showed a
request holding the new memory's placeholder (⟦m26⟧) next to an answer recorded for a different request. A
recorded turn now carries the SHA-256 of each request it sent, and replay answers only when the request built
now has the same hash; otherwise no model answered, and the Guard tab says why.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from confidant import Assistant, Budget, Guard, MemoryStore, ReceiptLog
from confidant.replay import RecordedTurn, ReplayBook

from conftest import PERSONA

REPLAY = Path(__file__).resolve().parents[1] / "data" / "replay"
MODEL_TURNS = ["What's on my plate next week?", "Find a 10-day taper plan for my half marathon.",
               "Draft a birthday message for Léa.", "Run my daily brief."]


def demo() -> tuple[MemoryStore, Assistant]:
    store = MemoryStore()
    store.load_seed(PERSONA)
    return store, Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), None,
                            replay=ReplayBook.load(REPLAY))


@pytest.mark.parametrize("question", MODEL_TURNS)
def test_every_recorded_turn_holds_the_hash_of_what_it_sent(question: str) -> None:
    recorded = ReplayBook.load(REPLAY).lookup(question)
    assert recorded is not None and set(recorded.requests) == set(recorded.outputs)


@pytest.mark.parametrize("question", MODEL_TURNS)
def test_the_demo_turns_still_replay(question: str) -> None:
    _, assistant = demo()
    turn = assistant.handle(question, session_id="s")
    nebius = [call for call in turn.outbound if call.provider != "tavily"]
    assert nebius and all(call.status == "answered" for call in nebius)
    assert "differs from the one recorded" not in turn.notice


def test_a_request_changed_by_a_new_memory_is_not_answered_from_the_recording() -> None:
    store, assistant = demo()
    assistant.handle("Remember I have a physiotherapy session on Monday 2 Nov 2026 at 18:00.", session_id="s")
    turn = assistant.handle("What's on my plate next week?", session_id="s")
    recorded = ReplayBook.load(REPLAY).lookup("What's on my plate next week?")
    assert recorded is not None
    assert turn.answer.startswith("The recording of this question answered a different request")
    answer_calls = [call for call in turn.outbound if call.purpose == "answer"]
    assert answer_calls and "⟦m26⟧" in str(answer_calls[0].payload)  # the Guard tab shows the request built now
    assert answer_calls[0].sha256 != recorded.requests["answer"]
    assert answer_calls[0].status.startswith("no answer")
    assert "differs from the one recorded" in turn.notice


def test_a_recording_without_hashes_replays_as_before() -> None:
    old = RecordedTurn.from_dict({"question": "What's on my plate next week?",
                                  "outputs": {"answer": "Next week: [m6v1]."}})
    store = MemoryStore()
    store.load_seed(PERSONA)
    assistant = Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), None, replay=ReplayBook([old]))
    turn = assistant.handle("What's on my plate next week?", session_id="s")
    assert turn.answer.startswith("Next week") and "differs from the one recorded" not in turn.notice
