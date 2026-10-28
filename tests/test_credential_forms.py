"""The video says a bank PIN is refused before it is stored or sent. On 28 October 2026 a probe of the credential
patterns found common ways of writing one that they let through: the value before the word ("4821 is my bank
PIN"), more than 20 characters between the word and the value ("PIN for the Barclays card is 4821"), numbers in
words other than digit by digit ("forty-eight twenty-one", "four thousand eight hundred twenty one"), and two names
of a card code ("the code to my bank card", "my cash machine code"). Each is reproduced here before its fix, with
ordinary messages that must still pass.
"""
from __future__ import annotations

from collections.abc import Callable

import pytest

from confidant.rules import classify_new_fact
from confidant.scan import detect_secret

from conftest import Rig

MISSED = [
    "4821 is my bank PIN",
    "Remember 4821, it's my bank PIN",
    "Remember: 4 8 2 1 is my PIN.",
    "4821 - that's the door code",
    "PIN for the Barclays card is 4821",
    "The PIN of the card I use for groceries is 4821",
    "My bank PIN is forty-eight twenty-one",
    "my bank pin is four thousand eight hundred twenty one",
    "Remember my PIN: forty eight, twenty one",
    "the code to my bank card is 4821",
    "my cash machine code is 4821",
    "Le 4821, c'est mon code de carte",
]


@pytest.mark.parametrize("message", MISSED)
def test_a_pin_written_this_way_is_refused(message: str) -> None:
    assert detect_secret(message) is not None


@pytest.mark.parametrize("message", ["Remember 4821, it's my bank PIN", "My bank PIN is forty-eight twenty-one"])
def test_a_refused_pin_is_neither_stored_nor_sent(make_rig: Callable[..., Rig], message: str) -> None:
    rig = make_rig()
    before = len(rig.store.all_versions())
    turn = rig.ask(message)
    assert len(rig.store.all_versions()) == before
    assert rig.llm.calls == []
    assert "4821" not in turn.answer and "forty-eight" not in turn.answer


@pytest.mark.parametrize("message", [
    "1500 is the time the door code changes",
    "The door code changes every month, the meeting is at 1500",
    "4821 people signed the petition; pin it to the top",
    "Flight 4821 is my usual one to Lyon",
    "The code review for PR 4821 is due Friday",
    "Order 48213 is the bank card replacement",
    "Pin number 3 on the board is mine",
    "2026 is the year I finally learn my PIN by heart",
    "The PIN for the new card arrives in forty-eight hours",
    "The door code is in the welcome email from the landlord",
    "Twenty-one people are coming, so pin the seating plan",
])
def test_these_ordinary_messages_are_still_not_refused(message: str) -> None:
    assert detect_secret(message) is None


# A code the patterns miss is stored like any new fact, private unless clearly everyday. "Gym" is an everyday word,
# so "the gym locker opens with my birthday digits 0412" was kept as personal, and personal memories may be sent
# when a task needs them. A number next to a lock or unlock word now makes the fact private.

@pytest.mark.parametrize("fact", [
    "The gym locker opens with my birthday digits 0412.",
    "My bike lock opens with 2580, I always forget it on Sundays.",
    "The garage door at home unlocks with 7319 on the keypad.",
])
def test_a_code_the_patterns_miss_is_stored_as_private(fact: str) -> None:
    assert classify_new_fact(fact) == "private"


def test_such_a_code_is_kept_on_the_device(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    rig.ask("Remember the gym locker opens with my birthday digits 0412")
    stored = [memory for memory in rig.store.all_versions() if "0412" in memory.text]
    assert [memory.sensitivity for memory in stored] == ["private"]


@pytest.mark.parametrize("fact", [
    "Léa's birthday dinner is at home on Saturday at 20:30.",
    "The gym opens at 06:30 on weekdays; I like the 07:00 class.",
    "I prefer the 1930 table at the Lebanese place.",
])
def test_everyday_facts_with_numbers_stay_personal(fact: str) -> None:
    assert classify_new_fact(fact) == "personal"
