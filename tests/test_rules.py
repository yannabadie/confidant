"""Local rules: availability questions, the periods and parts of a day they name, and item times."""
from __future__ import annotations

from datetime import date

import pytest

from confidant.memory import Memory
from confidant.rules import (
    event_date, is_availability_question, item_span, part_of_day, period_asked, rule_route, unread_period,
)

FRIDAY = date(2026, 10, 30)  # the demo persona's today
MONDAY = date(2026, 11, 2)


@pytest.mark.parametrize("message", [
    "Am I free Tuesday morning?", "Am I free next Tuesday morning?", "Is Friday evening free?",
    "Do I have anything tomorrow?", "Is my calendar clear on Wednesday afternoon?", "When am I free next week?",
    "Am I busy this weekend?", "Am I free?", "Is the 1030 slot on 2026/11/03 still free?",
    "Suis-je libre mardi matin ?", "Ai-je quelque chose demain après-midi ?",
])
def test_availability_questions_are_routed_to_the_device(message: str) -> None:
    assert rule_route(message).intent == "availability"


@pytest.mark.parametrize("message, intent", [
    ("What's on my plate next week?", "answer"),
    ("Do I have a meeting with Sam?", "answer"),  # no period named: a question for memory, not a calendar check
    ("Is the shop open on Sunday?", "answer"),
    ("Remember that I'm busy every Monday morning.", "remember"),
    ("My dermatologist appointment moved to Wednesday 4 Nov at 9:00.", "correct"),
    ("Find a 10-day taper plan for my half marathon.", "search"),
])
def test_other_messages_keep_their_intent(message: str, intent: str) -> None:
    assert rule_route(message).intent == intent
    assert is_availability_question(message) is False


@pytest.mark.parametrize("message, today, expected", [
    ("Am I free Tuesday morning?", FRIDAY, date(2026, 11, 3)),
    ("Am I free next Tuesday?", FRIDAY, date(2026, 11, 3)),
    ("Am I free on Tuesday next week?", FRIDAY, date(2026, 11, 3)),
    ("Am I free this Friday?", FRIDAY, FRIDAY),  # "this Friday", said on a Friday, is today
    ("Am I free Friday?", FRIDAY, date(2026, 11, 6)),  # a bare weekday is the next one after today
    ("Suis-je libre mardi ?", FRIDAY, date(2026, 11, 3)),
    ("Am I free next Tuesday?", MONDAY, date(2026, 11, 10)),  # the Tuesday of next week, not tomorrow
    ("Am I free Tuesday?", MONDAY, date(2026, 11, 3)),
    ("Am I free tomorrow?", FRIDAY, date(2026, 10, 31)),
    ("Am I free on 5 Nov?", FRIDAY, date(2026, 11, 5)),
])
def test_period_asked_reads_weekdays_relative_to_today(message: str, today: date, expected: date) -> None:
    assert period_asked(message, today) == (expected, expected)


def test_period_asked_reads_weeks_and_weekends() -> None:
    assert period_asked("When am I free next week?", FRIDAY) == (MONDAY, date(2026, 11, 8))
    assert period_asked("Am I busy this weekend?", FRIDAY) == (date(2026, 10, 31), date(2026, 11, 1))
    assert period_asked("Am I free?", FRIDAY) is None


@pytest.mark.parametrize("message, expected", [
    ("Am I free Tuesday morning?", ("morning", 360, 720)),
    ("Suis-je libre mardi après-midi ?", ("afternoon", 720, 1080)),  # not lunchtime, though it contains "midi"
    ("Am I free tonight?", ("evening", 1080, 1440)),
    ("Am I free at 11:00 on Thursday?", ("around 11:00", 600, 720)),
    ("Am I free on Thursday?", None),
])
def test_part_of_day(message: str, expected: tuple[str, int, int] | None) -> None:
    assert part_of_day(message) == expected


def memory(text: str) -> Memory:
    return Memory(id=1, version=1, text=text, kind="event", sensitivity="personal", status="current",
                  source_type="user", source_ref="t", author="user", created_at="2026-10-30T00:00:00Z",
                  supersedes=None, label="test")


@pytest.mark.parametrize("text, expected", [
    ("Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30.", (630, 631)),
    ("Weekly review, every Friday at 17:00: clear the inbox.", (1020, 1021)),
    ("Drinks at 7pm at the usual place.", (1140, 1141)),
    ("Rendez-vous chez le dentiste à 10h30.", (630, 631)),
    ("Long run every Sunday morning along the Rhône quays at an easy 5:45 per km.", (360, 720)),  # a pace, not a time
    ("Half marathon on Sunday 15 Nov 2026, with a goal of finishing under 1:50.", None),  # a goal, not a time
    ("Start a 10-day taper on Thursday.", None),
])
def test_item_span_reads_start_times_not_paces_or_goals(text: str, expected: tuple[int, int] | None) -> None:
    assert item_span(memory(text)) == expected


@pytest.mark.parametrize("message", [
    "Am I still free on Tuesday?", "Is the afternoon free on Thursday?", "Got anything on Thursday?",
    "What's free on Tuesday?", "Can I fit a run in on Tuesday morning?", "Any gaps on Thursday?",
    "Suis-je toujours libre mardi ?",
])
def test_more_ways_of_asking_are_availability_questions(message: str) -> None:
    assert rule_route(message).intent == "availability"


def test_an_explicit_web_search_and_a_price_are_not_availability() -> None:
    assert rule_route("Is Sunday free entry at the Musée des Confluences? Look it up online.").intent == "search"
    assert rule_route("Is Sunday free entry at the museum?").intent == "answer"


def test_item_span_keeps_pm_after_minutes_and_skips_days_of_the_month() -> None:
    assert item_span(memory("Call with the bank on Tuesday 3 Nov at 2:30pm.")) == (870, 871)
    assert item_span(memory("Yoga every Tuesday evening at 6.30pm.")) == (1110, 1111)
    moved = memory("Dermatologist appointment was rescheduled from 3 Nov to 4 Nov at 9:00.")
    assert item_span(moved) == (540, 541)  # not "from 3" read as 03:00
    assert event_date(moved, FRIDAY) == date(2026, 11, 4)  # the date it moved to
    assert item_span(memory("Meeting moved from 10:30 to 9:00 on Monday 2 Nov.")) == (540, 541)
    assert event_date(memory("Holiday from 3 Nov to 10 Nov."), FRIDAY) == date(2026, 11, 3)


@pytest.mark.parametrize("message, expected", [
    ("Am I free the day after tomorrow?", (date(2026, 11, 1), date(2026, 11, 1))),
    ("Suis-je libre après-demain ?", (date(2026, 11, 1), date(2026, 11, 1))),
    ("Am I free on Thursday the 12th?", (date(2026, 11, 12), date(2026, 11, 12))),
    ("Suis-je libre jeudi 12 ?", (date(2026, 11, 12), date(2026, 11, 12))),
    ("Am I free on 3/11?", (date(2026, 11, 3), date(2026, 11, 3))),  # day first, the nearer reading
    ("Is the 10:30 slot on 11/03/2026 still free?", (date(2026, 11, 3), date(2026, 11, 3))),  # not last March
    ("Am I free between Tuesday and Thursday?", (date(2026, 11, 3), date(2026, 11, 5))),
    ("Am I free this afternoon?", (FRIDAY, FRIDAY)),
])
def test_periods_resolve_to_the_day_asked(message: str, expected: tuple[date, date]) -> None:
    assert period_asked(message, FRIDAY) == expected


def test_a_period_that_cannot_be_read_is_flagged_not_replaced_by_today() -> None:
    assert unread_period("Am I free next month?") and unread_period("Am I free in December?")
    assert not unread_period("Am I free?") and not unread_period("Am I free at 11 on Thursday?")


@pytest.mark.parametrize("message, expected", [
    ("Is the 1030 slot on Wednesday still open?", ("around 10:30", 570, 690)),
    ("Remind me, am I booked at 1100 on Thursday?", ("around 11:00", 600, 720)),
    ("Am I free at 11 on Thursday?", ("around 11:00", 600, 720)),
    ("Am I free at eleven on Thursday?", ("around 11:00", 600, 720)),
    ("Am I free at 3 on Friday?", ("around 15:00", 840, 960)),  # a bare hour from 1 to 6 is in the afternoon
])
def test_part_of_day_reads_times_the_scanners_leave_as_numbers(message: str, expected: tuple[str, int, int]) -> None:
    assert part_of_day(message) == expected
