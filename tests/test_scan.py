"""Offline scanners: value normalisation, dates, secrets and accent folding."""
from __future__ import annotations

from datetime import date

import pytest

from confidant.scan import detect_secret, find_dates, find_values, fold, iban_valid, luhn_valid


def keys(text: str) -> list[str]:
    """Normalised value keys found in ``text``."""
    return [value.key for value in find_values(text)]


@pytest.mark.parametrize("text", ["3 Nov 2026", "Tue 3 Nov", "November 3rd, 2026", "2026-11-03", "03/11/2026",
                                  "the 3rd of November"])
def test_dates_normalise_whatever_the_format(text: str) -> None:
    assert keys(text) == ["date:11-03"]


@pytest.mark.parametrize(("text", "key"), [("10:30", "time:10:30"), ("10h30", "time:10:30"),
                                           ("9am", "time:09:00"), ("2:15 pm", "time:14:15")])
def test_times_normalise(text: str, key: str) -> None:
    assert keys(f"at {text}") == [key]


@pytest.mark.parametrize("text", ["€1,180", "1 180 euros", "1.180", "1180"])
def test_numbers_normalise_and_years_are_ignored(text: str) -> None:
    assert keys(f"Rent {text} in 2026") == ["number:1180"]


def test_names_are_mid_sentence_capitals_without_calendar_words() -> None:
    assert keys("Sister Léa's party is on Friday with Sam") == ["name:lea", "name:sam"]


def test_dates_without_year_resolve_to_the_next_occurrence() -> None:
    today = date(2026, 10, 30)
    assert find_dates("4 Nov")[0].resolve(today) == date(2026, 11, 4)
    assert find_dates("15 Jan")[0].resolve(today) == date(2027, 1, 15)
    assert find_dates("20 Oct")[0].resolve(today) == date(2026, 10, 20)
    assert find_dates("31/02/2026") == []


def test_checksums() -> None:
    assert luhn_valid("4111111111111111") and not luhn_valid("4111111111111112")
    assert iban_valid("GB82WEST12345698765432") and not iban_valid("GB82WEST12345698765431")


def test_ordinary_text_is_not_a_secret() -> None:
    for text in ("Reset my password tomorrow", "A password manager is useful", "Pin the tab at 12",
                 "Order 1234 5678 9012 3456", "Design tokens documentation"):
        assert detect_secret(text) is None, text


def test_fold_maps_back_to_original_characters() -> None:
    folded, origin = fold("Léa’s  Café")
    assert folded == "lea’s cafe"
    assert origin[folded.index("cafe")] == "Léa’s  Café".index("Café")
