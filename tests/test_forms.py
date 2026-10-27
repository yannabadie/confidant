"""Generated written forms of protected values (round 4 of the leak test, and what it generalises to)."""
from __future__ import annotations

import pytest

from confidant import forms
from confidant.forms import FormText, canon, surface_forms, years_of
from confidant.guard import NEBIUS, REDACTED, Guard
from confidant.memory import MemoryStore

YEARS = (2026, 2027)


@pytest.mark.parametrize(("text", "key", "found"), [
    ("Tenho alguma coisa marcada para 3 de novembro?", "date:11-03", "3 de novembro"),
    ("11月3日の予定は何ですか？", "date:11-03", "11月3日"),
    ("Что у меня 3 ноября?", "date:11-03", "3 ноября"),
    ("What do I have on 03NOV?", "date:11-03", "03NOV"),
    ("Anything on 3.XI.2026?", "date:11-03", "3.XI.2026"),
    ("¿Qué tengo el tres de noviembre?", "date:11-03", "tres de noviembre"),
    ("Am I still booked at half ten on Tuesday?", "time:10:30", "half ten"),
    ("Is ten and a half on Tuesday still my slot?", "time:10:30", "ten and a half"),
    ("Treffen wir uns um halb elf?", "time:10:30", "halb elf"),
    ("Is my video call still at eleven sharp?", "time:11:00", "eleven sharp"),
    ("Is eleven-eighty a month too much?", "number:1180", "eleven-eighty"),
    ("Is putting aside six grand realistic?", "number:6000", "six grand"),
    ("¿Es realista ahorrar seis mil euros?", "number:6000", "seis mil"),
    ("Ist es realistisch, sechstausend Euro zu sparen?", "number:6000", "sechstausend"),
])
def test_values_are_found_in_forms_the_scanners_do_not_read(text: str, key: str, found: str) -> None:
    assert [text[start:end] for start, end in FormText(text).spans(key, YEARS)] == [found]


@pytest.mark.parametrize("text", [
    "Find a 10-day taper plan for my half marathon.", "Draft a birthday message for Léa.",
    "What's on my plate next week?", "Once upon a time in Lyon", "Is my rent too high?",
])
def test_harmless_text_matches_no_protected_form(text: str) -> None:
    probe = FormText(text)
    for key in ("date:11-03", "date:11-05", "time:10:30", "time:11:00", "number:1180", "number:6000"):
        assert probe.spans(key, YEARS) == []


def test_forms_have_a_letter_and_leave_digit_only_forms_to_the_scanners() -> None:
    generated = surface_forms("date:11-03", YEARS)
    assert "3 de novembro" in generated and "03 11 2026" not in generated and "3 11" not in generated
    assert all(any(char.isalpha() for char in form) for form in generated)


def test_canon_maps_back_to_the_original_text() -> None:
    text = "On 3​ de Novembro, ok"
    folded, origin = canon(text)
    assert folded == "on 3 de novembro ok"
    start = folded.index("3 de")
    assert text[origin[start]: origin[start + len("3 de novembro") - 1] + 1] == "3​ de Novembro"


def test_years_of_reads_written_years_and_adds_this_year_and_the_next() -> None:
    from datetime import date
    assert years_of("Dermatologist on Tuesday 3 Nov 2026", date(2026, 10, 30)) == (2026, 2027)
    assert years_of("no year here", date(2025, 1, 2)) == (2025, 2026)


def test_the_guard_redacts_a_private_value_written_another_way(store: MemoryStore) -> None:
    decision = Guard().check_outbound("Tenho alguma coisa para 3 de novembro?", NEBIUS, store.all_current(),
                                      strict=True)
    assert decision.redacted_text == f"Tenho alguma coisa para {REDACTED}?"
    assert decision.withheld_ids == [7]
    assert any("a date written another way" in reason for reason in decision.reasons)


def test_without_babel_and_num2words_the_guard_falls_back_to_the_scanners(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(forms, "AVAILABLE", False)
    surface_forms.cache_clear()
    forms._patterns.cache_clear()
    try:
        decision = Guard().check_outbound("Is 3 Nov or 3 de novembro free?", NEBIUS, store.all_current(), strict=True)
        assert decision.redacted_text == f"Is {REDACTED} or 3 de novembro free?"  # the scanners alone miss it
    finally:
        surface_forms.cache_clear()
        forms._patterns.cache_clear()


def test_a_protected_phone_number_does_not_hang_the_guard(store: MemoryStore) -> None:
    import signal

    def give_up(*_: object) -> None:
        raise TimeoutError("form generation hung")

    previous = signal.signal(signal.SIGALRM, give_up)
    signal.alarm(10)
    try:
        assert surface_forms("number:612345678") == frozenset()  # above MAX_SPOKEN: never written in words
        memory = store.record("The dermatologist's office number is 06 12 34 56 78.", kind="fact",
                              sensitivity="private", source_type="user", source_ref="t", author="user", label="health")
        decision = Guard().check_outbound("Call 06 12 34 56 78 please", NEBIUS, store.all_current(), strict=True)
        assert memory.id in decision.withheld_ids
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize(("text", "key"), [
    ("huit heures et demie", "time:08:00"), ("dix-sept heures", "time:19:00"), ("seis mil quinientos", "number:6000"),
    ("half ten thousand pounds", "time:10:30"), ("an hour and a half. Ten minutes", "time:10:30"),
    ("squats 3 x 10", "date:10-03"), ("iPhone 15 Pro", "date:12-15"), ("10 min break", "number:1000"),
    ("recycling bin", "number:1000"), ("Step 3. I think", "date:01-03"), ("3 Mac laptops", "date:03-03"),
    ("best of 3 set", "date:09-03"), ("las dos opciones", "time:14:00"), ("As duas propostas", "time:14:00"),
    ("Tu as onze ans", "time:11:00"), ("vingt-trois novembre", "date:11-03"), ("vingt trois novembre", "date:11-03"),
    ("11月3日", "date:01-03"), ("六千五百", "number:6000"), ("千百八十五", "number:1180"), ("een uur later", "time:13:00"),
])
def test_forms_inside_other_values_or_everyday_words_do_not_match(text: str, key: str) -> None:
    assert FormText(text).spans(key, YEARS) == []


@pytest.mark.parametrize(("text", "key", "found"), [
    ("rendez-vous à une heure", "time:13:00", "une heure"), ("um ein Uhr", "time:13:00", "ein Uhr"),
    ("a la una en punto", "time:13:00", "a la una en punto"), ("ci vediamo all'una", "time:13:00", "all'una"),
    ("às uma hora", "time:13:00", "às uma hora"), ("à vingt et une heures", "time:21:00", "vingt et une heures"),
    ("bin yüz seksen lira", "number:1180", "bin yüz seksen"), ("eleven sharp, two of us", "time:11:00", "eleven sharp"),
    ("3 de novembro 10:30", "date:11-03", "3 de novembro"),
])
def test_hours_and_amounts_are_written_as_people_say_them(text: str, key: str, found: str) -> None:
    assert [text[start:end] for start, end in FormText(text).spans(key, YEARS)] == [found]
