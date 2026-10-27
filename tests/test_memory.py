"""MemoryStore: versions, provenance, credential refusal, search and seeding."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from confidant.memory import Memory, MemoryNotFound, MemoryStore, SecretRefused, VersionConflict

from conftest import PERSONA


def note(store: MemoryStore, text: str, **overrides: str) -> Memory:
    """Record a user memory with sensible defaults."""
    fields = {"kind": "fact", "sensitivity": "personal", "source_type": "user", "source_ref": "conv-1",
              "author": "user", "label": "note"}
    fields.update(overrides)
    return store.record(text, **fields)


def iban(country: str, bban: str) -> str:
    """A syntactically valid IBAN built from a made-up account number (check digits computed)."""
    digits = "".join(str(int(char, 36)) for char in bban + country + "00")
    return f"{country}{98 - int(digits) % 97:02d}{bban}"


def test_record_assigns_id_version_and_provenance() -> None:
    store = MemoryStore()
    memory = note(store, "  Prefers   window seats. ", label="seat preference")
    assert (memory.id, memory.version, memory.status, memory.supersedes) == (1, 1, "current", None)
    assert memory.text == "Prefers window seats."
    assert memory.cite() == "m1v1"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", memory.created_at)
    assert store.get(1) == memory
    assert note(store, "Second memory.").id == 2


def test_correct_adds_a_version_and_supersedes_the_old_one(store: MemoryStore) -> None:
    updated = store.correct(7, "Dermatologist appointment on Wednesday 4 Nov 2026 at 9:00.",
                            author="user", source_ref="conv-2", expected_version=1)
    assert (updated.version, updated.supersedes, updated.status) == (2, 1, "current")
    assert (updated.sensitivity, updated.label, updated.source_type) == ("private", "health appointment", "user")
    history = store.history(7)
    assert [(m.version, m.status) for m in history] == [(1, "superseded"), (2, "current")]
    assert history[0].text == "Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30."
    assert store.get(7) == updated


def test_stale_expected_version_raises_conflict_and_changes_nothing(store: MemoryStore) -> None:
    store.correct(7, "Moved to Wednesday 4 Nov 2026 at 9:00.", author="user", source_ref="c", expected_version=1)
    with pytest.raises(VersionConflict) as caught:
        store.correct(7, "Moved again.", author="other-tab", source_ref="c", expected_version=1)
    assert (caught.value.expected, caught.value.actual) == (1, 2)
    assert "version 2" in str(caught.value)
    assert len(store.history(7)) == 2


def test_retract_hides_the_memory_but_keeps_its_history(store: MemoryStore) -> None:
    retracted = store.retract(19, author="user", reason="no longer drinks coffee")
    assert (retracted.status, retracted.version, retracted.note) == ("retracted", 2, "no longer drinks coffee")
    assert 19 not in {m.id for m in store.all_current()}
    assert store.search("coffee") == []
    assert [m.status for m in store.history(19)] == ["superseded", "retracted"]
    with pytest.raises(ValueError):
        store.retract(19, author="user", reason="again")
    with pytest.raises(ValueError):
        store.correct(19, "Back on coffee.", author="user", source_ref="c", expected_version=2)


@pytest.mark.parametrize(
    ("text", "explains"),
    [
        ("Remember my bank PIN is 4821", "PIN"),
        ("My card is 4111 1111 1111 1111", "Luhn"),
        ("The wifi password is hunter22", "password"),
        ("Deploy key: sk-" + "fake" * 8, "API key"),
        ("token = " + "x9" * 20, "API key"),
        ("Rent goes to " + iban("FR", "30006000011234567890189"), "IBAN"),
    ],
)
def test_credentials_are_refused_with_an_explanation(text: str, explains: str) -> None:
    store = MemoryStore()
    with pytest.raises(SecretRefused) as caught:
        note(store, text)
    message = str(caught.value)
    assert explains in message and "Not stored" in message
    assert all(chunk not in message for chunk in ("4821", "4111", "hunter22", "fakefake"))
    assert store.all_current() == []


def test_credential_in_a_label_or_a_correction_is_refused(store: MemoryStore) -> None:
    with pytest.raises(SecretRefused):
        note(store, "Bank details", label="pin 4821")
    with pytest.raises(SecretRefused):
        store.correct(10, "Rent card 4111 1111 1111 1111", author="user", source_ref="c", expected_version=1)
    assert store.get(10).version == 1


def test_long_numbers_that_fail_luhn_are_not_cards() -> None:
    assert note(MemoryStore(), "Order reference 1234 5678 9012 3456 arrives Monday.").version == 1


def test_search_ignores_case_accents_plurals_and_fts_syntax(store: MemoryStore) -> None:
    assert 4 in {m.id for m in store.search("LEA")}
    assert [m.id for m in store.search("appointments")][:1] == [7]
    assert store.search('what\'s "on" (my) plate* AND -NOT ^?') == []
    assert store.search("the and of") == []
    assert store.search("   ") == []


def test_search_filters_by_sensitivity_and_limit(store: MemoryStore) -> None:
    assert {m.sensitivity for m in store.search("running race marathon", sensitivities=["personal"])} == {"personal"}
    assert store.search("dermatologist", sensitivities=["public", "personal"]) == []
    assert len(store.search("running race marathon coffee lea", limit=2)) == 2
    with pytest.raises(ValueError):
        store.search("rent", sensitivities=["secret"])


def test_search_finds_the_corrected_text_only(store: MemoryStore) -> None:
    store.correct(19, "Usual coffee order: double espresso.", author="user", source_ref="c", expected_version=1)
    assert [m.cite() for m in store.search("espresso")] == ["m19v2"]
    assert store.search("flat white") == []


def test_load_seed_keeps_ids_provenance_and_snapshot_date(store: MemoryStore) -> None:
    memories = store.all_current()
    assert [m.id for m in memories] == list(range(1, 26))
    assert {m.source_type for m in memories} == {"seed"} and {m.source_ref for m in memories} == {"seed"}
    assert {m.author for m in memories} == {"Alex Martin"}
    assert store.as_of() == date(2026, 10, 30)
    with pytest.raises(ValueError, match="already exists"):
        store.load_seed(PERSONA)


def test_file_backed_store_persists(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "memory.db"
    first = MemoryStore(path)
    note(first, "Prefers aisle seats.")
    first.correct(1, "Prefers window seats.", author="user", source_ref="c", expected_version=1)
    first.close()
    reopened = MemoryStore(path)
    assert reopened.get(1).cite() == "m1v2"
    assert [m.text for m in reopened.search("window")] == ["Prefers window seats."]


def test_invalid_fields_and_unknown_ids_are_rejected() -> None:
    store = MemoryStore()
    with pytest.raises(ValueError, match="kind"):
        note(store, "Something", kind="rumour")
    with pytest.raises(ValueError, match="text"):
        note(store, "   ")
    with pytest.raises(MemoryNotFound):
        store.get(99)
    with pytest.raises(MemoryNotFound):
        store.history(99)


def test_memory_to_dict_includes_citation() -> None:
    data = note(MemoryStore(), "Likes peonies.").to_dict()
    assert data["cite"] == "m1v1" and data["text"] == "Likes peonies." and data["supersedes"] is None
