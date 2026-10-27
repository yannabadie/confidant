"""The synthetic demo persona covers what the spec and the demo script need."""
from __future__ import annotations

import json
import re

from confidant.memory import KINDS, SENSITIVITIES
from confidant.scan import find_dates, find_secrets, find_values

from conftest import PERSONA

DATA = json.loads(PERSONA.read_text(encoding="utf-8"))
BY_LABEL = {entry["label"]: entry for entry in DATA["memories"]}


def test_persona_shape() -> None:
    memories = DATA["memories"]
    assert DATA["persona"]["name"] == "Alex Martin" and DATA["persona"]["city"] == "Lyon"
    assert DATA["as_of"] == "2026-10-30"
    assert [entry["id"] for entry in memories] == list(range(1, len(memories) + 1))
    assert 22 <= len(memories) <= 28
    assert {entry["sensitivity"] for entry in memories} == set(SENSITIVITIES)
    assert all(entry["kind"] in KINDS for entry in memories)
    assert all(re.fullmatch(r"2026-\d\d-\d\dT\d\d:\d\d:\d\dZ", entry["created_at"]) for entry in memories)


def test_required_memories_are_present() -> None:
    expected = {
        "half marathon": ("personal", "15 Nov 2026"),
        "sister's birthday": ("personal", "14 Nov 2026"),
        "health appointment": ("private", "3 Nov 2026 at 10:30"),
        "housing cost": ("private", "€"),
        "private appointment": ("private", "interview"),
        "team tools": ("public", "Figma"),
        "meeting preferences": ("personal", "Meeting"),
        "weekly review": ("personal", "Weekly review"),
    }
    for label, (sensitivity, fragment) in expected.items():
        assert BY_LABEL[label]["sensitivity"] == sensitivity and fragment in BY_LABEL[label]["text"], label
    assert BY_LABEL["weekly review"]["kind"] == "skill"
    assert BY_LABEL["health appointment"]["id"] == 7
    assert "Léa" in BY_LABEL["sister's birthday"]["text"]


def test_event_dates_are_in_november_2026() -> None:
    for entry in DATA["memories"]:
        if entry["kind"] == "event":
            dates = find_dates(entry["text"])
            assert dates and (dates[0].year, dates[0].month) == (2026, 11), entry["label"]


def test_labels_reveal_no_values_and_nothing_looks_like_a_credential() -> None:
    for entry in DATA["memories"]:
        assert find_values(entry["label"]) == [], entry["label"]
        assert find_secrets(entry["text"]) == [] and find_secrets(entry["label"]) == []
