"""ReplayBook: file formats, lookup by normalised question, save and load."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from confidant.replay import RecordedTurn, ReplayBook, normalize_question
from confidant.search import WebResult


def test_questions_are_normalised() -> None:
    assert normalize_question("What’s on my  PLATE, next week?") == "what is on my plate next week"
    assert normalize_question("whats on my plate next week") == "what is on my plate next week"
    assert normalize_question("Draft a birthday message for Léa.") == "draft a birthday message for lea"


def test_all_file_shapes_load(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps([{"question": "Q one?", "outputs": {"answer": "A1"}}]))
    (tmp_path / "b.json").write_text(json.dumps({"turns": [{"question": "Q two", "answer": "A2",
                                                            "route": {"intent": "answer"}}]}))
    (tmp_path / "c.json").write_text(json.dumps({
        "question": "Q three", "outputs": {"answer": "A3"}, "models": {"reason": "nvidia/x"},
        "web_results": [{"title": "T", "url": "https://example.org", "content": "C", "score": 0.5}],
    }))
    (tmp_path / "notes.txt").write_text("ignored")
    book = ReplayBook.load(tmp_path)
    assert len(book) == 3 and book.questions() == ["Q one?", "Q two", "Q three"]
    two = book.lookup("q two")
    assert two is not None and two.outputs == {"answer": "A2", "route": '{"intent": "answer"}'}
    three = book.lookup("Q three")
    assert three is not None and three.web_results == [WebResult("T", "https://example.org", "C", 0.5)]
    assert three.models == {"reason": "nvidia/x"}


RECORDED = ReplayBook([
    RecordedTurn("What's on my plate next week?", {"answer": "A"}),
    RecordedTurn("Draft a birthday message for Léa.", {"answer": "B"}),
    RecordedTurn("Find a 10-day taper plan for my half marathon.", {"answer": "C"}),
])


@pytest.mark.parametrize("asked", [
    "whats on my plate next week", "What is on my plate next week", "WHAT'S ON MY PLATE NEXT WEEK!!",
    "Draft a birthday message for Lea", "find a 10 day taper plan for my half-marathon",
])
def test_the_recorded_question_matches_in_any_case_punctuation_or_contraction(asked: str) -> None:
    assert RECORDED.lookup(asked) is not None


@pytest.mark.parametrize("asked", [
    # outside review, 28 Oct: each of these was served the recording of another question (fuzzy match at 0.88)
    "Draft a birthday message for Sam.", "What's on my plate this week?", "What's on my plate next weekend?",
    "Find a 3-day taper plan for my half marathon.", "What's the weather tomorrow?",
])
def test_a_different_question_never_gets_a_recorded_answer(asked: str) -> None:
    assert RECORDED.lookup(asked) is None


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    turn = RecordedTurn("Find a taper plan", {"route": '{"intent": "search"}', "answer": "Taper [m26v1]"},
                        [WebResult("T", "https://example.org/t", "C", 0.7)], {"fast": "nvidia/nano"},
                        "2026-10-28T10:00:00Z")
    path = tmp_path / "replay" / "recorded.json"
    ReplayBook([turn]).save(path)
    assert ReplayBook.load(path).lookup("find a taper plan") == turn


def test_missing_path_is_an_empty_book_and_bad_files_name_themselves(tmp_path: Path) -> None:
    assert len(ReplayBook.load(tmp_path / "absent")) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"outputs": {"answer": "no question"}}]))
    with pytest.raises(ValueError, match=r"bad\.json"):
        ReplayBook.load(bad)
    bad.write_text("{not json")
    with pytest.raises(ValueError, match=r"bad\.json"):
        ReplayBook.load(tmp_path)
