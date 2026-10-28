"""How a live recording replaces the replay placeholders (scripts/record_demo_turns.sh)."""
from __future__ import annotations

from confidant.replay import RecordedTurn, ReplayBook, merge_recording

PLATE = "What's on my plate next week?"
PIN = "Remember my bank PIN is 4821."
BRIEF = "Run my daily brief."


def placeholder(question: str, outputs: dict) -> dict:
    """A data/replay placeholder as written by hand."""
    return {"question": question, "mode": "replay", "recorded_at": None, "outputs": outputs, "web_results": []}


def test_a_live_turn_replaces_its_placeholder() -> None:
    live = ReplayBook([RecordedTurn(PLATE, {"answer": "Next week: ⟦m7⟧ [m6v1]"}, recorded_at="2026-10-28T06:53:43Z")])
    result = merge_recording({"01.json": placeholder(PLATE, {"route": "{}", "answer": "sample"})}, live)
    assert result.problems == [] and result.writes["01.json"]["mode"] == "recorded live"
    assert result.writes["01.json"]["outputs"] == {"answer": "Next week: ⟦m7⟧ [m6v1]"}


def test_a_turn_that_never_calls_a_model_keeps_its_placeholder() -> None:
    result = merge_recording({"05.json": placeholder(PIN, {})}, ReplayBook())
    assert result.problems == [] and result.kept == ["05.json"] and result.writes == {}


def test_a_missing_answer_writes_nothing_at_all() -> None:
    live = ReplayBook([RecordedTurn(PLATE, {"answer": "x"}, recorded_at="2026-10-28T06:53:43Z")])
    result = merge_recording({"01.json": placeholder(PLATE, {"answer": "sample"}),
                              "06.json": placeholder(BRIEF, {"brief": "sample"})}, live)
    assert result.writes == {} and result.problems == ["06.json: no live output (needs brief)"]


def test_a_live_turn_holding_only_its_route_is_refused() -> None:
    live = ReplayBook([RecordedTurn(BRIEF, {"route": "{}"}, recorded_at="2026-10-28T06:53:53Z")])
    result = merge_recording({"06.json": placeholder(BRIEF, {"brief": "sample"})}, live)
    assert result.writes == {} and "lacks brief" in result.problems[0]
