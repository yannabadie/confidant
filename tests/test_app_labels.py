"""Labels of the hosted demo that must stay true to what actually happened."""
from __future__ import annotations

import importlib

from confidant import ReplayBook
from confidant.replay import RecordedTurn


def _app():
    """The Gradio app module (built in replay mode: the test environment has no API key)."""
    return importlib.import_module("app")


def test_hand_written_samples_are_not_called_recorded_turns() -> None:
    app = _app()
    samples = ReplayBook([RecordedTurn("What's on my plate next week?", {"answer": "x"})])
    recorded = ReplayBook([RecordedTurn("What's on my plate next week?", {"answer": "x"},
                                        recorded_at="2026-10-26T21:00:00Z")])
    assert not app.replay_is_recorded(samples)
    assert app.replay_is_recorded(recorded)
    assert not app.replay_is_recorded(ReplayBook())
