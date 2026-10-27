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


def test_the_guard_tab_shows_each_exact_payload_with_the_hash_of_its_receipt() -> None:
    import hashlib
    app = _app()
    engine = app.Engine(app.RUNTIME)
    try:
        view = engine.ask("What's on my plate next week?")
    finally:
        engine.close()
    assert view.outbound, "the replayed turn has an answer call"
    html_text = app.render_guard_panel(view)
    assert "Exactly what left this device" in html_text
    for call in view.outbound:
        assert hashlib.sha256(call.canonical_json.encode("utf-8")).hexdigest() == call.payload_sha256
        assert call.payload_sha256 in html_text
    assert [c.payload_sha256 for c in view.outbound] == [r.payload_sha256 for r in view.receipts]


def test_the_guard_tab_hash_also_matches_for_a_web_search_body() -> None:
    import hashlib
    app = _app()
    engine = app.Engine(app.RUNTIME)
    try:
        view = engine.ask("Find a 10-day taper plan for my half marathon.")
    finally:
        engine.close()
    web = [call for call in view.outbound if call.provider == "tavily"]
    assert web, "the replayed turn has a web search"
    for call in web:
        assert hashlib.sha256(call.canonical_json.encode("utf-8")).hexdigest() == call.payload_sha256
    assert [c.payload_sha256 for c in view.outbound] == [r.payload_sha256 for r in view.receipts]
