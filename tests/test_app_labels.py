"""Labels of the hosted demo that must stay true to what actually happened."""
from __future__ import annotations

import importlib

from confidant import ReplayBook, WebResult
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


def test_a_turn_that_made_no_model_call_does_not_hide_a_live_recording() -> None:
    # The PIN refusal and the private correction never call a model, so a live run has nothing to record
    # for them; their files keep recorded_at empty, and the demo then called every recorded turn a sample.
    app = _app()
    live = RecordedTurn("What's on my plate next week?", {"answer": "x"}, recorded_at="2026-10-28T06:34:49Z")
    no_call = RecordedTurn("Remember my bank PIN is 4821.", {})
    assert app.replay_is_recorded(ReplayBook([live, no_call]))
    assert not app.replay_is_recorded(ReplayBook([no_call]))
    assert not app.replay_is_recorded(ReplayBook([RecordedTurn("What's on my plate next week?", {"answer": "x"}),
                                                  no_call]))


# A replayed web search, independent of what data/replay holds (a live run without a Tavily key has none).
TAPER_WITH_SEARCH = RecordedTurn(
    "Find a 10-day taper plan for my half marathon.",
    {"route": '{"intent": "search", "query": "10-day taper plan half marathon"}',
     "answer": "Race day is Sunday 15 November [m3v1]."},
    [WebResult("How to taper for a half marathon", "https://example.org/running/half-marathon-taper",
               "Start the taper about ten days out.", 0.91)],
    recorded_at="2026-10-26T21:00:00Z",
)


def test_the_guard_tab_hash_also_matches_for_a_web_search_body() -> None:
    import dataclasses
    import hashlib
    app = _app()
    engine = app.Engine(dataclasses.replace(app.RUNTIME, replay=ReplayBook([TAPER_WITH_SEARCH])))
    try:
        view = engine.ask("Find a 10-day taper plan for my half marathon.")
    finally:
        engine.close()
    web = [call for call in view.outbound if call.provider == "tavily"]
    assert web, "the replayed turn has a web search"
    for call in web:
        assert hashlib.sha256(call.canonical_json.encode("utf-8")).hexdigest() == call.payload_sha256
    assert [c.payload_sha256 for c in view.outbound] == [r.payload_sha256 for r in view.receipts]


def test_an_availability_answer_shows_its_private_item_and_no_replay_tag() -> None:
    app = _app()
    engine = app.Engine(app.RUNTIME)
    try:
        view = engine.ask("Am I free Tuesday morning?")
    finally:
        engine.close()
    assert view.receipts == () and view.outbound == ()
    assert "⟦m7⟧" in view.draft and 7 in view.values, "the device-only text, filled in on the page"
    reply = app.render_reply(view)
    assert "Dermatologist appointment" in reply and ">replay<" not in reply
    assert "Checked availability on this device" in reply


def test_the_guard_tab_says_an_availability_check_was_made_on_the_device() -> None:
    app = _app()
    engine = app.Engine(app.RUNTIME)
    try:
        view = engine.ask("Is Wednesday afternoon free?")
    finally:
        engine.close()
    panel = app.render_guard_panel(view)
    assert "Answered on this device: no request was made" in panel and "Replayed turn" not in panel
    assert "Read on this device" in panel and "may go to" not in panel


def test_a_web_query_that_was_never_sent_is_not_shown_as_an_allowed_request() -> None:
    # Live with no Tavily key (28 Oct), the Guard tab listed "Allowed · Web search to Tavily web search" under
    # "Outbound requests" while nothing was searched; the receipts and the exact payloads were right, this was not.
    import dataclasses
    app = _app()
    no_search = RecordedTurn("Find a 10-day taper plan for my half marathon.",
                             {"answer": "Race day is Sunday 15 November [m3v1]."}, recorded_at="2026-10-28T06:53:46Z")
    engine = app.Engine(dataclasses.replace(app.RUNTIME, replay=ReplayBook([no_search])))
    try:
        view = engine.ask("Find a 10-day taper plan for my half marathon.")
    finally:
        engine.close()
    assert not [call for call in view.outbound if call.provider == "tavily"]
    html_text = app.render_guard_panel(view)
    items = html_text.split('<li class="cf-check">')[1:]
    web = [item for item in items if "Web search to" in item]
    assert web and all('cf-verdict">Not sent</span>' in item for item in web)
    assert "Requests checked by the guard" in html_text and "Outbound requests" not in html_text


def test_hand_written_web_results_are_not_called_a_live_recording() -> None:
    # A turn with no model outputs can still hold web results, and those are replayed: they need a recording too.
    app = _app()
    live = RecordedTurn("What's on my plate next week?", {"answer": "x"}, recorded_at="2026-10-28T06:34:49Z")
    hand_written = RecordedTurn("Find a 10-day taper plan for my half marathon.", {},
                                [WebResult("How to taper", "https://example.org/taper", "Start ten days out.", 0.9)])
    assert not app.replay_is_recorded(ReplayBook([live, hand_written]))


def test_a_turn_recorded_with_web_search_off_says_why_its_query_was_not_sent() -> None:
    import dataclasses
    app = _app()
    off = RecordedTurn("Find a 10-day taper plan for my half marathon.",
                       {"answer": "Race day is Sunday 15 November [m3v1]."}, recorded_at="2026-10-28T06:53:46Z",
                       web_search="off")
    engine = app.Engine(dataclasses.replace(app.RUNTIME, replay=ReplayBook([off])))
    try:
        view = engine.ask("Find a 10-day taper plan for my half marathon.")
    finally:
        engine.close()
    web = [item for item in app.render_guard_panel(view).split('<li class="cf-check">')[1:] if "Web search to" in item]
    assert web and "Not sent" in web[0] and "web search was off when this turn was recorded" in web[0]


# --- Round 6 (outside review, 28 Oct): the replay copy promised what replay cannot do -----------------------

TAPER = "Find a 10-day taper plan for my half marathon."


def test_the_header_names_web_search_as_it_was_recorded() -> None:
    """The published recording ran without Tavily, yet the header said "Web search: recorded results"."""
    app = _app()
    off = ReplayBook([RecordedTurn(TAPER, {"answer": "x"}, recorded_at="2026-10-28T07:30:15Z", web_search="off")])
    ran = ReplayBook([RecordedTurn(TAPER, {"answer": "x"}, [WebResult("T", "https://example.org/t", "C", 0.9)],
                                   recorded_at="2026-10-28T07:30:15Z", web_search="ran")])
    samples = ReplayBook([RecordedTurn(TAPER, {"answer": "x"})])
    assert app.replay_web_label(off) == "off when recorded"
    assert app.replay_web_label(ran) == "recorded results"
    assert app.replay_web_label(samples) == "none in the samples"
    assert app.RUNTIME.web_search == "off when recorded"  # data/replay: step 3 was recorded with web search off


def test_the_first_screen_in_replay_says_what_runs_without_a_model() -> None:
    from datetime import date
    app = _app()
    replay = app.render_empty([], date(2026, 10, 30), replay=True)
    live = app.render_empty([], date(2026, 10, 30), replay=False)
    assert "ask your own question" not in replay and "without a model" in replay
    assert "or ask your own question" in live
    for page in (replay, live):  # on the Space, "this device" was a Hugging Face server
        assert "never leave this device" not in page and "never leave Confidant" in page


def test_a_receipt_for_a_request_with_no_answer_says_so() -> None:
    app = _app()
    view = app.ReceiptView("2026-10-28T17:00:00Z", "nebius_token_factory", "nvidia/x", "answer", 0, 0, 0, (4,), (7,),
                           "ab" * 32, 0.0, "live", "no answer (HTTP 503)")
    html_text = app.render_call(view)
    assert "No answer (HTTP 503)" in html_text and "the request left this device" in html_text
    call = app.OutboundView("nebius_token_factory", "nvidia/x", "answer", "[]", "[]", "ab" * 32, "live",
                            "no answer (HTTP 503)")
    assert "Sent, no answer (HTTP 503)" in app.render_outbound([call], set())
