"""The Guard tab shows each SHA-256 whole.

Found on 29 October 2026 by a zero-context reader of the Devpost gallery: in the Prove it still, the hash showed 63 of
its 64 characters. The panel's edge cut the last one, since a hash has no place to wrap. A reader who compares it
with a receipt sees two different hashes. Each hash now wraps in its middle when the panel is too narrow, and the
text read or copied from the page is still the whole hash.
"""
from __future__ import annotations

import importlib
import re


def visible(html_text: str) -> str:
    return re.sub(r"<[^>]+>", "", html_text)


def test_prove_it_shows_both_hashes_whole_and_breakable_in_the_middle() -> None:
    app = importlib.import_module("app")
    engine = app.Engine(app.RUNTIME)
    try:
        view = engine.ask("What's on my plate next week?")
        proof = app.prove(engine.store, view.question, intent=view.intent, guard=engine.guard)
    finally:
        engine.close()
    panel = app.render_proof(view, proof)
    hashes = [sha for _, sha in proof.original + proof.moved]
    assert hashes and all(len(sha) == 64 for sha in hashes)
    for sha in hashes:
        assert sha in visible(panel)
        assert f"{sha[:32]}<wbr>{sha[32:]}" in panel


def test_the_exact_requests_show_their_hash_whole_and_breakable_in_the_middle() -> None:
    app = importlib.import_module("app")
    engine = app.Engine(app.RUNTIME)
    try:
        view = engine.ask("What's on my plate next week?")
    finally:
        engine.close()
    panel = app.render_guard_panel(view)
    for call in view.outbound:
        assert call.payload_sha256 in visible(panel)
        assert f"{call.payload_sha256[:32]}<wbr>{call.payload_sha256[32:]}" in panel
