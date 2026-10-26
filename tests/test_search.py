"""Tavily over a mocked transport, and recording results as web memories."""
from __future__ import annotations

import json

import httpx
import pytest

from confidant.memory import MemoryStore
from confidant.net import ServiceError
from confidant.search import TAVILY_SEARCH_URL, Tavily, WebResult, record_results


def tavily(handler) -> Tavily:
    """A Tavily client whose HTTP calls go to ``handler``."""
    return Tavily("test-key", http=httpx.Client(transport=httpx.MockTransport(handler)), retry_delay=0)


def test_search_posts_the_documented_request_and_parses_results() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"query": "q", "results": [
            {"title": "Taper  guide", "url": "https://example.org/taper", "content": "Cut\nvolume.", "score": 0.9},
            {"title": "No link", "url": "", "content": "skipped"},
        ]})

    results = tavily(handler).search("half marathon taper", max_results=3)
    request = seen[0]
    assert str(request.url) == TAVILY_SEARCH_URL and request.method == "POST"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {"query": "half marathon taper", "max_results": 3, "search_depth": "basic"}
    assert results == [WebResult("Taper guide", "https://example.org/taper", "Cut volume.", 0.9)]


def test_search_errors_name_the_key_variable() -> None:
    with pytest.raises(ServiceError, match="TAVILY_API_KEY"):
        tavily(lambda request: httpx.Response(401, json={"detail": "nope"})).search("q")
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        Tavily("")


def test_results_become_public_web_memories_once_per_url(store: MemoryStore) -> None:
    results = [
        WebResult("Taper guide", "https://www.example.org/taper", "Cut volume " * 80, 0.9),
        WebResult("Leaked page", "https://bad.example.net/", "the admin password is hunter22", 0.5),
    ]
    first = record_results(store, results)
    assert len(first) == 1
    memory = first[0]
    assert (memory.kind, memory.sensitivity, memory.source_type) == ("source", "public", "web")
    assert memory.source_ref == "https://www.example.org/taper"
    assert (memory.author, memory.label) == ("example.org", "Taper guide")
    assert memory.text.startswith("Taper guide: Cut volume") and memory.text.endswith("…") and len(memory.text) < 460
    assert record_results(store, results[:1]) == first
    assert len([m for m in store.all_current() if m.source_type == "web"]) == 1
