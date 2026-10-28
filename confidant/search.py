"""Tavily web search, and recording results as public, sourced memories.

Queries must pass ``Guard.check_outbound(query, "tavily", ...)`` before they get
here; the assistant does that for every search it runs.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from .memory import Memory, MemoryStore, SecretRefused
from .net import request_json

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
# Tavily pay-as-you-go: a basic search costs 1 credit at $0.008. Used for receipt estimates.
COST_PER_SEARCH_USD = 0.008
_SNIPPET_CHARS = 420


@dataclass(frozen=True)
class WebResult:
    """One search hit."""

    title: str
    url: str
    content: str
    score: float


class WebResults(list):  # noqa: FURB189 - a list of WebResult that also carries the unanswered attempts
    """Search hits, plus one reason per earlier attempt of the search that left and got no answer."""

    def __init__(self, results: Iterable[WebResult] = (), *, unanswered: tuple[str, ...] = ()) -> None:
        super().__init__(results)
        self.unanswered = unanswered


def search_payload(query: str, max_results: int) -> dict[str, object]:
    """The JSON body of a Tavily search (the API key travels in a header, not here): the one sent and the
    one whose SHA-256 the receipt keeps."""
    return {"query": query, "max_results": max_results, "search_depth": "basic"}


class Tavily:
    """Minimal client for Tavily's search endpoint (Bearer auth, basic depth)."""

    def __init__(
        self,
        api_key: str,
        http: httpx.Client | None = None,
        *,
        timeout: float = 20.0,
        retry_delay: float = 1.0,
    ) -> None:
        if not api_key:
            raise ValueError("Tavily needs an API key (TAVILY_API_KEY).")
        self._api_key = api_key
        self.timeout = timeout
        self.retry_delay = retry_delay
        self._http = http if http is not None else httpx.Client()
        self._owns_http = http is None

    def search(self, query: str, *, max_results: int = 5) -> WebResults:
        """Search the web and return results that have a URL, with the reasons of earlier attempts that left
        and got no answer (``unanswered``). Raises ServiceError on failure."""
        payload = search_payload(query, max_results)
        unanswered: list[str] = []
        data, _ = request_json(
            self._http, "POST", TAVILY_SEARCH_URL, service="Tavily search", api_key=self._api_key,
            key_env="TAVILY_API_KEY", timeout=self.timeout, retry_delay=self.retry_delay, payload=payload,
            unanswered=unanswered,
        )
        results = WebResults(unanswered=tuple(unanswered))
        for item in (data.get("results") if isinstance(data, dict) else None) or []:
            url = str(item.get("url") or "").strip()
            if url:
                results.append(WebResult(
                    title=" ".join(str(item.get("title") or url).split()),
                    url=url,
                    content=" ".join(str(item.get("content") or "").split()),
                    score=float(item.get("score") or 0.0),
                ))
        return results

    def close(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._owns_http:
            self._http.close()


def record_results(store: MemoryStore, results: Iterable[WebResult]) -> list[Memory]:
    """Store search results as public ``source`` memories (``source_type="web"``, ``source_ref=url``).

    A URL that is already a current memory is reused rather than duplicated, and a
    result whose text looks like a credential is skipped.
    """
    known = {memory.source_ref: memory for memory in store.all_current() if memory.source_type == "web"}
    recorded: list[Memory] = []
    for result in results:
        if result.url in known:
            recorded.append(known[result.url])
            continue
        snippet = result.content
        if len(snippet) > _SNIPPET_CHARS:
            snippet = snippet[:_SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
        try:
            memory = store.record(
                f"{result.title}: {snippet}" if snippet else result.title,
                kind="source", sensitivity="public", source_type="web", source_ref=result.url,
                author=urlparse(result.url).netloc.removeprefix("www.") or "web",
                label=result.title[:60],
            )
        except SecretRefused:
            continue
        known[result.url] = memory
        recorded.append(memory)
    return recorded
