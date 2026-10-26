"""Confidant's turn loop: route → act → retrieve → guard → answer → late-bind.

The cloud models (Nemotron on Nebius Token Factory) only see what the guard lets
through: shared memories verbatim and private ones as ``⟦m7⟧`` placeholders,
filled in on the device once the answer is back. Every outbound call goes
through one place, :class:`_Exchange`, which checks the payload with the guard,
calls the live client or reads the recorded replay, and writes a receipt.
Credentials are refused before anything else happens, and messages that touch a
private memory are routed by local rules instead of the cloud router.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

from .budget import Budget
from .guard import NEBIUS, TAVILY, Guard, GuardDecision, PreparedContext, placeholder
from .llm import ChatResult, ModelSelectionError, discover_models, estimate_cost, strip_reasoning
from .memory import KINDS, SENSITIVITIES, Memory, MemoryStore, SecretRefused, VersionConflict, utc_timestamp
from .net import ServiceError
from .receipts import Receipt, ReceiptLog
from .replay import RecordedTurn, ReplayBook
from .rules import (
    INTENTS,
    Route,
    classify_kind,
    classify_sensitivity,
    event_date,
    is_open_item,
    label_mentioned,
    make_label,
    memory_text_from,
    recurs_on,
    rule_route,
    stricter,
    time_window,
    web_query_from,
)
from .scan import detect_secret, find_secrets, find_values, words
from .search import COST_PER_SEARCH_USD, WebResult, record_results

REASON_MAX_TOKENS = 1024  # leaves room for a reasoning trace before the answer
FAST_MAX_TOKENS = 256
REASON_TEMPERATURE = 0.3
FAST_TEMPERATURE = 0.0
MAX_CONTEXT_MEMORIES = 10
BRIEF_HORIZON_DAYS = 7
BRIEF_QUESTION = "Run my daily brief."
WEB_RESULT_COUNT = 5
TAVILY_MODEL = "tavily-search-basic"

ANSWER_SYSTEM_PROMPT = """\
You are Confidant, a personal assistant that answers from the user's own memories.
Rules:
1. Use the memories provided. Cite every memory you rely on with its tag exactly as written, for example [m12v2].
2. Some memories are withheld for privacy and appear as placeholders such as ⟦m7⟧ with a short label. \
Copy the placeholder verbatim where that information belongs; it is filled in on the user's device after you answer. \
A placeholder stands for a whole sentence, so put it on its own line or after a colon.
3. Never guess, reconstruct or ask for a withheld value, and add nothing that its label does not say.
4. If the memories hold nothing relevant, say so clearly ("Nothing in memory about that.") and offer only \
general knowledge, labelled as such.
5. Keep answers short: a few sentences or a brief list."""

ROUTER_SYSTEM_PROMPT = """\
You route messages for Confidant, a personal memory assistant. Reply with exactly one JSON object and no other text:
{"intent": "answer" | "remember" | "correct" | "search" | "brief", "text": "...", "target": "...", "query": "..."}
- remember: the user tells you something new to keep. "text": that fact as one short standalone sentence.
- correct: the user changes or fixes something already known. "target": a few words naming that memory; \
"text": the full corrected fact as one sentence.
- search: the user wants information from the web. "query": a short web query with no names, dates or personal details.
- brief: the user asks for their daily brief.
- answer: anything else (questions about their life, drafting, advice).
Leave out fields that do not apply. /no_think"""

CLASSIFY_SYSTEM_PROMPT = """\
You label memories for a privacy guard. Reply with exactly one JSON object and no other text:
{"sensitivity": "public" | "personal" | "private", "kind": "fact" | "preference" | "event" | "skill", "label": "..."}
- public: harmless in a web search (work tools, public interests).
- personal: everyday life, family, friends, plans and tastes; fine for the assistant's cloud model, \
never for web search.
- private: health, money, legal matters, job search, intimate details, or anything the user would not want \
a cloud model to read.
- label: 2 to 4 lowercase words naming the topic, with no name, number or date (e.g. "dentist appointment").
/no_think"""

_CITATION = re.compile(r"⟦\s*m(\d+)(?:v\d+)?\s*⟧|\bm(\d+)v\d+\b")


class ChatClient(Protocol):
    """What the assistant needs from a chat model client (TokenFactory or a test double)."""

    def chat(
        self, model: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int, temperature: float, purpose: str
    ) -> ChatResult:
        """Run one chat completion."""
        ...

    def list_models(self) -> list[str]:
        """Model ids available to the caller."""
        ...


class SearchClient(Protocol):
    """What the assistant needs from a web search client (Tavily or a test double)."""

    def search(self, query: str, *, max_results: int = 5) -> list[WebResult]:
        """Search the web."""
        ...


@dataclass(frozen=True)
class GuardReport:
    """What the guard did during one turn.

    ``prepared`` is the context composed for the reasoning model (None if no
    answer was composed); ``decisions`` every outbound check, in order;
    ``kept_local`` the memories withheld from every call of the turn;
    ``late_bound_ids`` the placeholders filled in locally; ``unresolved``
    placeholders the model invented, left visible.
    """

    prepared: PreparedContext | None = None
    decisions: list[GuardDecision] = field(default_factory=list)
    kept_local: list[Memory] = field(default_factory=list)
    late_bound_ids: list[int] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    sent_to_cloud: bool = False

    def summary(self) -> str:
        """One line for under the answer, e.g. "2 private memories never left Confidant"."""
        total = len(self.kept_local)
        if not total:
            return "No private memory was involved" if self.sent_to_cloud else "Nothing was sent to the cloud"
        counts = [(level, sum(m.sensitivity == level for m in self.kept_local)) for level in reversed(SENSITIVITIES)]
        parts = [f"{count} {level}" for level, count in counts if count]
        noun = "memory" if total == 1 else "memories"
        if len(parts) == 1:
            return f"{parts[0]} {noun} never left Confidant"
        return f"{total} memories never left Confidant ({', '.join(parts)})"


@dataclass(frozen=True)
class Turn:
    """The result of one message.

    ``answer`` is the late-bound text for the user and ``draft`` the same text
    with placeholders still in it (safe to pass on); ``citations`` the memories
    the answer cites or fills in, in order of appearance; ``actions`` what
    changed (e.g. "recorded m26v1", "corrected m7 → v2"); ``receipts`` the rows
    logged during the turn; ``mode`` is "live" or "replay"; ``notice`` tells the
    user about replay, budget limits or failed calls.
    """

    answer: str
    citations: list[Memory]
    guard: GuardReport
    actions: list[str]
    receipts: list[Receipt]
    mode: str
    intent: str = "answer"
    draft: str = ""
    notice: str = ""


@dataclass(frozen=True)
class _Outcome:
    """What an intent handler produced: the draft answer and the context it was written from."""

    draft: str
    prepared: PreparedContext


def extract_json(text: str) -> dict[str, Any] | None:
    """The first JSON object in model output, ignoring reasoning traces, code fences and prose."""
    cleaned = re.sub(r"```(?:json)?", "", strip_reasoning(text))
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _text_field(data: Mapping[str, Any], name: str) -> str:
    """A string field of model JSON, trimmed; empty if absent or not a string."""
    value = data.get(name)
    return " ".join(value.split())[:500] if isinstance(value, str) else ""


def parse_route(text: str) -> Route | None:
    """The router's decision from its raw output, or None if it is not valid JSON with a known intent."""
    data = extract_json(text)
    intent = str(data.get("intent", "")).strip().lower() if data else ""
    if data is None or intent not in INTENTS:
        return None
    return Route(
        intent, text=_text_field(data, "text"), target=_text_field(data, "target"),
        query=_text_field(data, "query"), source="model",
    )


def parse_classification(text: str) -> tuple[str, str, str] | None:
    """``(sensitivity, kind, label)`` from the classifier's raw output, or None.

    Kind and label come back empty when invalid; a label that contains a value
    (name, number, date) or a credential is rejected, since labels are shown to
    cloud models.
    """
    data = extract_json(text)
    sensitivity = str(data.get("sensitivity", "")).strip().lower() if data else ""
    if data is None or sensitivity not in SENSITIVITIES:
        return None
    kind = str(data.get("kind", "")).strip().lower()
    label = _text_field(data, "label").lower()
    safe_label = 0 < len(label) <= 60 and not find_values(label) and not find_secrets(label)
    valid_kind = kind in KINDS and kind != "source"
    return sensitivity, kind if valid_kind else "", label if safe_label else ""


def cited_memories(text: str, prepared: PreparedContext) -> list[Memory]:
    """Memories of ``prepared`` that ``text`` cites ([m12v2]) or fills in (⟦m7⟧), in order of appearance."""
    known = {memory.id: memory for memory in [*prepared.shared, *prepared.withheld]}
    cited: dict[int, Memory] = {}
    for match in _CITATION.finditer(text):
        memory_id = int(match.group(1) or match.group(2))
        if memory_id in known:
            cited.setdefault(memory_id, known[memory_id])
    return list(cited.values())


def _long_date(day: date) -> str:
    """A date written out, e.g. "Friday 30 October 2026"."""
    return f"{day:%A} {day.day} {day:%B %Y}"


class _Exchange:
    """Every outbound call of one turn, live or replayed, and what the turn will report."""

    def __init__(
        self, assistant: Assistant, session_id: str, mode: str, recorded: RecordedTurn | None, notices: list[str]
    ) -> None:
        self.assistant = assistant
        self.session_id = session_id
        self.mode = mode
        self.recorded = recorded
        self.notices = notices
        self.actions: list[str] = []
        self.decisions: list[GuardDecision] = []
        self.receipts: list[Receipt] = []
        self.sent_ids: set[int] = set()
        self.withheld_ids: set[int] = set()
        self.live_calls = 0
        self.tokens = 0
        self.outputs: dict[str, str] = {}
        self.web_results: list[WebResult] = []

    def complete(
        self,
        purpose: str,
        role: str,
        system: str,
        user: str,
        *,
        prepared: PreparedContext | None = None,
        max_tokens: int = FAST_MAX_TOKENS,
        temperature: float = FAST_TEMPERATURE,
    ) -> str | None:
        """Ask the ``role`` model (live) or read its recorded output (replay).

        The user message passes the guard first and goes out redacted. Returns
        None when it is blocked or no output is available, so callers fall back.
        """
        assistant = self.assistant
        decision = assistant.guard.check_outbound(user, NEBIUS, assistant.store.all_current(), purpose=purpose)
        self.decisions.append(decision)
        sent = prepared.shared_ids if prepared else []
        withheld = sorted(set((prepared.withheld_ids if prepared else []) + decision.withheld_ids))
        self.withheld_ids.update(withheld)
        if not decision.allowed:
            self.notices.append(f"The guard blocked the {purpose} request, so nothing was sent.")
            return None
        self.sent_ids.update(sent)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": decision.redacted_text}]
        if self.mode == "replay":
            if self.recorded is None or purpose not in self.recorded.outputs:
                return None
            self._log(NEBIUS, self.recorded.models.get(role, "replay"), purpose, messages, sent, withheld)
            return strip_reasoning(self.recorded.outputs[purpose])
        assert assistant.llm is not None and assistant.models is not None  # guaranteed in live mode
        try:
            result = assistant.llm.chat(
                assistant.models[role], messages, max_tokens=max_tokens, temperature=temperature, purpose=purpose
            )
        except ServiceError as exc:
            self.notices.append(f"The live {purpose} call failed ({exc}); a local fallback was used.")
            return None
        self.live_calls += 1
        self.tokens += result.prompt_tokens + result.completion_tokens
        self._log(
            NEBIUS, result.model, purpose, messages, sent, withheld, prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens, latency_ms=result.latency_ms,
            cost=estimate_cost(result.model, result.prompt_tokens, result.completion_tokens),
        )
        if not result.text:
            hint = " (it spent its token budget reasoning)" if result.finish_reason == "length" else ""
            self.notices.append(f"The {purpose} model returned no answer text{hint}; a local fallback was used.")
            return None
        self.outputs[purpose] = result.text
        return result.text

    def web(self, query: str) -> list[WebResult]:
        """Run a guarded web search (live) or return the recorded results (replay)."""
        assistant = self.assistant
        decision = assistant.guard.check_outbound(query, TAVILY, assistant.store.all_current(), purpose="web search")
        self.decisions.append(decision)
        self.withheld_ids.update(decision.withheld_ids)
        if not decision.allowed:
            self.notices.append("The guard blocked the web query: it would have revealed private details.")
            return []
        payload = {"query": decision.redacted_text, "max_results": WEB_RESULT_COUNT, "search_depth": "basic"}
        if self.mode == "replay":
            results = list(self.recorded.web_results) if self.recorded else []
            if results:
                self._log(TAVILY, TAVILY_MODEL, "web_search", payload, [], decision.withheld_ids)
            else:
                self.notices.append("Replay has no recorded web results for this request.")
            return results
        if assistant.search is None:
            self.notices.append("Web search is off (no TAVILY_API_KEY), so the answer uses memory only.")
            return []
        started = time.perf_counter()
        try:
            results = assistant.search.search(decision.redacted_text, max_results=WEB_RESULT_COUNT)
        except ServiceError as exc:
            self.notices.append(f"Web search failed ({exc}); the answer uses memory only.")
            return []
        self.live_calls += 1
        self._log(
            TAVILY, TAVILY_MODEL, "web_search", payload, [], decision.withheld_ids,
            latency_ms=round((time.perf_counter() - started) * 1000), cost=COST_PER_SEARCH_USD,
        )
        self.web_results = results
        return results

    def _log(
        self,
        provider: str,
        model: str,
        purpose: str,
        payload: object,
        sent: Sequence[int],
        withheld: Sequence[int],
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Write one receipt for this turn's mode."""
        self.receipts.append(self.assistant.receipts.record(
            provider=provider, model=model, purpose=purpose, payload=payload, mode=self.mode,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, latency_ms=latency_ms,
            sent_memory_ids=sent, withheld_memory_ids=withheld, cost_usd_estimate=cost,
            session_id=self.session_id,
        ))


class Assistant:
    """Confidant: answers from memory without letting private values leave the device.

    ``llm`` is a TokenFactory (None without a key) and ``search`` a Tavily client
    (optional). ``replay`` serves recorded outputs when live calls are not
    possible; ``recorder`` captures live outputs into a ReplayBook. ``models``
    fixes the model ids (else they are discovered), and ``today`` the date used
    for "next week" (else the seed snapshot date, else today in UTC).
    """

    def __init__(
        self,
        store: MemoryStore,
        guard: Guard,
        receipts: ReceiptLog,
        budget: Budget,
        llm: ChatClient | None,
        search: SearchClient | None = None,
        replay: ReplayBook | None = None,
        *,
        models: Mapping[str, str] | None = None,
        today: Callable[[], date] | None = None,
        recorder: ReplayBook | None = None,
    ) -> None:
        self.store = store
        self.guard = guard
        self.receipts = receipts
        self.budget = budget
        self.llm = llm
        self.search = search
        self.replay = replay
        self.recorder = recorder
        self.models: dict[str, str] | None = dict(models) if models else None
        self._today = today

    # --- public API ----------------------------------------------------------

    def today(self) -> date:
        """The assistant's notion of today (see the class docstring)."""
        if self._today is not None:
            return self._today()
        return self.store.as_of() or datetime.now(timezone.utc).date()

    def resolve_models(self) -> dict[str, str]:
        """The ``{"reason": id, "fast": id}`` models in use, discovered on first call.

        Raises RuntimeError without a live client, and ModelSelectionError or
        ServiceError when discovery fails.
        """
        if self.models is None:
            if self.llm is None:
                raise RuntimeError("No live model client is configured (NEBIUS_API_KEY is not set).")
            self.models = discover_models(self.llm)
        return self.models

    def handle(self, message: str, *, session_id: str, intent: str | None = None) -> Turn:
        """Answer one user message end to end.

        ``intent`` (one of INTENTS) skips routing, e.g. "answer" for read-only callers.
        Raises ValueError for an empty message or an unknown intent.
        """
        message = " ".join(str(message).split())
        if not message:
            raise ValueError("message must not be empty")
        if intent is not None and intent not in INTENTS:
            raise ValueError(f"intent must be one of {', '.join(INTENTS)}; got {intent!r}")
        secret = detect_secret(message)
        if secret:
            return self._refuse(message, secret, session_id)
        exchange = self._open(session_id, message)
        if intent is None:
            route, private = self._route(exchange, message)
        else:
            route, private = Route(intent, source="caller"), False
        handlers = {
            "answer": self._answer, "remember": self._remember, "correct": self._correct,
            "search": self._search, "brief": self._brief,
        }
        outcome = handlers[route.intent](exchange, message, route, private)
        return self._close(exchange, message, route, outcome)

    def daily_brief(self, session_id: str) -> Turn:
        """Upcoming events and open items from memory, written in one reasoning call."""
        return self.handle(BRIEF_QUESTION, session_id=session_id, intent="brief")

    # --- turn lifecycle --------------------------------------------------------

    def _open(self, session_id: str, message: str) -> _Exchange:
        """Pick live or replay mode for this turn and say why when it is replay."""
        notices: list[str] = []
        mode = "replay"
        if self.llm is None:
            notices.append(
                "No NEBIUS_API_KEY is set, so answers come from recorded replays."
                if self.replay else
                "No NEBIUS_API_KEY is set and there is no replay data, so answers are built from local memory only."
            )
        else:
            allowed, reason = self.budget.check(session_id)
            if not allowed:
                notices.append(reason)
            else:
                try:
                    self.resolve_models()
                    mode = "live"
                except (ModelSelectionError, ServiceError) as exc:
                    notices.append(f"No usable Nemotron model ({exc}), so Confidant switched to replay.")
        recorded = self.replay.lookup(message) if mode == "replay" and self.replay else None
        return _Exchange(self, session_id, mode, recorded, notices)

    def _close(self, exchange: _Exchange, message: str, route: Route, outcome: _Outcome) -> Turn:
        """Late-bind the draft, settle the budget and recorder, and build the Turn."""
        prepared = outcome.prepared
        segments = self.guard.segments(outcome.draft, prepared)
        unresolved = [segment.text for segment in segments if segment.placeholder and segment.memory is None]
        if unresolved:
            exchange.notices.append(
                f"The answer mentions unknown placeholder(s) {', '.join(unresolved)}; they are shown as written."
            )
        late_bound = list(dict.fromkeys(segment.memory.id for segment in segments if segment.memory is not None))
        kept_local = [self.store.get(memory_id) for memory_id in sorted(exchange.withheld_ids - exchange.sent_ids)]
        live = exchange.mode == "live" and exchange.live_calls > 0
        if live:
            self.budget.consume(exchange.session_id, exchange.tokens)
        if live and self.recorder is not None and exchange.outputs:
            self.recorder.add(RecordedTurn(
                question=message, outputs=dict(exchange.outputs), web_results=list(exchange.web_results),
                models=dict(self.models or {}), recorded_at=utc_timestamp(),
            ))
        report = GuardReport(
            prepared=prepared, decisions=exchange.decisions, kept_local=kept_local,
            late_bound_ids=late_bound, unresolved=unresolved, sent_to_cloud=live,
        )
        return Turn(
            answer=self.guard.late_bind(outcome.draft, prepared), citations=cited_memories(outcome.draft, prepared),
            guard=report, actions=exchange.actions, receipts=exchange.receipts, mode=exchange.mode,
            intent=route.intent, draft=outcome.draft, notice=" ".join(exchange.notices),
        )

    def _refuse(self, message: str, reason: str, session_id: str) -> Turn:
        """The turn for a message holding a credential: nothing is stored or sent."""
        masked = self.guard.check_outbound(message, NEBIUS).redacted_text
        decision = GuardDecision(False, masked, [f"looks like {reason}"], [], NEBIUS, "message")
        text = (
            f"I won't keep or send that: it looks like {reason}. Confidant never stores credentials; "
            "a password manager is the right place for them. Nothing was saved or sent."
        )
        mode = "live" if self.llm is not None and self.budget.check(session_id)[0] else "replay"
        return Turn(
            answer=text, citations=[], guard=GuardReport(decisions=[decision]),
            actions=[f"refused {reason}: nothing stored, nothing sent"], receipts=[], mode=mode,
            intent=rule_route(message).intent, draft=text,
        )

    def _route(self, exchange: _Exchange, message: str) -> tuple[Route, bool]:
        """Route with the fast model, or locally when the message touches private data.

        Returns the route and whether the message was judged private.
        """
        reasons, ids = self._private_signals(message)
        if reasons:
            exchange.decisions.append(GuardDecision(
                False, "", [*reasons, "routed on this device; the message was not sent to the router"],
                ids, NEBIUS, "route",
            ))
            exchange.withheld_ids.update(ids)
            return rule_route(message), True
        raw = exchange.complete("route", "fast", ROUTER_SYSTEM_PROMPT, message)
        route = parse_route(raw) if raw is not None else None
        if route is None:
            if raw is not None:
                exchange.actions.append("router reply was not valid JSON; used keyword rules")
            route = rule_route(message)
        return route, False

    def _private_signals(self, message: str) -> tuple[list[str], list[int]]:
        """Why a message must stay local: private values in it, private labels it
        names, or a private topic according to the local rules."""
        current = self.store.all_current()
        decision = self.guard.check_outbound(message, NEBIUS, current, purpose="route")
        reasons, ids = list(decision.reasons), list(decision.withheld_ids)
        for memory in current:
            named = label_mentioned(memory.label, message)
            if named and memory.id not in ids and not self.guard.may_send(memory, NEBIUS):
                reasons.append(f"mentions {memory.cite()} ({memory.sensitivity}: {memory.label})")
                ids.append(memory.id)
        if not reasons and classify_sensitivity(message) == "private":
            reasons.append("looks private to the local rules (health, money, legal or job search)")
        return reasons, ids

    # --- intents ---------------------------------------------------------------

    def _answer(self, exchange: _Exchange, message: str, route: Route, private: bool) -> _Outcome:
        """Answer from memory with the reasoning model."""
        window = time_window(message, self.today())
        notes = []
        if window:
            first, last = window
            notes.append(
                f"Period asked about: {_long_date(first)}"
                + ("" if first == last else f" to {_long_date(last)}")
                + "; every event of that period is listed below."
            )
        return self._compose(exchange, "answer", message, self._retrieve(message, window), notes)

    def _remember(self, exchange: _Exchange, message: str, route: Route, private: bool) -> _Outcome:
        """Store what the user stated, classified locally first (and by the fast model if not private)."""
        text = route.text or memory_text_from(message)
        if not text:
            return _Outcome("Tell me what to remember, for example: Remember that I prefer aisle seats.",
                            self.guard.prepare_context([], NEBIUS))
        sensitivity, kind, label = self._classify(exchange, text, private)
        try:
            memory = self.store.record(
                text, kind=kind, sensitivity=sensitivity, source_type="user",
                source_ref=exchange.session_id, author="user", label=label,
            )
        except SecretRefused as exc:
            exchange.actions.append(f"refused {exc.reason}: nothing stored")
            return _Outcome(str(exc), self.guard.prepare_context([], NEBIUS))
        exchange.actions.append(f"recorded {memory.cite()}")
        lead = f"Saved as [{memory.cite()}] ({memory.sensitivity} {memory.kind}, “{memory.label}”): "
        return self._confirm(memory, lead)

    def _correct(self, exchange: _Exchange, message: str, route: Route, private: bool) -> _Outcome:
        """Add a new version of the memory the message corrects (or record it if none matches)."""
        target = self._find_target(route.target, message)
        if target is None:
            exchange.actions.append("no existing memory matched the correction; recorded it as new")
            return self._remember(exchange, message, route, private)
        try:
            updated = self.store.correct(
                target.id, route.text or memory_text_from(message), author="user",
                source_ref=exchange.session_id, expected_version=target.version,
            )
        except VersionConflict as exc:
            exchange.actions.append(f"correction of m{target.id} not applied: version conflict")
            return _Outcome(
                f"[{target.cite()}] changed while I was updating it (it is now at version {exc.actual}); "
                "please try again.",
                self.guard.prepare_context([target], NEBIUS),
            )
        except SecretRefused as exc:
            exchange.actions.append(f"refused {exc.reason}: nothing stored")
            return _Outcome(str(exc), self.guard.prepare_context([], NEBIUS))
        exchange.actions.append(f"corrected m{updated.id} → v{updated.version}")
        return self._confirm(updated, f"Updated [{updated.cite()}] (“{updated.label}”, was v{target.version}): ")

    def _search(self, exchange: _Exchange, message: str, route: Route, private: bool) -> _Outcome:
        """Search the web with a guarded query, record the results, then answer from them."""
        results = exchange.web(route.query or web_query_from(message))
        found = record_results(self.store, results)
        if found:
            exchange.actions.append("recorded web sources " + ", ".join(f"{m.cite()} ({m.author})" for m in found))
        found_ids = {memory.id for memory in found}
        related = [m for m in self._retrieve(message, time_window(message, self.today())) if m.id not in found_ids]
        return self._compose(exchange, "answer", message, (found + related)[:MAX_CONTEXT_MEMORIES])

    def _brief(self, exchange: _Exchange, message: str, route: Route, private: bool) -> _Outcome:
        """Events of the coming week, open to-dos and today's routines, plus today's receipts."""
        if exchange.mode == "replay" and exchange.recorded is None and self.replay:
            exchange.recorded = self.replay.lookup(BRIEF_QUESTION)
        today = self.today()
        events = self._events_between(today, today + timedelta(days=BRIEF_HORIZON_DAYS))
        event_ids = {memory.id for memory in events}
        extras = [
            memory for memory in self.store.all_current()
            if memory.id not in event_ids and (is_open_item(memory) or recurs_on(memory, today))
        ]
        request = (
            f"Write my daily brief for {_long_date(today)}: the events of the next {BRIEF_HORIZON_DAYS} days, "
            "soonest first; my open to-do items; and one practical suggestion for today."
        )
        outcome = self._compose(exchange, "brief", request, events + extras)
        summary = self.receipts.today_summary(exchange.session_id).line()
        return _Outcome(f"{outcome.draft}\n\n{summary}", outcome.prepared)

    # --- helpers ---------------------------------------------------------------

    def _classify(self, exchange: _Exchange, text: str, private: bool) -> tuple[str, str, str]:
        """``(sensitivity, kind, label)``: local rules, refined by the fast model unless
        the text is private (then nothing is sent). The model can only make it stricter."""
        sensitivity, kind, label = classify_sensitivity(text), classify_kind(text), make_label(text)
        if private or sensitivity == "private":
            return "private", kind, label
        raw = exchange.complete("classify", "fast", CLASSIFY_SYSTEM_PROMPT, text)
        parsed = parse_classification(raw) if raw is not None else None
        if parsed is None:
            return sensitivity, kind, label
        model_sensitivity, model_kind, model_label = parsed
        return stricter(model_sensitivity, sensitivity), model_kind or kind, model_label or label

    def _confirm(self, memory: Memory, lead: str) -> _Outcome:
        """A local confirmation draft; private values appear as their placeholder."""
        prepared = self.guard.prepare_context([memory], NEBIUS)
        if prepared.withheld:
            return _Outcome(f"{lead}{placeholder(memory.id)} Its value stays local; "
                            "cloud models only ever see its label.", prepared)
        return _Outcome(f"{lead}{memory.text}", prepared)

    def _find_target(self, hint: str, message: str) -> Memory | None:
        """The memory a correction refers to: one whose label the hint or message names
        (most specific label first), else the best full-text match."""
        candidates = [memory for memory in self.store.all_current() if memory.source_type != "web"]
        for text in (hint, message):
            named = [memory for memory in candidates if text and label_mentioned(memory.label, text)]
            if named:
                return max(named, key=lambda memory: (len(words(memory.label)), -memory.id))
        hits = [memory for memory in self.store.search(hint or message, limit=3) if memory.source_type != "web"]
        return hits[0] if hits else None

    def _events_between(self, first: date, last: date) -> list[Memory]:
        """Current event memories dated within [first, last], soonest first."""
        today = self.today()
        dated = [(event_date(memory, today), memory) for memory in self.store.all_current() if memory.kind == "event"]
        within = [(day, memory) for day, memory in dated if day is not None and first <= day <= last]
        return [memory for _, memory in sorted(within, key=lambda pair: (pair[0], pair[1].id))]

    def _retrieve(self, message: str, window: tuple[date, date] | None) -> list[Memory]:
        """Events in the asked period (if any), then the best full-text matches."""
        found = {memory.id: memory for memory in (self._events_between(*window) if window else [])}
        for memory in self.store.search(message, limit=8):
            found.setdefault(memory.id, memory)
        return list(found.values())[:MAX_CONTEXT_MEMORIES]

    def _compose(
        self, exchange: _Exchange, purpose: str, request: str, memories: list[Memory], notes: Sequence[str] = ()
    ) -> _Outcome:
        """Prepare the guarded context, ask the reasoning model, fall back to a local list."""
        prepared = self.guard.prepare_context(memories, NEBIUS)
        prompt = "\n".join([
            f"Today is {_long_date(self.today())}.", *notes, "",
            "Memories:", prepared.context_text or "(none relevant)", "", f"Request: {request}",
        ])
        raw = exchange.complete(
            purpose, "reason", ANSWER_SYSTEM_PROMPT, prompt, prepared=prepared,
            max_tokens=REASON_MAX_TOKENS, temperature=REASON_TEMPERATURE,
        )
        return _Outcome(raw if raw is not None else self._fallback(exchange, prepared, memories), prepared)

    @staticmethod
    def _fallback(exchange: _Exchange, prepared: PreparedContext, memories: list[Memory]) -> str:
        """A local answer listing the relevant memories (private ones as placeholders)."""
        lead = (
            "Replay has no recorded answer for this request"
            if exchange.mode == "replay" else "The cloud model could not answer this time"
        )
        included = {memory.id for memory in [*prepared.shared, *prepared.withheld]}
        withheld = set(prepared.withheld_ids)
        lines: list[str] = []
        for memory in memories:
            if memory.id in included:
                included.discard(memory.id)
                shown = placeholder(memory.id) if memory.id in withheld else f"{memory.text} [{memory.cite()}]"
                lines.append(f"- {shown}")
        if not lines:
            return f"{lead}, and nothing in memory matches it."
        return f"{lead}, so here are the most relevant memories:\n" + "\n".join(lines)
