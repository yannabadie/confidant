"""Confidant web demo: the Gradio Blocks app behind the Hugging Face Space.

The screen has three parts: the demo script (left), the conversation (centre)
and an inspector (right) with Memory, Guard, Receipts and Daily brief tabs.
Every browser tab works on its own in-memory copy of the seed memory, so
visitors never see each other's edits.

Mode is decided at start-up: live when NEBIUS_API_KEY is set (web search also
needs TAVILY_API_KEY), replay otherwise. In replay the recorded model outputs
in data/replay/ stand in for Nemotron; memory, guard, late binding and receipts
still run here, and every replayed turn is labelled "replay".

Run locally with ``python app.py``.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import gradio as gr

# =============================================================================
# Core adapter
# The only section that knows the confidant package's names and shapes. The
# rest of the file works with the small view classes defined here, so a rename
# in the core is a one-line fix in this section.
# =============================================================================
from confidant.assistant import Assistant
from confidant.budget import Budget
from confidant.config import DEMO_PERSONA, REPLAY_DIR, env
from confidant.forms import warm_up
from confidant.guard import DEFAULT_POLICY, NEBIUS, PLACEHOLDER_RE, Guard, drop_echoed_labels
from confidant.llm import ModelSelectionError, TokenFactory, discover_models
from confidant.memory import MemoryStore
from confidant.net import ServiceError
from confidant.proof import Proof, prove
from confidant.receipts import ReceiptLog
from confidant.replay import ReplayBook, normalize_question
from confidant.search import Tavily

PERSONA: dict[str, str] = json.loads(DEMO_PERSONA.read_text(encoding="utf-8")).get("persona", {})
PROVIDER_POLICY: dict[str, list[str]] = DEFAULT_POLICY["providers"]
# The header badge already says why turns are replayed, so this notice is not repeated per answer.
REPLAY_NOTICE = "No NEBIUS_API_KEY is set, so answers come from recorded replays."
# The brief ends with a one-line receipts summary; the UI draws it as a receipt slip instead.
BRIEF_RECEIPTS_LINE = re.compile(r"\s*Receipts today:[^\n]*\Z")


@dataclass(frozen=True)
class MemoryView:
    """One version of one memory, in the shape the UI renders."""

    id: int
    version: int
    text: str
    label: str
    kind: str
    sensitivity: str
    status: str
    source_type: str
    source_ref: str
    author: str
    created_at: str
    note: str

    @property
    def cite(self) -> str:
        """Citation id such as ``m7v2``."""
        return f"m{self.id}v{self.version}"


@dataclass(frozen=True)
class CheckView:
    """One outbound check by the guard: what a request carried and whether it could go."""

    purpose: str
    provider: str
    allowed: bool
    reasons: tuple[str, ...]
    text: str  # exactly what was (or, in replay, would be) sent; empty when blocked
    not_sent: str = ""  # why an allowed request never left (web search off, or no recorded search)


@dataclass(frozen=True)
class ReceiptView:
    """One outbound call as logged by the receipt log (a payload hash, never the payload)."""

    at: str
    provider: str
    model: str
    purpose: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    sent_ids: tuple[int, ...]
    withheld_ids: tuple[int, ...]
    payload_sha256: str
    cost_usd: float
    mode: str
    status: str = "answered"  # or why a request that left got no answer: "no answer (HTTP 503)"


@dataclass(frozen=True)
class OutboundView:
    """One call exactly as it left the device: the JSON payload and the SHA-256 its receipt keeps."""

    provider: str
    model: str
    purpose: str
    payload_json: str  # pretty-printed for reading; the hash is over the compact form
    canonical_json: str  # compact, sorted keys: hash this to get payload_sha256
    payload_sha256: str
    mode: str
    status: str = "answered"


@dataclass(frozen=True)
class SummaryView:
    """Today's receipt totals for this session."""

    calls: int
    replay_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    kept_local: int


@dataclass(frozen=True)
class TurnView:
    """Everything the UI shows about one answered message (or the daily brief)."""

    question: str
    draft: str = ""  # the answer as shown on this device, placeholders included
    written: str = ""  # the draft as the cloud model wrote it, when it differs (this device put it in date order)
    values: Mapping[int, str] = field(default_factory=dict)  # placeholder id -> value filled in here
    citations: tuple[MemoryView, ...] = ()
    shared: tuple[MemoryView, ...] = ()  # sent verbatim with the answer request
    placeholders: tuple[MemoryView, ...] = ()  # sent only as ⟦m7⟧ and a label
    kept_local: tuple[MemoryView, ...] = ()  # withheld from every request of the turn
    checks: tuple[CheckView, ...] = ()
    actions: tuple[str, ...] = ()
    receipts: tuple[ReceiptView, ...] = ()
    outbound: tuple[OutboundView, ...] = ()
    mode: str = "replay"
    is_brief: bool = False
    on_device: bool = False  # answered on this device with no model call (an availability check)
    notice: str = ""
    error: str = ""
    intent: str = ""  # the intent the turn ran: what the Guard tab's "Prove it" check can rebuild


@dataclass(frozen=True)
class Runtime:
    """Process-wide pieces shared by every session: mode, models, clients and budget."""

    mode: str  # "live" or "replay"
    models: dict[str, str]  # "reason" / "fast" -> model id
    web_search: str  # live: "Tavily" or "off"; replay: what the recording holds (replay_web_label)
    reason: str  # why the app runs in replay, for the header badge
    llm: Any
    search: Any
    replay: Any
    budget: Any


def replay_is_recorded(book: ReplayBook) -> bool:
    """Whether every replay turn that holds model outputs or web results comes from a live recording
    (hand-written samples have no recorded_at). A turn with neither replays nothing: the PIN refusal and the
    private correction never call a model, so a live run has nothing to record for them."""
    turns = [book.lookup(question) for question in book.questions()]
    replayed = [turn for turn in turns if turn is not None and (turn.outputs or turn.web_results)]
    return bool(replayed) and all(turn.recorded_at for turn in replayed)


def replay_web_label(book: ReplayBook) -> str:
    """What replay does for web search, as the header shows it: results only if the recording holds some."""
    turns = [turn for turn in (book.lookup(question) for question in book.questions()) if turn is not None]
    recorded = replay_is_recorded(book)
    if any(turn.web_results for turn in turns):
        return "recorded results" if recorded else "sample results"
    return "off when recorded" if recorded else "none in the samples"


def recorded_models(book: ReplayBook) -> dict[str, str]:
    """Model ids named in the replay recordings (shown in the header in replay mode)."""
    for question in book.questions():
        recorded = book.lookup(question)
        if recorded is not None and recorded.models:
            return dict(recorded.models)
    return {}


def build_runtime() -> Runtime:
    """Choose live or replay from the environment and build the shared clients."""
    replay = ReplayBook.load(REPLAY_DIR)
    budget = Budget()

    def replay_runtime(reason: str) -> Runtime:
        web = replay_web_label(replay)
        if replay_is_recorded(replay):
            return Runtime("replay", recorded_models(replay), web, reason, None, None, replay, budget)
        samples = {"reason": "sample answers, no model call", "fast": "sample answers, no model call"}
        return Runtime("replay", samples, web, reason, None, None, replay, budget)

    api_key = env("NEBIUS_API_KEY")
    if not api_key:
        if replay_is_recorded(replay):
            return replay_runtime("No API key on this Space, so recorded Nemotron outputs are replayed.")
        return replay_runtime("No API key on this Space, and no live recording yet, so hand-written sample answers are shown.")
    llm = TokenFactory(api_key)
    try:
        models = discover_models(llm)
    except (ModelSelectionError, ServiceError) as exc:
        return replay_runtime(f"Live models are unavailable ({exc}), so recorded outputs are replayed.")
    tavily_key = env("TAVILY_API_KEY")
    search = Tavily(tavily_key) if tavily_key else None
    return Runtime("live", models, "Tavily" if search else "off", "", llm, search, replay, budget)


def _memory(m: Any) -> MemoryView:
    """Convert a core ``Memory`` into a view."""
    return MemoryView(
        id=m.id, version=m.version, text=m.text, label=m.label, kind=m.kind, sensitivity=m.sensitivity,
        status=m.status, source_type=m.source_type, source_ref=m.source_ref, author=m.author,
        created_at=m.created_at, note=m.note,
    )


def _receipt(r: Any) -> ReceiptView:
    """Convert a core ``Receipt`` into a view."""
    return ReceiptView(
        at=r.at, provider=r.provider, model=r.model, purpose=r.purpose, prompt_tokens=r.prompt_tokens,
        completion_tokens=r.completion_tokens, latency_ms=r.latency_ms, sent_ids=tuple(r.sent_memory_ids),
        withheld_ids=tuple(r.withheld_memory_ids), payload_sha256=r.payload_sha256,
        cost_usd=r.cost_usd_estimate, mode=r.mode, status=getattr(r, "status", "answered"),
    )


def _outbound(o: Any) -> OutboundView:
    """Convert a core ``Outbound`` into a view; the canonical JSON is what the receipt hash covers."""
    payload = o.payload
    if isinstance(payload, (bytes, str)):
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        pretty = canonical = text
    else:
        pretty = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return OutboundView(o.provider, o.model, o.purpose, pretty, canonical, o.sha256, o.mode,
                        getattr(o, "status", "answered"))


class Engine:
    """One visitor's private core objects: a fresh seed memory, a receipt log and the assistant."""

    def __init__(self, runtime: Runtime) -> None:
        """Load the seed persona into a new in-memory store."""
        self.session_id = uuid.uuid4().hex
        self.store = MemoryStore(":memory:")
        self.store.load_seed(DEMO_PERSONA)
        self.receipts = ReceiptLog(":memory:")
        self.guard = Guard(DEFAULT_POLICY)
        self.assistant = Assistant(
            self.store, self.guard, self.receipts, runtime.budget, runtime.llm, search=runtime.search,
            replay=runtime.replay, models=runtime.models if runtime.llm else None,
        )

    def ask(self, message: str) -> TurnView:
        """Send one chat message through the assistant."""
        return self._view(self.assistant.handle(message, session_id=self.session_id), message)

    def brief(self) -> TurnView:
        """Run the daily brief."""
        return self._view(self.assistant.daily_brief(self.session_id), "Daily brief")

    def memories(self) -> list[MemoryView]:
        """The current version of every memory, by id."""
        return [_memory(m) for m in self.store.all_current()]

    def histories(self, ids: Sequence[int]) -> dict[int, list[MemoryView]]:
        """Every version of each memory in ``ids``, newest first."""
        return {i: [_memory(m) for m in reversed(self.store.history(i))] for i in ids}

    def receipts_today(self) -> list[ReceiptView]:
        """Today's outbound calls, newest first."""
        return [_receipt(r) for r in reversed(self.receipts.today())]

    def summary(self) -> SummaryView:
        """Today's totals."""
        s = self.receipts.today_summary()
        return SummaryView(
            s.calls, s.replay_calls, s.prompt_tokens, s.completion_tokens, s.cost_usd_estimate, s.memories_kept_local
        )

    def today(self) -> date:
        """The date the demo persona lives in (the seed's snapshot date)."""
        return self.assistant.today()

    def close(self) -> None:
        """Close the session's databases."""
        self.store.close()
        self.receipts.close()

    def _view(self, turn: Any, question: str) -> TurnView:
        """Convert a core ``Turn`` into a view."""
        report = turn.guard
        prepared = report.prepared
        draft = getattr(turn, "local", "") or turn.draft or turn.answer  # this page is on the device
        values = {}
        if prepared is not None:
            values = {s.memory.id: s.rendered for s in self.guard.segments(draft, prepared) if s.memory is not None}
        is_brief = turn.intent == "brief"
        return TurnView(
            question=question,
            draft=BRIEF_RECEIPTS_LINE.sub("", draft) if is_brief else draft,
            # what the model wrote, when the device then changed it (filled in, trimmed, put in date order)
            written=(BRIEF_RECEIPTS_LINE.sub("", turn.draft) if is_brief else turn.draft)
            if getattr(turn, "local", "") and turn.intent in ("answer", "brief") else "",
            values=values,
            citations=tuple(_memory(m) for m in turn.citations),
            shared=tuple(_memory(m) for m in prepared.shared) if prepared else (),
            placeholders=tuple(_memory(m) for m in prepared.withheld) if prepared else (),
            kept_local=tuple(_memory(m) for m in report.kept_local),
            checks=tuple(
                CheckView(d.purpose, d.provider, d.allowed, tuple(d.reasons), d.redacted_text if d.allowed else "",
                          d.not_sent)
                for d in report.decisions
            ),
            actions=tuple(turn.actions),
            receipts=tuple(_receipt(r) for r in turn.receipts),
            outbound=tuple(_outbound(o) for o in getattr(turn, "outbound", ())),
            mode=turn.mode,
            is_brief=is_brief,
            on_device=turn.intent == "availability",
            intent=turn.intent,
            notice=turn.notice.replace(REPLAY_NOTICE, "").strip(),
        )


# =============================================================================
# Demo content and per-visitor session state
# =============================================================================
RUNTIME = build_runtime()
# The written forms of every date and time take about ten seconds to generate: start now, not at the first message.
threading.Thread(target=warm_up, name="forms-warm-up", daemon=True).start()
PERSONA_NAME = PERSONA.get("name", "Alex Martin")
PERSONA_LINE = f"a fictional {PERSONA.get('role', 'product designer').lower()} in {PERSONA.get('city', 'Lyon')}"

DEMO_SCRIPT: tuple[tuple[str, str], ...] = (
    ("What's on my plate next week?", "Private appointments are filled in on this device"),
    ("My dermatologist appointment moved to Wednesday 4 Nov at 9:00.", "Corrected on this device as version 2"),
    ("Find a 10-day taper plan for my half marathon.", "A web query, guarded before it may leave"),
    ("Draft a birthday message for Léa.", "Personal memories may go to the cloud model"),
    ("Remember my bank PIN is 4821.", "Refused: credentials are never stored or sent"),
    ("Run my daily brief.", "The week ahead, plus today's receipts"),
    ("Am I free Wednesday morning?", "Answered on this device: no model sees private dates"),
)

PROVIDER_NAMES = {NEBIUS: "Nebius Token Factory", "tavily": "Tavily web search", "local": "this device"}
SOURCE_NAMES = {
    "seed": "seed data for the demo persona", "user": "told by the user", "web": "web", "inference": "inferred",
}
PURPOSE_NAMES = {
    "route": "Routing", "classify": "Sensitivity check", "answer": "Answer", "brief": "Daily brief",
    "web search": "Web search", "web_search": "Web search", "message": "Your message",
}
SENSITIVITIES = ("public", "personal", "private")
FILTERS = ("all", *SENSITIVITIES)


SCRIPT_INDEX = {normalize_question(prompt): number for number, (prompt, _) in enumerate(DEMO_SCRIPT, start=1)}

# First screen. Hugging Face sets SPACE_ID on a Space, where Confidant runs on the Space's server.
HOSTED = bool(os.environ.get("SPACE_ID"))
HOSTED_NOTE = " On this hosted demo, Confidant runs on the Space's server, in place of your device."
LIVE_CTA = "Start with <strong>step 1</strong> of the demo script, or ask your own question."
REPLAY_CTA = (
    "Start with <strong>step 1</strong> of the demo script: its steps replay a live recording. A question of "
    "your own gets no model answer here, only the memories that match it. What the device recognises without a "
    "model works as it does live: <em>Remember that…</em>, a correction such as <em>…moved to Friday</em>, "
    "<em>Am I free…?</em>, and refusing a password or PIN."
)


@dataclass
class Session:
    """What one browser tab owns: its engine, its conversation and its UI choices."""

    engine: Engine
    turns: list[TurnView] = field(default_factory=list)
    pending: str | None = None
    memory_filter: str = "all"
    changed: frozenset[int] = frozenset()
    last_guarded: TurnView | None = None
    brief: TurnView | None = None
    brief_pending: bool = False
    fell_back: bool = False  # live mode hit a limit and replayed

    @classmethod
    def start(cls) -> Session:
        """A fresh session on a fresh copy of the seed memory."""
        return cls(engine=Engine(RUNTIME))

    def steps_done(self) -> set[int]:
        """Demo-script steps this visitor has already run."""
        return {n for t in self.turns if (n := SCRIPT_INDEX.get(normalize_question(t.question)))}

    def snapshot(self) -> dict[int, tuple[int, str]]:
        """Version and status of every memory, to spot what a turn changed."""
        return {m.id: (m.version, m.status) for m in self.engine.memories()}

    def record(self, turn: TurnView, before: dict[int, tuple[int, str]]) -> None:
        """Keep a finished turn and note which memories it changed."""
        after = self.snapshot()
        self.changed = frozenset(i for i, v in after.items() if before.get(i) != v)
        self.last_guarded = turn
        self.fell_back = self.fell_back or (RUNTIME.mode == "live" and turn.mode == "replay")


def run_safely(question: str, work: Callable[[], TurnView]) -> TurnView:
    """Run a core call; turn any failure into a visible, specific error turn."""
    try:
        return work()
    except Exception as exc:  # the page must survive a core or network failure
        reason = str(exc).strip() or exc.__class__.__name__
        return TurnView(question=question, mode=RUNTIME.mode, error=reason)


# =============================================================================
# HTML rendering
# Each function returns markup for a gr.HTML component. Text from the model,
# the memory or the visitor is always escaped before it goes into markup.
# =============================================================================
def esc(value: object) -> str:
    """Escape text for HTML content and attributes."""
    return html.escape(str(value), quote=True)


def sha_html(value: str) -> str:
    """A SHA-256 that may wrap in its middle, and only there, when its panel is too narrow: cut at the edge, it
    showed 63 of its 64 characters. The text read or copied from the page is still the whole hash."""
    return f"{esc(value[:32])}<wbr>{esc(value[32:])}" if len(value) == 64 else esc(value)


def plural(n: int, one: str, many: str) -> str:
    """``1 memory`` / ``3 memories``."""
    return f"{n} {one if n == 1 else many}"


def day(iso: str) -> str:
    """``2026-10-26T18:02:11Z`` -> ``2026-10-26``."""
    return iso[:10]


def minute(iso: str) -> str:
    """``2026-10-26T18:02:11Z`` -> ``2026-10-26 18:02 UTC``."""
    return f"{iso[:10]} {iso[11:16]} UTC" if len(iso) >= 16 else iso


def capitalised(text: str) -> str:
    """First letter upper-case, the rest untouched (labels such as "sister's birthday")."""
    return text[:1].upper() + text[1:]


GLYPHS = {
    "public": '<circle cx="6" cy="6" r="4.4" fill="none" stroke="currentColor" stroke-width="1.6"/>',
    "personal": '<circle cx="6" cy="6" r="4.4" fill="none" stroke="currentColor" stroke-width="1.6"/>'
    '<path d="M6 1.6a4.4 4.4 0 0 1 0 8.8z" fill="currentColor"/>',
    "private": '<circle cx="6" cy="6" r="5.2" fill="currentColor"/>',
}
GUARD_ICON = (
    '<svg viewBox="0 0 32 32" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="2.6" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M11.5 5.5H5.5v21h6"/><path d="M20.5 5.5h6v21h-6"/></g>'
    '<rect x="11" y="12.5" width="10" height="7" rx="1.6" fill="currentColor"/></svg>'
)
CHEVRON = (
    '<svg class="cf-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3.5 10.5 8 6 12.5" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def glyph(sensitivity: str) -> str:
    """Ink-density mark: hollow = public, half = personal, solid = private."""
    shape = GLYPHS.get(sensitivity, GLYPHS["public"])
    return f'<svg class="cf-glyph" viewBox="0 0 12 12" aria-hidden="true">{shape}</svg>'


def mark_svg() -> str:
    """The logo mark, inlined so it follows the theme colours."""
    svg = (Path(__file__).resolve().parent / "assets" / "logo.svg").read_text(encoding="utf-8")
    return re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)


def sensitivity_label(sensitivity: str) -> str:
    """Glyph plus the capitalised sensitivity."""
    return f'<span class="cf-sens">{glyph(sensitivity)}{esc(sensitivity.capitalize())}</span>'


def reach(sensitivity: str) -> str:
    """Plain-language summary of where a memory of this sensitivity may go."""
    names = [p for p in PROVIDER_POLICY if p != "local"]
    allowed = [PROVIDER_NAMES.get(p, p) for p in names if sensitivity in PROVIDER_POLICY[p]]
    blocked = [PROVIDER_NAMES.get(p, p) for p in names if sensitivity not in PROVIDER_POLICY[p]]
    if not allowed:
        return "Never. Cloud models only see a placeholder and its label; the value is filled in on this device."
    text = f"May be sent to {' and '.join(allowed)}."
    return text + (f" Never sent to {' or '.join(blocked)}." if blocked else "")


# --- Rich text: model output to safe HTML ----------------------------------------
_CITE = re.compile(r"\[m(\d+)v(\d+)\]")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
_BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_HEADING = re.compile(r"^\s*#{1,6}\s+(.*)$")


def cite_button(memory_id: int, version: int) -> str:
    """Inline citation that opens the memory's provenance."""
    return (
        f'<button type="button" class="cf-cite" data-mem="{memory_id}" data-ver="{version}" '
        f'title="Open memory m{memory_id}, version {version}">m{memory_id}v{version}</button>'
    )


def token_html(memory_id: int, *, known: bool = True) -> str:
    """A placeholder as the cloud model saw it."""
    if not known:
        return (
            '<span class="cf-token cf-token--unknown" title="Unknown placeholder, shown as written">'
            f"⟦m{memory_id}⟧</span>"
        )
    return (
        f'<button type="button" class="cf-token" data-mem="{memory_id}" '
        f'title="Private memory m{memory_id}: its value stayed on this device">⟦m{memory_id}⟧</button>'
    )


def filled_html(value: str, memory_id: int) -> str:
    """A private value filled in locally after the cloud model answered."""
    return (
        f'<mark class="cf-filled" data-mem="{memory_id}" tabindex="0" '
        f'title="Filled in on this device from m{memory_id}. The cloud model only saw ⟦m{memory_id}⟧.">'
        f"{esc(value)}</mark>"
    )


def _inline(text: str, fill: Mapping[int, str], known: Collection[int]) -> str:
    """Escape one line, then add bold, code, citations and placeholders."""
    def placeholder(match: re.Match[str]) -> str:
        memory_id = int(match[1])
        if memory_id in fill:
            return filled_html(fill[memory_id], memory_id)
        return token_html(memory_id, known=memory_id in known)

    out = html.escape(text, quote=False)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _CODE.sub(r"<code>\1</code>", out)
    out = _CITE.sub(lambda m: cite_button(int(m[1]), int(m[2])), out)
    return PLACEHOLDER_RE.sub(placeholder, out)


def render_rich(text: str, fill: Mapping[int, str] | None = None, known: Collection[int] = ()) -> str:
    """Render model text as HTML: paragraphs, lists, headings, **bold**, citations, placeholders.

    Each placeholder ``⟦m7⟧`` whose id is in ``fill`` shows the value filled in
    on this device; otherwise it is drawn as a token, flagged as unknown when
    its id is not in ``known`` either.
    """
    fill = fill or {}
    text = drop_echoed_labels(text, fill) if fill else text  # the filled-in value replaces the label it repeats
    blocks: list[str] = []
    paragraph: list[str] = []
    items: list[str] = []
    list_tag = ""

    def flush() -> None:
        nonlocal list_tag
        if paragraph:
            blocks.append(f"<p>{'<br>'.join(paragraph)}</p>")
            paragraph.clear()
        if items:
            blocks.append(f"<{list_tag}>{''.join(f'<li>{i}</li>' for i in items)}</{list_tag}>")
            items.clear()
            list_tag = ""

    for line in text.strip().splitlines():
        bullet, numbered, heading = _BULLET.match(line), _NUMBERED.match(line), _HEADING.match(line)
        if not line.strip():
            flush()
        elif bullet or numbered:
            tag = "ul" if bullet else "ol"
            if paragraph or (items and tag != list_tag):
                flush()
            list_tag = tag
            items.append(_inline((bullet or numbered)[1], fill, known))
        elif heading:
            flush()
            blocks.append(f'<p class="cf-subhead">{_inline(heading[1], fill, known)}</p>')
        else:
            if items:
                flush()
            paragraph.append(_inline(line.strip(), fill, known))
    flush()
    return "".join(blocks)


# --- Header and demo script ---------------------------------------------------------
def render_header(runtime: Runtime, *, fell_back: bool = False) -> str:
    """Name, tagline, mode badge and the model ids in use."""
    if runtime.mode == "live" and not fell_back:
        badge = (
            '<span class="cf-badge cf-badge--live"><span class="cf-badge__dot"></span>Live'
            '<span class="cf-badge__extra">· Nemotron on Nebius Token Factory</span></span>'
        )
    else:
        why = ("live limit reached, replaying recorded turns" if fell_back
               else "recorded Nemotron turns" if replay_is_recorded(runtime.replay) else "sample answers, not a live recording")
        title = runtime.reason or "The live budget is used up for now, so recorded outputs are replayed."
        title += " Memory, guard, late binding and receipts still run here."
        badge = (
            f'<span class="cf-badge" title="{esc(title)}"><span class="cf-badge__dot"></span>Replay'
            f'<span class="cf-badge__extra">· {esc(why)}</span></span>'
        )
    rows = (
        ("Answers", runtime.models.get("reason", "unavailable")),
        ("Routing", runtime.models.get("fast", "unavailable")),
        ("Web search", runtime.web_search),
    )
    models = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
    return (
        '<header class="cf cf-header">'
        f'<div class="cf-brand"><span class="cf-mark">{mark_svg()}</span><div>'
        '<h1 class="cf-wordmark">Confidant</h1>'
        '<p class="cf-tagline">A personal AI that keeps your secrets and shows its receipts.</p></div></div>'
        f'<div class="cf-mode">{badge}<dl class="cf-models" aria-label="Models in use">{models}</dl></div>'
        "</header>"
    )


def render_script(done: set[int], busy: bool) -> str:
    """The demo script; each step sends its prompt."""
    steps = []
    for number, (prompt, note) in enumerate(DEMO_SCRIPT, start=1):
        state = " is-done" if number in done else ""
        disabled = " disabled" if busy else ""
        steps.append(
            f'<li><button type="button" class="cf-step{state}" data-prompt="{esc(prompt)}"{disabled}>'
            f'<span class="cf-step__num" aria-hidden="true">{number}</span>'
            f'<span class="cf-step__body"><span class="cf-step__prompt">{esc(prompt)}</span>'
            f'<span class="cf-step__note">{esc(note)}</span></span></button></li>'
        )
    return (
        '<nav class="cf cf-script" aria-label="Demo script">'
        '<div><h2 class="cf-script__title">Demo script</h2>'
        '<p class="cf-script__intro">Seven steps, about two minutes. Select a step to send it.</p></div>'
        f'<ol class="cf-steps">{"".join(steps)}</ol></nav>'
    )


# --- Conversation -------------------------------------------------------------------
def render_chips(citations: Sequence[MemoryView]) -> str:
    """Source chips: ``m7v2 · user · 2026-10-26``; each opens provenance and history."""
    chips = "".join(
        f'<button type="button" class="cf-chip" data-mem="{m.id}" data-ver="{m.version}" '
        f'title="{esc(capitalised(m.label))}, {esc(m.sensitivity)}">{glyph(m.sensitivity)}'
        f'<span class="cf-chip__id">{m.cite}</span><span>· {esc(m.source_type)}</span>'
        f'<span class="cf-chip__date">· {day(m.created_at)}</span></button>'
        for m in citations
    )
    return f'<div class="cf-chips"><span class="cf-chips__label">Sources</span>{chips}</div>' if chips else ""


def answered(turn: TurnView) -> bool:
    """Whether the turn's answer request went out (or, in replay, was replayed)."""
    return any(r.purpose in ("answer", "brief") for r in turn.receipts)


def render_guardline(turn: TurnView) -> str:
    """One line on what the guard kept back; opens the Guard tab."""
    kept = len(turn.kept_local)
    if kept:
        main = f"{plural(kept, 'private memory', 'private memories')} stayed on this device"
    elif not turn.receipts:
        main = "Nothing was sent to the cloud for this turn"
    else:
        main = "No private memory was involved"
    shared = len(turn.shared) if answered(turn) else 0
    if shared:
        aside = f"{plural(shared, 'memory', 'memories')} shared with Nemotron. See exactly what was sent."
    elif turn.receipts:
        aside = "See exactly what was sent."
    else:
        aside = "See what the guard checked."
    return (
        '<button type="button" class="cf-guardline" data-open-tab="guard">'
        f'{GUARD_ICON}<span class="cf-guardline__text"><span class="cf-guardline__main">{main}</span>'
        f'<span class="cf-guardline__aside">{aside}</span></span>{CHEVRON}</button>'
    )


def render_actions(turn: TurnView) -> str:
    """What the turn changed (recorded, corrected, refused), plus the replay label."""
    tags = []
    for action in turn.actions:
        refused = action.lower().startswith("refused")
        text = re.sub(r"\b(m\d+(?:v\d+)?)\b", r"<code>\1</code>", esc(capitalised(action)))
        tags.append(f'<li class="cf-action{" cf-action--refused" if refused else ""}">{text}</li>')
    if turn.mode == "replay" and turn.receipts:  # a turn answered on the device replayed nothing
        tags.append('<li><span class="cf-tag" title="Model outputs replayed from data/replay">replay</span></li>')
    return f'<ul class="cf-actions">{"".join(tags)}</ul>' if tags else ""


def render_slip(summary: SummaryView) -> str:
    """This session's receipt totals, printed like a till slip."""
    calls = f"{summary.calls}" + (f" ({summary.replay_calls} replayed)" if summary.replay_calls else "")
    return (
        '<div class="cf-slip"><div class="cf-slip__head">This session<span>receipts</span></div>'
        '<dl class="cf-leaders">'
        f"<div><dt>Outbound calls</dt><dd>{calls}</dd></div>"
        f"<div><dt>Tokens sent</dt><dd>{summary.prompt_tokens:,}</dd></div>"
        f"<div><dt>Tokens received</dt><dd>{summary.completion_tokens:,}</dd></div>"
        f"<div><dt>Estimated cost</dt><dd>${summary.cost_usd:.4f}</dd></div>"
        '<div class="cf-leaders__strong"><dt>Withheld memories that never left</dt>'
        f"<dd>{summary.kept_local}</dd></div>"
        "</dl>"
        '<p class="cf-slip__foot">Receipts keep a SHA-256 of each payload, never the payload itself. '
        "Replayed calls send nothing and use no tokens.</p></div>"
    )


def render_reply(turn: TurnView, summary: SummaryView | None = None) -> str:
    """The assistant's side of a turn: answer, guard line, sources and actions."""
    if turn.error:
        return (
            '<div class="cf-reply"><p class="cf-error">'
            f"No answer came back: {esc(turn.error)}. Try again, or run a demo step.</p></div>"
        )
    slip = render_slip(summary) if summary and turn.is_brief else ""
    notice = f'<p class="cf-notice">{esc(turn.notice)}</p>' if turn.notice else ""
    return (
        f'<div class="cf-reply"><div class="cf-answer">{render_rich(turn.draft, turn.values)}</div>{slip}'
        f'<div class="cf-footer">{render_guardline(turn)}{render_chips(turn.citations)}'
        f"{render_actions(turn)}{notice}</div></div>"
    )


def render_user(question: str) -> str:
    """The visitor's message, labelled with its demo step when it is one."""
    step = SCRIPT_INDEX.get(normalize_question(question))
    label = f'<span class="cf-user__step">Step {step}</span>' if step else ""
    return f'<div class="cf-user">{label}<p>{esc(question)}</p></div>'


def render_pending(text: str) -> str:
    """A working indicator."""
    return (
        '<p class="cf-pending"><span class="cf-pending__dots" aria-hidden="true">'
        f"<i></i><i></i><i></i></span>{esc(text)}</p>"
    )


def render_empty(memories: Sequence[MemoryView], today: date, *, replay: bool, hosted: bool = HOSTED) -> str:
    """First screen: who you are, late binding shown on a real private memory, and what this mode can do."""
    private = [m for m in memories if m.sensitivity == "private"]
    example = next((m for m in private if m.kind == "event"), private[0] if private else None)
    binding = ""
    if example:
        binding = (
            '<figure class="cf-binding">'
            '<div class="cf-binding__row">'
            '<p class="cf-binding__who">Nemotron writes<small>on Nebius Token Factory</small></p>'
            f'<p class="cf-binding__text">Also on your calendar: {token_html(example.id)}</p></div>'
            '<div class="cf-binding__row">'
            '<p class="cf-binding__who">You read<small>filled in by Confidant</small></p>'
            f'<p class="cf-binding__text">Also on your calendar: {filled_html(example.text, example.id)}</p></div>'
            "</figure>"
        )
    return (
        '<div class="cf cf-transcript"><section class="cf-empty">'
        f'<h2 class="cf-empty__title">You are {esc(PERSONA_NAME)} today.</h2>'
        f'<p class="cf-empty__lead">Confidant keeps {len(memories)} memories about {esc(PERSONA_NAME)}, '
        f"{esc(PERSONA_LINE)}, and the demo is set on {today:%A} {today.day} {today:%B %Y}. "
        "Private memories never leave Confidant: Nemotron sees a placeholder and writes around it, "
        "and Confidant fills in the value before you read it."
        f"{HOSTED_NOTE if hosted else ''}</p>"
        f"{binding}"
        f'<p class="cf-empty__cta">{REPLAY_CTA if replay else LIVE_CTA}</p>'
        "</section></div>"
    )


def render_transcript(session: Session, memories: Sequence[MemoryView], summary: SummaryView) -> str:
    """The whole conversation, or the empty state before the first message."""
    if not session.turns and not session.pending:
        return render_empty(memories, session.engine.today(), replay=RUNTIME.mode != "live")
    turns = [
        f'<article class="cf-turn">{render_user(t.question)}'
        f"{render_reply(t, summary if t is session.turns[-1] else None)}</article>"
        for t in session.turns
    ]
    if session.pending:
        turns.append(
            f'<article class="cf-turn">{render_user(session.pending)}'
            f'<div class="cf-reply">{render_pending("Working on it")}</div></article>'
        )
    return f'<div class="cf cf-transcript" aria-live="polite">{"".join(turns)}</div>'


# --- Memory tab ----------------------------------------------------------------------
def filter_choices(memories: Sequence[MemoryView]) -> list[tuple[str, str]]:
    """Filter labels with counts, e.g. ``Private 4``."""
    counts = Counter(m.sensitivity for m in memories)
    return [(f"All {len(memories)}", "all")] + [(f"{s.capitalize()} {counts[s]}", s) for s in SENSITIVITIES]


def render_memory_head(memories: Sequence[MemoryView]) -> str:
    """Title and one line of context above the filter."""
    return (
        '<div class="cf cf-panel__head"><h3 class="cf-panel__title">What Confidant remembers</h3>'
        f'<p class="cf-panel__lead">{len(memories)} memories about {esc(PERSONA_NAME)}, a fictional persona. '
        "Changes stay in this browser tab. Select a row for its source and version history.</p></div>"
    )


def source_text(memory: MemoryView) -> str:
    """Where a memory came from, as markup (web sources link to their page)."""
    ref, kind = memory.source_ref, SOURCE_NAMES.get(memory.source_type, esc(memory.source_type))
    if ref.startswith(("http://", "https://")):
        return f'web · <a href="{esc(ref)}" target="_blank" rel="noopener noreferrer">{esc(ref)}</a>'
    if memory.source_type == "seed" or ref in ("", memory.source_type):
        return kind
    return f"{kind} · session {esc(ref[:8])}"


def render_memory_detail(history: Sequence[MemoryView]) -> str:
    """Provenance and version history of one memory (shown in the dialog)."""
    current = history[0]
    versions = "".join(
        f'<li data-ver="{m.version}" class="{"is-current" if m.status == "current" else ""}">'
        f'<p class="cf-history__text">{esc(m.text)}</p>'
        f'<p class="cf-history__meta"><code>{m.cite}</code> · {esc(m.status)} · {esc(m.source_type)} · '
        f"{minute(m.created_at)} · {esc(m.author)}{' · ' + esc(m.note) if m.note else ''}</p></li>"
        for m in history
    )
    return (
        f'<p class="cf-detail__kicker"><code>m{current.id}</code>{sensitivity_label(current.sensitivity)}'
        f"<span>· {esc(current.kind)}</span></p>"
        f'<h3 class="cf-detail__title" id="cf-dialog-title">{esc(capitalised(current.label))}</h3>'
        f'<p class="cf-detail__text">{esc(current.text)}</p>'
        '<dl class="cf-facts">'
        f"<dt>Source</dt><dd>{source_text(current)}</dd>"
        f"<dt>Author</dt><dd>{esc(current.author)}</dd>"
        f"<dt>Recorded</dt><dd>{minute(current.created_at)}</dd>"
        f"<dt>Version</dt><dd>{current.version} of {len(history)}, {esc(current.status)}</dd>"
        f"<dt>Leaves this device</dt><dd>{esc(reach(current.sensitivity))}</dd>"
        "</dl>"
        f'<h4 class="cf-detail__section">Version history</h4><ol class="cf-history">{versions}</ol>'
    )


def render_memory_table(
    memories: Sequence[MemoryView], histories: Mapping[int, list[MemoryView]], active: str, changed: frozenset[int]
) -> str:
    """The filtered memory table, plus the hidden provenance templates the dialog shows."""
    rows = []
    for m in memories:
        if active not in ("all", m.sensitivity):
            continue
        new = '<span class="cf-new">changed</span>' if m.id in changed else ""
        rows.append(
            f'<tr data-mem="{m.id}" tabindex="0" class="{"is-changed" if m.id in changed else ""}" '
            f'aria-label="Open m{m.id}, {esc(m.label)}">'
            f'<td class="cf-col-id">m{m.id}</td>'
            f'<td class="cf-col-label"><strong>{esc(capitalised(m.label))}{new}</strong><span>{esc(m.text)}</span></td>'
            f"<td>{sensitivity_label(m.sensitivity)}</td>"
            f'<td class="cf-col-source">{esc(m.source_type)}</td>'
            f'<td class="cf-col-ver">v{m.version}</td>'
            f'<td class="cf-col-status cf-status--{esc(m.status)}">{esc(m.status)}</td></tr>'
        )
    body = "".join(rows) or '<tr><td colspan="6">No memories with this sensitivity.</td></tr>'
    templates = "".join(
        f'<template data-mem-detail="{i}">{render_memory_detail(h)}</template>' for i, h in histories.items() if h
    )
    return (
        '<div class="cf"><div class="cf-tablewrap"><table class="cf-table">'
        "<thead><tr><th>ID</th><th>Memory</th><th>Sensitivity</th>"
        '<th class="cf-col-source">Source</th><th>Ver.</th><th class="cf-col-status">Status</th></tr></thead>'
        f'<tbody>{body}</tbody></table></div><div id="cf-mem-details" hidden>{templates}</div></div>'
    )


# --- Guard tab -----------------------------------------------------------------------
def render_memlinks(memories: Sequence[MemoryView], why: Callable[[MemoryView], str], empty: str) -> str:
    """A list of memories, each opening its provenance, with the guard's reason."""
    if not memories:
        return f'<p class="cf-empty-list">{esc(empty)}</p>'
    items = "".join(
        f'<li><button type="button" class="cf-memlink" data-mem="{m.id}" data-ver="{m.version}">{glyph(m.sensitivity)}'
        f'<span class="cf-memlink__label"><code>{m.cite}</code>{esc(capitalised(m.label))}</span>'
        f'<span class="cf-memlink__why">{esc(why(m))}</span></button></li>'
        for m in memories
    )
    return f'<ul class="cf-memlist">{items}</ul>'


def render_payload(text: str, known: Collection[int]) -> str:
    """An outbound payload exactly as sent, with citations and placeholders highlighted."""
    out = html.escape(text, quote=False)
    out = _CITE.sub(lambda m: f'<span class="cf-ctx-cite">{m[0]}</span>', out)
    out = PLACEHOLDER_RE.sub(lambda m: token_html(int(m[1]), known=int(m[1]) in known), out)
    return f'<pre class="cf-context">{out}</pre>'


def render_check(check: CheckView, known: Collection[int], *, open_text: bool) -> str:
    """One request the guard checked: verdict, destination, reasons and the exact text. A request the guard
    allowed that never left says so, with the reason the assistant recorded (web search off, or a replay with
    no recorded search); a request that left and failed is not one of them."""
    notes = list(check.reasons)
    if not check.allowed:
        verdict, css = "Blocked", " cf-verdict--blocked"
    elif check.not_sent:
        verdict, css = "Not sent", ""
        notes.append(f"the guard allowed it, but {check.not_sent}")
    elif check.reasons:
        verdict, css = "Redacted", ""
    else:
        verdict, css = "Allowed", ""
    reasons = "".join(f'<span class="cf-check__reasons">{esc(capitalised(r))}</span>' for r in notes)
    payload = ""
    if check.text:
        payload = (
            f'<details class="cf-payload"{" open" if open_text else ""}><summary>Request after the guard</summary>'
            f"{render_payload(check.text, known)}</details>"
        )
    what = f"{PURPOSE_NAMES.get(check.purpose, check.purpose)} to {PROVIDER_NAMES.get(check.provider, check.provider)}"
    return (
        f'<li class="cf-check"><span class="cf-verdict{css}">{verdict}</span>'
        f'<div class="cf-check__body"><span class="cf-check__what">{esc(what)}</span>{reasons}{payload}</div></li>'
    )


def outbound_verdict(call: OutboundView) -> str:
    """How a call ended, as the Guard tab labels it."""
    if call.mode == "replay":
        return "Replay" if call.status == "answered" else f"Replay, {call.status}"
    return "Sent" if call.status == "answered" else f"Sent, {call.status}"


def render_outbound(calls: Sequence[OutboundView], known: Collection[int]) -> str:
    """Every call of the turn exactly as it left the device, with the hash its receipt keeps."""
    if not calls:
        return '<p class="cf-empty-list">Nothing left this device.</p>'
    items = []
    for call in calls:
        what = f"{PURPOSE_NAMES.get(call.purpose, call.purpose)} to {PROVIDER_NAMES.get(call.provider, call.provider)}"
        verb = "would send in a live run" if call.mode == "replay" else "sent"
        body = ("request body" if call.provider == "tavily"
                else "messages (the request also names the model, max_tokens and temperature)")
        items.append(
            f'<li class="cf-check"><span class="cf-verdict">{esc(outbound_verdict(call))}</span>'
            f'<div class="cf-check__body"><span class="cf-check__what">{esc(what)} · {esc(call.model)}</span>'
            f'<span class="cf-check__reasons">SHA-256 <code>{sha_html(call.payload_sha256)}</code>, the same as its '
            "receipt. Hash the compact JSON below to check it.</span>"
            f'<details class="cf-payload"><summary>Exact {body} Confidant {verb} (JSON)</summary>'
            f"{render_payload(call.payload_json, known)}"
            f'<p class="cf-panel__lead">Compact form, the one hashed:</p><pre class="cf-context">'
            f"{html.escape(call.canonical_json, quote=False)}</pre></details></div></li>"
        )
    return f'<ul class="cf-checks">{"".join(items)}</ul>'


def render_guard_panel(turn: TurnView | None) -> str:
    """For the last turn: what went to the cloud, what stayed, and the exact text."""
    if turn is None:
        return (
            '<div class="cf cf-panel"><div class="cf-panel__head"><h3 class="cf-panel__title">The guard</h3>'
            '<p class="cf-panel__lead">Decides, memory by memory, what may leave this device.</p></div>'
            '<p class="cf-placeholder">Send a message first. This tab then shows what the last turn shared with '
            "Nemotron, which memories stayed on this device and why, and the exact text of every request.</p></div>"
        )
    cloud = PROVIDER_NAMES[NEBIUS]
    shared_title = "Shared with Nemotron" if answered(turn) or not turn.shared else "Prepared, not sent"
    placeholder_ids = {m.id for m in turn.placeholders}
    sent = render_memlinks(
        turn.shared, lambda m: f"{m.sensitivity.capitalize()} memories may go to {cloud}", "No memory was shared."
    )
    if turn.on_device:
        shared_title = "Read on this device"
        sent = render_memlinks(turn.shared, lambda m: "Checked here; no model saw it this turn", "No shared item.")
    went_out = answered(turn)
    kept = render_memlinks(
        turn.kept_local,
        lambda m: (
            f"Sent only as ⟦m{m.id}⟧ and its label, filled in here"
            if m.id in placeholder_ids and went_out
            else "Kept out of every request" if went_out else "Stayed on this device: nothing was sent this turn"
        ),
        "No private memory was involved.",
    )
    known = {m.id for m in turn.kept_local} | placeholder_ids
    main = next((i for i, c in enumerate(turn.checks) if c.purpose in ("answer", "brief") and c.text), -1)
    checks = "".join(render_check(c, known, open_text=i == main) for i, c in enumerate(turn.checks))
    replay_note = (
        '<p class="cf-panel__lead">Replayed turn: these are the exact requests a live run makes. '
        "In replay nothing leaves at all.</p>"
        if turn.mode == "replay" and turn.receipts else
        '<p class="cf-panel__lead">Answered on this device: no request was made, live or replayed.</p>'
        if turn.on_device else ""
    )
    compare = ""
    if turn.values:
        compare = (
            '<section class="cf-section"><h4 class="cf-section-title">Late binding</h4><div class="cf-binding">'
            f'<div class="cf-binding__row"><p class="cf-binding__who">{"Nemotron wrote" if went_out else "Written on this device"}</p>'
            f'<div class="cf-binding__text cf-answer">{render_rich((went_out and turn.written) or turn.draft, known=known)}</div></div>'
            '<div class="cf-binding__row"><p class="cf-binding__who">You read'
            f'{"<small>filled in and put in date order on this device</small>" if went_out and turn.written else ""}</p>'
            f'<div class="cf-binding__text cf-answer">{render_rich(turn.draft, turn.values)}</div></div>'
            "</div></section>"
        )
    requests = f'<ul class="cf-checks">{checks}</ul>' if checks else '<p class="cf-empty-list">None.</p>'
    return (
        '<div class="cf cf-panel"><div class="cf-panel__head"><h3 class="cf-panel__title">Last turn</h3>'
        f'<p class="cf-quote">{esc(turn.question)}</p>{replay_note}</div>'
        '<div class="cf-split">'
        f'<section class="cf-side"><h4 class="cf-section-title">{shared_title}'
        f'<span class="cf-count">{len(turn.shared)}</span></h4>{sent}</section>'
        f'<section class="cf-side cf-side--kept"><h4 class="cf-section-title">Kept on this device'
        f'<span class="cf-count">{len(turn.kept_local)}</span></h4>{kept}</section>'
        "</div>"
        '<section class="cf-section"><h4 class="cf-section-title">Requests checked by the guard'
        f'<span class="cf-count">{len(turn.checks)}</span></h4>{requests}</section>'
        '<section class="cf-section"><h4 class="cf-section-title">Exactly what left this device'
        f'<span class="cf-count">{len(turn.outbound)}</span></h4>{render_outbound(turn.outbound, known)}</section>'
        f"{compare}</div>"
    )


# --- Prove it (Guard tab) ------------------------------------------------------------
PROOF_INTRO = (
    '<div class="cf cf-panel cf-proof"><p class="cf-panel__lead">After a question or the daily brief, this rebuilds '
    "its requests on this device, with no model call and nothing sent: once with your memory as it is, once with "
    "every private appointment moved to another week, weekday and hour, and, when your question names a date or "
    "a time, once with them moved onto it. If Nemotron could learn when they are, the requests would differ.</p></div>"
)


def render_proof(turn: TurnView, proof: Proof) -> str:
    """The comparison: what moved, each request's SHA-256 both ways, and the verdict."""
    moved = "".join(
        f'<li class="cf-check"><span class="cf-verdict">{token_html(move.memory_id)}</span>'
        f'<div class="cf-check__body"><span class="cf-check__what">{esc(move.before)}</span>'
        f'<span class="cf-check__reasons">Moved to: {esc(move.after)}</span></div></li>'
        for move in proof.moves
    ) or '<li class="cf-empty-list">No private appointment to move.</li>'
    receipts = {call.purpose: call.payload_sha256 for call in turn.outbound}
    rows = []
    for (purpose, before), (_, after) in zip(proof.original, proof.moved):
        same = before == after
        what = f"{PURPOSE_NAMES.get(purpose, purpose)} to {PROVIDER_NAMES.get(NEBIUS, NEBIUS)}"
        mark = " (its receipt's hash)" if receipts.get(purpose) == before else ""
        rows.append(
            f'<li class="cf-check"><span class="cf-verdict{"" if same else " cf-verdict--blocked"}">'
            f'{"Same" if same else "Differs"}</span><div class="cf-check__body"><span class="cf-check__what">'
            f'{esc(what)}</span><span class="cf-check__reasons">As stored{mark}: <code>{sha_html(before)}</code></span>'
            f'<span class="cf-check__reasons">Moved: <code>{sha_html(after)}</code></span></div></li>'
        )
    aimed = ""
    if proof.aimed_moves:
        same = proof.aimed == proof.original
        aimed = (
            '<section class="cf-section"><h4 class="cf-section-title">Also moved onto what you asked about'
            f'<span class="cf-count">{len(proof.aimed_moves)}</span></h4><ul class="cf-checks">'
            f'<li class="cf-check"><span class="cf-verdict{"" if same else " cf-verdict--blocked"}">'
            f'{"Same" if same else "Differs"}</span><div class="cf-check__body"><span class="cf-check__what">'
            f'{esc(proof.aimed_at)}</span><span class="cf-check__reasons">'
            f'{"Every request keeps the same SHA-256." if same else "A request changes."}</span></div></li>'
            "</ul></section>"
        )
    if not proof.sent:
        verdict = "This turn sends nothing, either way: it is answered on this device."
    elif proof.identical:
        verdict = ("Identical, byte for byte. For this question, Nemotron cannot tell when these appointments are, "
                   "nor whether one falls in the period you asked about.")
    else:
        verdict = "The requests differ: what Nemotron receives would depend on when a private item happens."
    return (
        '<div class="cf cf-panel cf-proof"><div class="cf-panel__head"><h3 class="cf-panel__title">Prove it</h3>'
        f'<p class="cf-quote">{esc(turn.question)}</p>'
        '<p class="cf-panel__lead">Rebuilt on this device, with no model call and nothing sent. The moves exist '
        "only for this check; your memory is unchanged.</p></div>"
        '<section class="cf-section"><h4 class="cf-section-title">Moved for the check'
        f'<span class="cf-count">{len(proof.moves)}</span></h4><ul class="cf-checks">{moved}</ul></section>'
        '<section class="cf-section"><h4 class="cf-section-title">Each request, SHA-256'
        f'<span class="cf-count">{len(rows)}</span></h4>'
        f'<ul class="cf-checks">{"".join(rows) or "<li class=cf-empty-list>No request.</li>"}</ul></section>'
        f'{aimed}<p class="cf-proof__verdict{"" if proof.identical else " cf-proof__verdict--bad"}">{esc(verdict)}</p></div>'
    )


def run_proof(session: Session | None) -> str:
    """Rebuild the last turn's requests with the private appointments moved, and compare."""
    turn = session.last_guarded if session is not None else None
    if turn is None or turn.error or turn.intent not in ("answer", "brief"):
        return (
            '<div class="cf cf-panel cf-proof"><p class="cf-panel__lead">Ask a question about your memory first, '
            "or run the daily brief: this check rebuilds the requests of that turn. Web searches and corrections "
            "are not covered.</p></div>"
        )
    try:
        proof = prove(session.engine.store, turn.question, intent=turn.intent, guard=session.engine.guard)
    except Exception as exc:  # the page must survive; say what failed
        return f'<div class="cf cf-panel cf-proof"><p class="cf-panel__lead">The check could not run: {esc(exc)}</p></div>'
    return render_proof(turn, proof)


# --- Receipts and brief tabs --------------------------------------------------------
def render_call(r: ReceiptView) -> str:
    """One receipt: what was called, with what, and the payload hash."""
    recorded = replay_is_recorded(RUNTIME.replay)
    if r.mode == "replay":
        source = "a live recording" if recorded else "a hand-written sample"
        numbers = f"Replayed from {source}: nothing was sent and no tokens were used."
    elif r.status != "answered":
        numbers = f"{esc(capitalised(r.status))}: the request left this device; no tokens were reported."
    else:
        numbers = (
            f"{r.prompt_tokens:,} tokens in, {r.completion_tokens:,} out · {r.latency_ms:,} ms · ${r.cost_usd:.4f}"
        )
    ids = f'Sent {" ".join(f"m{i}" for i in r.sent_ids) or "no memories"}'
    if r.withheld_ids:
        ids += " · withheld " + " ".join(f"m{i}" for i in r.withheld_ids)
    tag = '<span class="cf-tag">replay</span>' if r.mode == "replay" else ""
    return (
        '<li class="cf-call"><div class="cf-call__top">'
        f'<span class="cf-call__purpose">{esc(PURPOSE_NAMES.get(r.purpose, r.purpose))}</span>'
        f'<span class="cf-call__time">{esc(r.at[11:19])} UTC</span>{tag}</div>'
        f'<div class="cf-call__provider">{esc(PROVIDER_NAMES.get(r.provider, r.provider))}</div>'
        f'<div class="cf-call__model">{esc(r.model if r.mode != "replay" or recorded else "sample, no model call")}</div>'
        f'<div class="cf-call__nums">{numbers}</div><div class="cf-call__nums">{ids}</div>'
        f'<div class="cf-call__hash" title="{esc(r.payload_sha256)}">sha256 {esc(r.payload_sha256[:16])}…</div></li>'
    )


def render_receipts(summary: SummaryView, receipts: Sequence[ReceiptView]) -> str:
    """Today's totals and one entry per outbound call."""
    if receipts:
        calls = f'<ol class="cf-calls">{"".join(render_call(r) for r in receipts)}</ol>'
    else:
        calls = (
            '<p class="cf-placeholder">No outbound calls yet. Each call to Nemotron or Tavily appears here with '
            "its tokens, latency, cost estimate and a hash of the payload.</p>"
        )
    return (
        '<div class="cf cf-panel"><div class="cf-panel__head"><h3 class="cf-panel__title">Receipts</h3>'
        '<p class="cf-panel__lead">Every request that left this device during this session, answered or not: a '
        "request that got no answer (an HTTP 503, a timeout) has its receipt too.</p></div>"
        f'{render_slip(summary)}<section class="cf-section"><h4 class="cf-section-title">Calls'
        f'<span class="cf-count">{len(receipts)}</span></h4>{calls}</section></div>'
    )


BRIEF_HEAD = (
    '<div class="cf cf-panel__head"><h3 class="cf-panel__title">Daily brief</h3>'
    '<p class="cf-panel__lead">The week ahead and open items from memory, written in one guarded call. '
    "Private details are filled in on this device.</p></div>"
)


def render_brief(session: Session, summary: SummaryView) -> str:
    """The daily brief tab output."""
    if session.brief_pending:
        return f'<div class="cf cf-brief">{render_pending("Writing the brief")}</div>'
    if session.brief is None:
        return ""
    return f'<div class="cf cf-brief">{render_reply(session.brief, summary)}</div>'


# =============================================================================
# Event handlers
# Every handler returns a Frame: one value per output component.
# =============================================================================
class Frame(NamedTuple):
    """Values for every output component, in the order build_app wires them."""

    state: Session
    header: str
    script: str
    transcript: str
    message: Any
    memory_filter: Any
    memory_head: str
    memory_table: str
    guard: str
    receipts: str
    brief: str


def render_all(session: Session, *, clear_input: bool = False) -> Frame:
    """Render every panel for ``session``."""
    memories = session.engine.memories()
    summary = session.engine.summary()
    return Frame(
        state=session,
        header=render_header(RUNTIME, fell_back=session.fell_back),
        script=render_script(session.steps_done(), busy=session.pending is not None),
        transcript=render_transcript(session, memories, summary),
        message=gr.update(value="") if clear_input else gr.update(),
        memory_filter=gr.update(choices=filter_choices(memories), value=session.memory_filter),
        memory_head=render_memory_head(memories),
        memory_table=render_memory_table(
            memories, session.engine.histories([m.id for m in memories]), session.memory_filter, session.changed
        ),
        guard=render_guard_panel(session.last_guarded),
        receipts=render_receipts(summary, session.engine.receipts_today()),
        brief=render_brief(session, summary),
    )


def start(session: Session | None) -> Frame:
    """Page load: give this tab its own session on a fresh copy of the seed."""
    return render_all(session or Session.start())


def queue_message(message: str, session: Session | None) -> Frame:
    """Show the visitor's message straight away; the answer follows in ``respond``."""
    session = session or Session.start()
    text = " ".join((message or "").split())
    if not text or session.pending is not None:
        return render_all(session)
    session.pending = text
    return render_all(session, clear_input=True)


def respond(session: Session | None) -> Frame:
    """Answer the pending message and refresh every panel."""
    session = session or Session.start()
    question, session.pending = session.pending, None
    if question is None:
        return render_all(session)
    before = session.snapshot()
    turn = run_safely(question, lambda: session.engine.ask(question))
    session.turns.append(turn)
    session.record(turn, before)
    return render_all(session)


def filter_memories(choice: str, session: Session | None) -> Frame:
    """Apply the sensitivity filter to the memory table."""
    session = session or Session.start()
    session.memory_filter = choice if choice in FILTERS else "all"
    return render_all(session)


def begin_brief(session: Session | None) -> Frame:
    """Show that the brief is being written."""
    session = session or Session.start()
    session.brief_pending = True
    return render_all(session)


def run_brief(session: Session | None) -> Frame:
    """Write the daily brief and refresh the guard and receipts."""
    session = session or Session.start()
    before = session.snapshot()
    session.brief = run_safely("Daily brief", session.engine.brief)
    session.brief_pending = False
    session.record(session.brief, before)
    return render_all(session)


def reset(session: Session | None) -> Frame:
    """Drop this tab's changes and start again from the seed memory."""
    if session is not None:
        session.engine.close()
    return render_all(Session.start())


def close_session(session: Session | None) -> None:
    """Gradio calls this when a tab's state expires or the tab closes."""
    if session is not None:
        session.engine.close()


# =============================================================================
# Layout
# =============================================================================
ASSETS = Path(__file__).resolve().parent / "assets"


def _both(**values: str) -> dict[str, str]:
    """Give theme variables the same value in light and dark mode (the CSS tokens switch)."""
    return {**values, **{f"{name}_dark": value for name, value in values.items()}}


THEME = gr.themes.Base(
    font=[
        gr.themes.GoogleFont("Atkinson Hyperlegible Next", weights=(400, 500, 700, 800)),
        "ui-sans-serif", "system-ui", "sans-serif",
    ],
    font_mono=[gr.themes.GoogleFont("Atkinson Hyperlegible Mono", weights=(400, 600)), "ui-monospace", "monospace"],
).set(
    color_accent="var(--cf-accent)",
    **_both(
        body_background_fill="var(--cf-page)",
        body_text_color="var(--cf-ink)",
        body_text_color_subdued="var(--cf-muted)",
        background_fill_primary="var(--cf-sheet)",
        background_fill_secondary="var(--cf-well)",
        block_background_fill="var(--cf-sheet)",
        block_border_color="var(--cf-rule)",
        border_color_primary="var(--cf-rule)",
        color_accent_soft="var(--cf-accent-soft)",
        input_background_fill="var(--cf-page)",
        input_border_color="var(--cf-rule-strong)",
        input_border_color_focus="var(--cf-accent)",
        button_primary_background_fill="var(--cf-accent)",
        button_primary_background_fill_hover="var(--cf-accent-hover)",
        button_primary_text_color="var(--cf-on-accent)",
        loader_color="var(--cf-accent)",
        link_text_color="var(--cf-accent)",
    ),
)

HEAD = f"<script>{(ASSETS / 'confidant.js').read_text(encoding='utf-8')}</script>"


def build_app() -> gr.Blocks:
    """Assemble the Blocks app. The first paint shows a fresh seed session."""
    preview = Session.start()
    first = render_all(preview)
    choices = filter_choices(preview.engine.memories())
    preview.engine.close()

    with gr.Blocks(
        title="Confidant", theme=THEME, css_paths=[ASSETS / "confidant.css"], head=HEAD,
        fill_width=True, analytics_enabled=False,
    ) as demo:
        state = gr.State(None, time_to_live=3 * 3600, delete_callback=close_session)
        header = gr.HTML(first.header, padding=False, elem_id="cf-header", elem_classes=["cf-bare"])
        with gr.Row(elem_id="cf-main"):
            with gr.Column(elem_id="cf-script-col", min_width=0):
                script = gr.HTML(first.script, padding=False, elem_classes=["cf-bare"])
                reset_button = gr.Button("Start over with fresh memory", size="sm", elem_classes=["cf-button-quiet"])
            with gr.Column(elem_id="cf-chat-col", min_width=0):
                transcript = gr.HTML(first.transcript, padding=False, elem_id="cf-transcript", elem_classes=["cf-bare"])
                with gr.Row(elem_id="cf-composer"):
                    message = gr.Textbox(
                        show_label=False, placeholder="Message Confidant", lines=1, max_lines=4, container=False,
                        elem_id="cf-input",
                        html_attributes=gr.InputHTMLAttributes(autocomplete="off", enterkeyhint="send"),
                    )
                    send = gr.Button("Send", variant="primary", elem_id="cf-send", scale=0)
            with gr.Column(elem_id="cf-inspector-col", min_width=0):
                with gr.Tabs(elem_id="cf-tabs"):
                    with gr.Tab("Memory", elem_id="cf-tab-memory"):
                        memory_head = gr.HTML(first.memory_head, padding=False, elem_classes=["cf-bare"])
                        memory_filter = gr.Radio(
                            choices=choices, value="all", show_label=False, container=False, elem_id="cf-filter"
                        )
                        memory_table = gr.HTML(first.memory_table, padding=False, elem_classes=["cf-bare"])
                    with gr.Tab("Guard", elem_id="cf-tab-guard"):
                        prove_button = gr.Button("Prove it: move my private appointments and compare",
                                                 elem_id="cf-prove", elem_classes=["cf-button-accent"])
                        proof_panel = gr.HTML(PROOF_INTRO, padding=False, elem_classes=["cf-bare"])
                        guard_panel = gr.HTML(first.guard, padding=False, elem_classes=["cf-bare"])
                    with gr.Tab("Receipts", elem_id="cf-tab-receipts"):
                        receipts_panel = gr.HTML(first.receipts, padding=False, elem_classes=["cf-bare"])
                    with gr.Tab("Daily brief", elem_id="cf-tab-brief"):
                        gr.HTML(BRIEF_HEAD, padding=False, elem_classes=["cf-bare"])
                        brief_button = gr.Button("Run daily brief", elem_classes=["cf-button-accent"])
                        brief_panel = gr.HTML(first.brief, padding=False, elem_classes=["cf-bare"])

        outputs = [
            state, header, script, transcript, message, memory_filter,
            memory_head, memory_table, guard_panel, receipts_panel, brief_panel,
        ]  # same order as Frame
        quick = {"show_progress": "hidden", "queue": False}
        work = {"show_progress": "hidden", "concurrency_limit": 4}

        demo.load(start, inputs=[state], outputs=outputs, **quick)
        for trigger in (message.submit, send.click):
            trigger(queue_message, inputs=[message, state], outputs=outputs, **quick).then(
                respond, inputs=[state], outputs=outputs, **work
            )
        memory_filter.input(filter_memories, inputs=[memory_filter, state], outputs=outputs, **quick)
        brief_button.click(begin_brief, inputs=[state], outputs=outputs, **quick).then(
            run_brief, inputs=[state], outputs=outputs, **work
        )
        reset_button.click(reset, inputs=[state], outputs=outputs, **quick)
        prove_button.click(run_proof, inputs=[state], outputs=[proof_panel], **work)
        for trigger in (send.click, message.submit, brief_button.click, reset_button.click):
            trigger(lambda: PROOF_INTRO, outputs=[proof_panel], **quick)  # a new turn: the last proof no longer applies
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch(show_api=False, favicon_path=str(ASSETS / "logo.svg"))
