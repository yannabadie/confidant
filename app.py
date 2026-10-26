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
import re
import uuid
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
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
from confidant.guard import DEFAULT_POLICY, NEBIUS, PLACEHOLDER_RE, Guard
from confidant.llm import ModelSelectionError, TokenFactory, discover_models
from confidant.memory import MemoryStore
from confidant.net import ServiceError
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
    draft: str = ""  # the answer as the cloud model wrote it, placeholders included
    values: Mapping[int, str] = field(default_factory=dict)  # placeholder id -> value filled in here
    citations: tuple[MemoryView, ...] = ()
    shared: tuple[MemoryView, ...] = ()  # sent verbatim with the answer request
    placeholders: tuple[MemoryView, ...] = ()  # sent only as ⟦m7⟧ and a label
    kept_local: tuple[MemoryView, ...] = ()  # withheld from every request of the turn
    checks: tuple[CheckView, ...] = ()
    actions: tuple[str, ...] = ()
    receipts: tuple[ReceiptView, ...] = ()
    mode: str = "replay"
    is_brief: bool = False
    notice: str = ""
    error: str = ""


@dataclass(frozen=True)
class Runtime:
    """Process-wide pieces shared by every session: mode, models, clients and budget."""

    mode: str  # "live" or "replay"
    models: dict[str, str]  # "reason" / "fast" -> model id
    web_search: str  # "Tavily", "recorded results" or "off"
    reason: str  # why the app runs in replay, for the header badge
    llm: Any
    search: Any
    replay: Any
    budget: Any


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
        return Runtime("replay", recorded_models(replay), "recorded results", reason, None, None, replay, budget)

    api_key = env("NEBIUS_API_KEY")
    if not api_key:
        return replay_runtime("No API key on this Space, so recorded Nemotron outputs are replayed.")
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
        cost_usd=r.cost_usd_estimate, mode=r.mode,
    )


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
        draft = turn.draft or turn.answer
        values = {}
        if prepared is not None:
            values = {s.memory.id: s.rendered for s in self.guard.segments(draft, prepared) if s.memory is not None}
        is_brief = turn.intent == "brief"
        return TurnView(
            question=question,
            draft=BRIEF_RECEIPTS_LINE.sub("", draft) if is_brief else draft,
            values=values,
            citations=tuple(_memory(m) for m in turn.citations),
            shared=tuple(_memory(m) for m in prepared.shared) if prepared else (),
            placeholders=tuple(_memory(m) for m in prepared.withheld) if prepared else (),
            kept_local=tuple(_memory(m) for m in report.kept_local),
            checks=tuple(
                CheckView(d.purpose, d.provider, d.allowed, tuple(d.reasons), d.redacted_text if d.allowed else "")
                for d in report.decisions
            ),
            actions=tuple(turn.actions),
            receipts=tuple(_receipt(r) for r in turn.receipts),
            mode=turn.mode,
            is_brief=is_brief,
            notice=turn.notice.replace(REPLAY_NOTICE, "").strip(),
        )


# =============================================================================
# Demo content and per-visitor session state
# =============================================================================
RUNTIME = build_runtime()
PERSONA_NAME = PERSONA.get("name", "Alex Martin")
PERSONA_LINE = f"a fictional {PERSONA.get('role', 'product designer').lower()} in {PERSONA.get('city', 'Lyon')}"

DEMO_SCRIPT: tuple[tuple[str, str], ...] = (
    ("What's on my plate next week?", "Private appointments are filled in on this device"),
    ("My dermatologist appointment moved to Wednesday 4 Nov at 9:00.", "Corrected on this device as version 2"),
    ("Find a 10-day taper plan for my half marathon.", "Web search with nothing private in the query"),
    ("Draft a birthday message for Léa.", "Personal memories may go to the cloud model"),
    ("Remember my bank PIN is 4821.", "Refused: credentials are never stored or sent"),
    ("Run my daily brief.", "The week ahead, plus today's receipts"),
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
        why = "live limit reached, replaying recorded turns" if fell_back else "recorded Nemotron turns"
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
    """The six-step demo script; each step sends its prompt."""
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
        '<p class="cf-script__intro">Six steps, about two minutes. Select a step to send it.</p></div>'
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
    if turn.mode == "replay":
        tags.append('<li><span class="cf-tag" title="Model outputs replayed from data/replay">replay</span></li>')
    return f'<ul class="cf-actions">{"".join(tags)}</ul>' if tags else ""


def render_slip(summary: SummaryView) -> str:
    """Today's receipt totals, printed like a till slip."""
    today = datetime.now(timezone.utc).strftime("%a %d %b %Y")
    calls = f"{summary.calls}" + (f" ({summary.replay_calls} replayed)" if summary.replay_calls else "")
    return (
        f'<div class="cf-slip"><div class="cf-slip__head">Today<span>{today}</span></div>'
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


def render_empty(memories: Sequence[MemoryView], today: date) -> str:
    """First screen: who you are, and late binding shown on a real private memory."""
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
            '<p class="cf-binding__who">You read<small>filled in on this device</small></p>'
            f'<p class="cf-binding__text">Also on your calendar: {filled_html(example.text, example.id)}</p></div>'
            "</figure>"
        )
    return (
        '<div class="cf cf-transcript"><section class="cf-empty">'
        f'<h2 class="cf-empty__title">You are {esc(PERSONA_NAME)} today.</h2>'
        f'<p class="cf-empty__lead">Confidant keeps {len(memories)} memories about {esc(PERSONA_NAME)}, '
        f"{esc(PERSONA_LINE)}, and the demo is set on {today:%A} {today.day} {today:%B %Y}. "
        "Private memories never leave this device: Nemotron sees a placeholder and writes around it, "
        "and the value is filled in here.</p>"
        f"{binding}"
        '<p class="cf-empty__cta">Start with <strong>step 1</strong> of the demo script, or ask your own question.</p>'
        "</section></div>"
    )


def render_transcript(session: Session, memories: Sequence[MemoryView], summary: SummaryView) -> str:
    """The whole conversation, or the empty state before the first message."""
    if not session.turns and not session.pending:
        return render_empty(memories, session.engine.today())
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
    """One outbound request: verdict, destination, reasons and the exact text."""
    if not check.allowed:
        verdict, css = "Blocked", " cf-verdict--blocked"
    elif check.reasons:
        verdict, css = "Redacted", ""
    else:
        verdict, css = "Allowed", ""
    reasons = "".join(f'<span class="cf-check__reasons">{esc(capitalised(r))}</span>' for r in check.reasons)
    payload = ""
    if check.text:
        payload = (
            f'<details class="cf-payload"{" open" if open_text else ""}><summary>Exact text sent</summary>'
            f"{render_payload(check.text, known)}</details>"
        )
    what = f"{PURPOSE_NAMES.get(check.purpose, check.purpose)} to {PROVIDER_NAMES.get(check.provider, check.provider)}"
    return (
        f'<li class="cf-check"><span class="cf-verdict{css}">{verdict}</span>'
        f'<div class="cf-check__body"><span class="cf-check__what">{esc(what)}</span>{reasons}{payload}</div></li>'
    )


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
    kept = render_memlinks(
        turn.kept_local,
        lambda m: (
            f"Sent only as ⟦m{m.id}⟧ and its label, filled in here"
            if m.id in placeholder_ids else "Kept out of every request"
        ),
        "No private memory was involved.",
    )
    known = {m.id for m in turn.kept_local} | placeholder_ids
    main = next((i for i, c in enumerate(turn.checks) if c.purpose in ("answer", "brief") and c.text), -1)
    checks = "".join(render_check(c, known, open_text=i == main) for i, c in enumerate(turn.checks))
    replay_note = (
        '<p class="cf-panel__lead">Replayed turn: these are the exact requests a live run makes. '
        "In replay nothing leaves at all.</p>"
        if turn.mode == "replay" else ""
    )
    compare = ""
    if turn.values:
        compare = (
            '<section class="cf-section"><h4 class="cf-section-title">Late binding</h4><div class="cf-binding">'
            '<div class="cf-binding__row"><p class="cf-binding__who">Nemotron wrote</p>'
            f'<div class="cf-binding__text cf-answer">{render_rich(turn.draft, known=known)}</div></div>'
            '<div class="cf-binding__row"><p class="cf-binding__who">You read</p>'
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
        '<section class="cf-section"><h4 class="cf-section-title">Outbound requests'
        f'<span class="cf-count">{len(turn.checks)}</span></h4>{requests}</section>'
        f"{compare}</div>"
    )


# --- Receipts and brief tabs --------------------------------------------------------
def render_call(r: ReceiptView) -> str:
    """One receipt: what was called, with what, and the payload hash."""
    if r.mode == "replay":
        numbers = "Replayed from a recording: nothing was sent and no tokens were used."
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
        f'<span class="cf-call__time">{esc(r.at[11:19])}</span>{tag}</div>'
        f'<div class="cf-call__provider">{esc(PROVIDER_NAMES.get(r.provider, r.provider))}</div>'
        f'<div class="cf-call__model">{esc(r.model)}</div>'
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
        '<p class="cf-panel__lead">Every call that left this device during this session.</p></div>'
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
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch(show_api=False, favicon_path=str(ASSETS / "logo.svg"))
