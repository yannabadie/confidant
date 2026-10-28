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
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

from .budget import Budget
from .forms import calendar_form_spans
from .guard import (
    NEBIUS, PLACEHOLDER_RE, TAVILY, Guard, GuardDecision, PreparedContext, drop_echoed_labels, placeholder,
)
from .llm import ChatResult, ModelSelectionError, discover_models, estimate_cost, strip_reasoning
from .memory import (
    KINDS, SENSITIVITIES, Memory, MemoryNotFound, MemoryStore, SecretRefused, VersionConflict, utc_timestamp,
)
from .net import ServiceError
from .receipts import ANSWERED, Receipt, ReceiptLog, payload_sha256
from .replay import RecordedTurn, ReplayBook
from .rules import (
    INTENTS,
    Route,
    classify_kind,
    classify_new_fact,
    classify_sensitivity,
    ONGOING_MINUTES,
    end_noted,
    event_date,
    event_days,
    is_open_item,
    item_span,
    item_span_on,
    label_mentioned,
    make_label,
    memory_text_from,
    names_time,
    part_of_day,
    period_asked,
    recurs_on,
    rule_route,
    stem,
    stricter,
    time_window,
    two_days_asked,
    unread_period,
    web_query_from,
)
from .scan import (
    MONTH_WORDS, STOPWORDS, WEEKDAY_WORDS, detect_secret, digit_runs, find_dates, find_secrets, find_values,
    has_value_cues, words,
)
from .search import COST_PER_SEARCH_USD, WebResult, record_results, search_payload

REASON_MAX_TOKENS = 8192  # a reasoning trace, then the answer: live, the brief took up to 2,146; 1,024 cut it off
FAST_MAX_TOKENS = 256  # one short JSON object, with reasoning off (see THINKING)
# Which roles reason. The fast model's job, routing, is one JSON object: with reasoning on, Nemotron Nano spent
# its whole budget thinking and returned no route (scripts/probe_reasoning.py, docs/token-factory-probe.md).
THINKING = {"reason": True, "fast": False}
# A question or request even without its "?": "What's on my plate next week", "Show me my agenda for next week".
# How a question about the user's own time starts when it has no "?": a question word, a request to tell, show,
# list or give ("Tell me what's on my plate next week", "Can you list my plans..."), "remind me what", or an
# auxiliary with I, we, my, our, there or it. Not a request to keep something ("Remind me to...", "Could you
# remember that...", "Do keep in mind that..."): the router can tell those.
_QUESTION_START = re.compile(
    r"^\s*(?:please\s+)?(?:what|what['’]s|whats|when|which|who|how|where|any|anything|show|list|tell|give"
    r"|(?:(?:can|could|would|will)\s+you\s+)?remind\s+me\s+(?:what|when)"
    r"|(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:tell|show|list|give)"
    r"|(?:do|does)\s+you\s+know\s+(?:what|when|if|whether)"
    r"|(?:do|does|did|am|are|is|have|has|can|could|will|would|should|shall)\s+(?:i|we|my|our|there|it)\b)\b",
    re.I,
)
_FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"})
REASON_TEMPERATURE = 0.3
FAST_TEMPERATURE = 0.0
MAX_CONTEXT_MEMORIES = 10
# The private schedule goes whole, a placeholder line per item, so that none is dropped from an answer about its
# own day by a cap shared with the memories picked for the message; this bounds only the prompt's size.
MAX_WITHHELD_ITEMS = 60
BRIEF_HORIZON_DAYS = 7
AVAILABILITY_MAX_DAYS = 14  # an availability check covers at most this many days
BRIEF_QUESTION = "Run my daily brief."
# Said when replay has no recording for a question: what can be checked without a model.
REPLAY_SCOPE = (
    "In replay, only the recorded demo questions have a model answer. What the device recognises without a model "
    "runs as it does live: \"Remember that...\", a correction such as \"... moved to Friday\", \"Am I free...?\", "
    "and refusing a password or PIN."
)
WEB_RESULT_COUNT = 5
TAVILY_MODEL = "tavily-search-basic"

ANSWER_SYSTEM_PROMPT = """\
You are Confidant, a personal assistant that answers from the user's own memories.
Rules:
1. Use the memories provided. Cite every memory you rely on with its tag exactly as written, for example [m12v2].
2. Some memories are withheld for privacy and appear as placeholders such as ⟦m7⟧ with a short label, \
listed after the others with no date. Copy the placeholder verbatim where that information belongs; it is filled in, \
with its date, on the user's device after you answer. A placeholder stands for a whole sentence, so put it on its \
own line or after a colon, and do not say when it happens.
3. Never guess, reconstruct or ask for a withheld value, and add nothing that its label does not say.
4. If the memories hold nothing relevant, say so clearly ("Nothing in memory about that.") and offer only \
general knowledge, labelled as such.
5. Keep answers short: a few sentences or a brief list, no tables (the chat and Telegram do not render them).
6. When asked to draft a message for someone else (a birthday note, an e-mail), write only what the user would \
send. Never mention a gift, a surprise or a plan the user is keeping from that person, and put the citations on \
one line after the message, not inside it."""

ROUTER_SYSTEM_PROMPT = """\
You route messages for Confidant, a personal assistant that knows the user's own life from their memories. \
Reply with exactly one JSON object and no other text:
{"intent": "answer" | "remember" | "correct" | "search" | "brief" | "availability", \
"text": "...", "target": "...", "query": "..."}
- answer, the default: any question their memories could answer, about their own plans, schedule, events, \
people, tastes, training or past choices; also drafting and advice. Examples: "When is my next dentist visit?", \
"What did Sam suggest for the trip?", "Write a thank-you note to my landlord."
- search: only when they ask for outside knowledge that their memories cannot hold, such as a how-to, a recipe \
or public facts. Examples: "How do I fix a flat bike tyre?", "Look up the opening hours of the Louvre." \
"query": a short web query with no names, dates or personal details.
- remember: they tell you something new to keep. "text": that fact as one short standalone sentence.
- correct: they change or fix something already known. "target": a few words naming that memory; \
"text": the full corrected fact as one sentence.
- brief: they ask for their daily brief.
- availability: they ask whether they are free or busy at a given time, e.g. "Am I free on Tuesday at 15:00?".
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
        self, model: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int, temperature: float, purpose: str,
        thinking: bool = True,
    ) -> ChatResult:
        """Run one chat completion; ``thinking=False`` asks the model not to reason first."""
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
class Outbound:
    """One call that left (or, in replay, would leave) the device, exactly as sent: ``payload`` is the
    messages list sent to Nemotron or the query sent to Tavily, and ``sha256`` the hash its receipt keeps.
    Held in memory for the turn only; receipts never store the payload."""

    provider: str
    model: str
    purpose: str
    payload: Any
    sha256: str
    mode: str
    status: str = ANSWERED  # or why this request got no answer: "no answer (HTTP 503)"


@dataclass(frozen=True)
class Turn:
    """The result of one message.

    ``answer`` is the late-bound text for the user and ``draft`` the same text
    with placeholders still in it (safe to pass on); ``citations`` the memories
    the answer cites or fills in, in order of appearance; ``actions`` what
    changed (e.g. "recorded m26v1", "corrected m7 → v2"); ``receipts`` the rows
    logged during the turn; ``mode`` is "live" or "replay"; ``notice`` tells the
    user about replay, budget limits or failed calls. ``local``, when set, is the
    text ``answer`` was bound from instead of the draft: it places private items
    in time (an availability check), so it is for this device only.
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
    outbound: list[Outbound] = field(default_factory=list)
    local: str = ""


def delivery_text(turn: Turn, store: MemoryStore) -> str:
    """A turn as it may be sent off this device (Telegram, another agent): built from the draft, where every
    withheld memory is still a placeholder, and each placeholder shown as its label, never its value."""
    if not turn.draft.strip():
        return "Confidant could not prepare this message."

    def label(match: re.Match[str]) -> str:
        try:
            memory = store.get(int(match.group(1)))
        except MemoryNotFound:
            return "[withheld item]"
        return f"[{memory.sensitivity}: {memory.label or 'withheld'}]"

    lines = [PLACEHOLDER_RE.sub(label, drop_echoed_labels(turn.draft)).strip(), "",
             f"Private items stay on your device; open Confidant to see them. ({turn.guard.summary()}; {turn.mode})"]
    return "\n".join(lines)


@dataclass(frozen=True)
class _Outcome:
    """What an intent handler produced: the draft answer, the context it was written from and, when the
    answer shown on this device must differ from the draft that may be passed on, that device-only text."""

    draft: str
    prepared: PreparedContext
    local: str = ""


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


def masked(message: str, *, keep_kind: bool = True) -> str:
    """``message`` with every date, time, number, e-mail address and name replaced by its kind
    ("[date]", "[number]"…), or removed when ``keep_kind`` is False: enough for a router to choose
    an intent, nothing of the values."""
    spans = [(value.start, value.end, value.kind) for value in find_values(message)]
    spans += [(start, end, "number") for start, end, _ in digit_runs(message)]
    spans += calendar_form_spans(message)  # "half ten", "3 de novembro": whatever private memories hold
    out, cursor = [], 0
    for start, end, kind in sorted(spans, key=lambda span: (span[0], -span[1])):  # longest first at a start
        if start < cursor:
            continue
        out += [message[cursor:start], f"[{kind}]" if keep_kind else ""]
        cursor = end
    out.append(message[cursor:])
    return "".join(out)


CALENDAR_KINDS = ("date", "time", "day")
# Dates and times written as digits in forms the scanners do not read, matched whatever private memories hold: a
# bare digit run is otherwise caught only by the digit signature of a private value, which tells the receiver that
# it equals one ("Remind me what 20261103 holds", "wasn't it first set for 1030?", "on 3-11", "on 3.XI.2026").
_DIGIT_CALENDAR = (
    (re.compile(r"(?<![\d.])(?P<v>(?:19|20)\d\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))(?![\d.])"), "date"),
    (re.compile(r"(?i)(?<![\w.])(?P<v>(?:0?[1-9]|[12]\d|3[01]) ?[./ -] ?(?:xii|xi|ix|x|viii|vii|vi|iv|v|iii|ii|i)"
                r" ?[./ -] ?(?:\d{4}|\d{2}))(?![\w.])"), "date"),
    (re.compile(r"(?i)\b(?:on|by|until|till|from|before|after|since|for) (?P<v>(?:0?[1-9]|[12]\d|3[01])-(?:0?[1-9]|1[0-2]))"
                r"(?![\w/-]|\.\d)"), "date"),
    (re.compile(r"(?i)\b(?:at|for|by|around|until|till|from|before|after|about) (?P<v>(?:[01]?\d|2[0-3])[0-5]\d)\b"
                r"(?!\s*(?:km|kg|m|mi|miles?|steps|people|guests|calories|kcal|euros?|eur|dollars?|usd|pounds?|gbp"
                r"|[€$£%]))"), "time"),
)


_TIME_AFTER_DATE = re.compile(r"(?:, ?| | at )(?P<v>(?:[01]?\d|2[0-3])[0-5]\d)(?![\d:]|[.,]\d)")


def names_calendar_value(message: str) -> bool:
    """True when ``message`` holds a date, a clock time or a day of the month, in a form the scanners read or in
    one of the written forms of any date or time ("half ten", "3 de novembro")."""
    return bool(_calendar_spans(message))


def _calendar_spans(message: str) -> list[tuple[int, int, str]]:
    """The dates, clock times and days of the month of ``message``, as (start, end, kind), in order."""
    spans = [(value.start, value.end, value.kind) for value in find_values(message) if value.kind in CALENDAR_KINDS]
    spans += [(found.start, found.end, "date") for found in find_dates(message)]
    # The written forms of every date and time, not only the private ones: "Anything at half ten?" is masked
    # whether or not a private item is at 10:30 (round 10, tests/test_review_round10.py).
    spans += calendar_form_spans(message)
    spans += [(*match.span("v"), kind) for pattern, kind in _DIGIT_CALENDAR for match in pattern.finditer(message)]
    for start, end, kind in list(spans):  # a time in bare digits right after a date: "confirming 3-11, 1030"
        after = _TIME_AFTER_DATE.match(message, end) if kind == "date" else None
        if after:
            spans.append((*after.span("v"), "time"))
    kept, cursor = [], 0
    for start, end, kind in sorted(spans, key=lambda span: (span[0], -span[1])):  # longest first at a start
        if start >= cursor:
            kept.append((start, end, kind))
            cursor = end
    return kept


def calendar_values(message: str) -> list[tuple[str, str]]:
    """``(kind, text)`` of each date, clock time and day of the month of ``message``, in order."""
    return [(kind, message[start:end]) for start, end, kind in _calendar_spans(message)]


def masked_calendar(message: str) -> str:
    """``message`` with every date, clock time and day of the month replaced by its kind ("[date]", "[time]",
    "[day]"), whatever they are: what Nebius gets of the dates a user typed, so that which ones equal a private
    value cannot show."""
    out, cursor = [], 0
    for start, end, kind in _calendar_spans(message):
        out += [message[cursor:start], f"[{kind}]"]
        cursor = end
    out.append(message[cursor:])
    return "".join(out)


def _calendar_restored(text: str, typed: Sequence[tuple[str, str]]) -> str:
    """``text`` with each "[date]", "[time]" or "[day]" the model wrote replaced by what the user typed, in order
    for each kind (the last one repeats). For this device only: the draft that may leave keeps the masks."""
    if not typed:
        return text
    by_kind: dict[str, list[str]] = {}
    for kind, value in typed:
        by_kind.setdefault(kind, []).append(value)
    used: dict[str, int] = {}

    def back(match: re.Match[str]) -> str:
        kind = match.group(1)
        values = by_kind.get(kind)
        if not values:
            return match.group()
        index = used.get(kind, 0)
        used[kind] = index + 1
        return values[min(index, len(values) - 1)]

    return re.sub(r"\[(date|time|day)\]", back, text)


# Day and month names, which masked() leaves alone: in a private message they can say when a private item is.
# Those that are also everyday words ("may", "march", "sun", "sat", "wed", "sept") count only when capitalised
# inside a sentence: "in May", not "May I ask" or "you may".
_AMBIGUOUS_CALENDAR = frozenset({"may", "march", "mar", "mars", "sun", "sat", "wed", "mon", "august", "sept"})
_LETTERS = re.compile(r"[^\W\d_]+")


def masked_private(message: str) -> str:
    """:func:`masked`, and day and month names too, whole or abbreviated, in English, French, Spanish, Italian
    or German: what Nebius gets of a message the rules find private. Names are masked as masked() reads them,
    capitalised ("Sophie", not "sophie")."""
    text = masked(message)

    def swap(match: re.Match[str]) -> str:
        word = match.group()
        key = word.lower()
        if key not in WEEKDAY_WORDS and key not in MONTH_WORDS:
            return word
        if key in _AMBIGUOUS_CALENDAR:
            before = text[:match.start()].rstrip(" \t\"'“‘([")
            if not word[0].isupper() or not before or before[-1] in ".!?:;\n":
                return word
        return "[day]" if key in WEEKDAY_WORDS else "[month]"

    return _LETTERS.sub(swap, text)


# Words a correction uses to say what changed; they bring no new content of their own.
_CHANGE_WORDS = frozenset({
    "moved", "move", "moves", "changed", "change", "changes", "rescheduled", "reschedule", "postponed", "cancelled",
    "canceled", "brought", "forward", "pushed", "earlier", "later", "now", "instead", "actually", "correction",
    "update", "updated", "new", "not", "longer", "anymore", "rather", "than", "same", "next", "last", "this",
})


def added_words(old: str, new: str) -> list[str]:
    """The content words of ``new`` that ``old`` does not hold: no stop words, words saying what changed, or
    dates, times and numbers, however written."""
    known = set(words(old))
    return [word for word in dict.fromkeys(words(new))
            if word not in known and len(word) >= 3 and word not in STOPWORDS and word not in _CHANGE_WORDS
            and not has_value_cues(word)]


def correction_level(target: Memory, text: str, *, private: bool = False) -> tuple[str, str]:
    """``(sensitivity, label)`` of a correction of ``target`` with ``text``, chat or MCP alike. A correction never
    makes a memory less private; one whose words the local rules find private (a health, money, legal,
    job-search or intimate topic, a secrecy cue) makes it private, with a category label: its label travels
    wherever its value does not. The words it adds are judged like a new fact, private unless clearly everyday:
    the old memory's everyday words ("dinner with friends") cannot vouch for them ("brunch with my sponsor").
    The memory's own level and label stand otherwise ("the review moved to Thursday" keeps a public meeting
    public)."""
    sensitivity = stricter(target.sensitivity, "private" if private else classify_sensitivity(text))
    added = added_words(target.text, text)
    if added:
        sensitivity = stricter(sensitivity, classify_new_fact(" ".join(added)))
    label = target.label if sensitivity == target.sensitivity else make_label(text, sensitivity)
    return sensitivity, label


def _stated_text(route: Route, message: str) -> str:
    """The text to store for a remember or correct route: the route's text when the device
    chose the route, the user's own words when a cloud router did (it decides the intent,
    never the content of a memory)."""
    if route.source == "model":
        return memory_text_from(message)
    return route.text or memory_text_from(message)


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


_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_PLACEHOLDER_LINE = re.compile(r"⟦\s*m\d+")


_HEADING = re.compile(r"^\s*(?:#+\s+.*|\*\*[^*]+\*\*\s*:?|[^\s].{0,58}:)\s*$")


def _lone_placeholder(line: str) -> int | None:
    """The id of the one withheld item a line stands for ("- ⟦m7⟧ (private: …)"), else None."""
    ids = {int(match.group(1) or match.group(2)) for match in _CITATION.finditer(line)}
    held = {int(match.group(1)) for match in PLACEHOLDER_RE.finditer(line)}
    return next(iter(ids)) if len(ids) == 1 and held == ids else None


def _without_items(text: str, drop: Callable[[int], bool]) -> str:
    """``text`` without the lines that stand for one withheld item ``drop`` selects: one that is over ("- ⟦m7⟧"
    once 3 Nov has passed), or outside the period asked. A heading whose lines all went goes too; a sentence
    that led into them says that none falls in the period asked.

    For this device only: Nebius gets every private item whatever the period and whatever the day, so that
    none is dated by being left out, and the draft that may leave keeps them all."""
    lines = text.split("\n")
    gone = [(memory_id := _lone_placeholder(line)) is not None and drop(memory_id) for line in lines]
    if not any(gone):
        return text
    out: list[str] = []
    for index, line in enumerate(lines):
        if gone[index]:
            continue
        block = []
        for later in range(index + 1, len(lines)):
            if not lines[later].strip() or (_HEADING.match(lines[later]) and not gone[later]):
                break
            block.append(later)
        if block and all(gone[later] for later in block) and line.rstrip().endswith((":", "**")):
            if _HEADING.match(line):
                continue
            line = line.rstrip()[:-1] + ": none falls in the period asked."
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n") if text.strip() else text


def _merged_withheld(text: str, date_key: Callable[[int], tuple[date, int] | None]) -> str:
    """``text`` (a brief) with its withheld items moved into its list of dated events, in date order, and the
    heading they were listed under taken out once empty: the model lists them apart, since they carry no date.
    Only when there is such a list and every moved item has a date. For this device only, like
    :func:`_in_date_order`."""
    lines = text.split("\n")
    held = [index for index, line in enumerate(lines)
            if (memory_id := _lone_placeholder(line)) is not None and date_key(memory_id) is not None]
    run: list[int] = []
    for index, line in enumerate(lines):
        refs = {int(match.group(1) or match.group(2)) for match in _CITATION.finditer(line)}
        if _LIST_ITEM.match(line) and len(refs) == 1 and index not in held and date_key(next(iter(refs))) is not None:
            if run and index != run[-1] + 1:
                break
            run.append(index)
        elif run:
            break
    if not held or not run or any(run[0] <= index <= run[-1] for index in held):
        return text
    bullet = _LIST_ITEM.match(lines[run[0]]).group(0)  # type: ignore[union-attr]

    def key(line: str) -> tuple[date, int]:
        refs = [int(match.group(1) or match.group(2)) for match in _CITATION.finditer(line)]
        return date_key(refs[0]) or (date.max, 0)

    merged = sorted([lines[index] for index in run] + [bullet + _LIST_ITEM.sub("", lines[index]).strip()
                                                       for index in held], key=key)
    emptied = set()
    for index, line in enumerate(lines):
        if _HEADING.match(line) and index not in held:
            block = []
            for later in range(index + 1, len(lines)):
                if not lines[later].strip() or _HEADING.match(lines[later]):
                    break
                block.append(later)
            if block and all(later in held for later in block):
                emptied.add(index)
    out: list[str] = []
    for index, line in enumerate(lines):
        if index == run[0]:
            out.extend(merged)
        elif index in held or index in emptied or run[0] < index <= run[-1]:
            continue
        else:
            out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def _in_date_order(text: str, date_key: Callable[[int], tuple[date, int] | None]) -> str:
    """``text`` with each run of items sorted by date, when every item of the run names one dated memory
    (by citation or placeholder) and the dates differ; other lines and runs are left as written. An item is a
    list item, or a plain line naming one memory in a run of such lines that holds a withheld item (models
    also write a week as plain lines, one per item).

    For this device only: the model lists withheld items apart, since they carry no date, and the device, which
    knows their dates, puts them back in order. The draft that may leave keeps the model's order."""
    out: list[str] = []
    run: list[str] = []
    kind = ""

    def refs(line: str) -> set[int]:
        return {int(match.group(1) or match.group(2)) for match in _CITATION.finditer(line)}

    def flush() -> None:
        keys = [date_key(next(iter(ids))) if len(ids := refs(line)) == 1 else None for line in run]
        sortable = len(run) > 1 and all(key is not None for key in keys) and len({key[0] for key in keys}) > 1
        if kind == "plain":
            sortable = sortable and any(_PLACEHOLDER_LINE.search(line) for line in run)
        if sortable:
            out.extend(line for _, _, line in sorted(zip(keys, range(len(run)), run)))
        else:
            out.extend(run)
        run.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        line_kind = ("list" if _LIST_ITEM.match(line)
                     else "plain" if stripped and not stripped.endswith(":") and len(refs(line)) == 1 else "")
        if line_kind != kind or not line_kind:
            flush()
            kind = line_kind
        if line_kind:
            run.append(line)
        else:
            out.append(line)
    flush()
    return "\n".join(out)


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


def _period(window: tuple[date, date], today: date) -> str:
    """A period by its start and length, never its end date: "the 5 days from today",
    "Wednesday 4 November 2026", "the 7 days from Monday 2 November 2026"."""
    first, last = window
    days = (last - first).days + 1
    start = "today" if first == today else _long_date(first)
    return start if days == 1 else f"the {days} days from {start}"


def _days(window: tuple[date, date]) -> str:
    """How many days a period holds, in words a note can carry without its dates: "day", "3 days"."""
    days = (window[1] - window[0]).days + 1
    return "day" if days == 1 else f"{days} days"


def _capitalised(text: str) -> str:
    return text[:1].upper() + text[1:]


def _short_date(day: date) -> str:
    """A date in a list, e.g. "Tue 3 Nov"."""
    return f"{day:%a} {day.day} {day:%b}"


def _split_by_part(
    items: Sequence[Memory], part: tuple[str, int, int] | None, day: date, today: date
) -> tuple[list[Memory], ...]:
    """A day's items in the part of the day asked about (all of them for a whole day); those that started
    earlier with no end noted, so may still be going on; those with no time noted; and the others."""
    if part is None:
        return list(items), [], [], []
    _, start, end = part
    inside: list[Memory] = []
    ongoing: list[Memory] = []
    untimed: list[Memory] = []
    other: list[Memory] = []
    for memory in items:
        span = item_span_on(memory, day, today)
        if span is None:
            untimed.append(memory)
        elif span[0] < end and start < span[1]:
            inside.append(memory)
        elif not end_noted(span) and span[0] < start <= span[0] + ONGOING_MINUTES:
            ongoing.append(memory)
        else:
            other.append(memory)
    return inside, ongoing, untimed, other


def _availability_texts(
    heading: str, schedule: Mapping[date, list[Memory]], part: tuple[str, int, int] | None,
    prepared: PreparedContext, *, single: bool, today: date, routed_by_cloud: bool = False,
) -> tuple[str, str]:
    """The answer to an availability question, for this device only (private items appear as placeholders,
    filled in here, next to their day), and the draft that may be passed on (shared items only)."""
    withheld = set(prepared.withheld_ids)

    def shown(memory: Memory) -> str:
        return placeholder(memory.id) if memory.id in withheld else f"{memory.text} [{memory.cite()}]"

    local: list[str] = []
    passed_on: list[str] = []
    if single:
        day, items = next(iter(schedule.items()))
        inside, ongoing, untimed, other = _split_by_part(items, part, day, today)
        if inside:
            local.append(f"{heading}: not free.")
        elif ongoing or untimed:
            reasons = [f"{len(ongoing)} item(s) started earlier that day with no end time noted"] if ongoing else []
            reasons += [f"{len(untimed)} item(s) that day have no time noted"] if untimed else []
            local.append(f"{heading}: maybe. Nothing is noted at that time, but {' and '.join(reasons)}.")
        else:
            local.append(f"{heading}: free, nothing in memory.")
        local += [f"- {shown(memory)}" for memory in inside]
        if ongoing:
            local += ["Started earlier, no end time noted:", *(f"- {shown(memory)}" for memory in ongoing)]
        if untimed and part:
            local += ["No time noted:", *(f"- {shown(memory)}" for memory in untimed)]
        if other:
            local += ["Also that day:", *(f"- {shown(memory)}" for memory in other)]
        passed_on = [f"- {shown(memory)}" for memory in inside + ongoing + untimed if memory.id not in withheld]
    else:
        free: list[date] = []
        for day, items in schedule.items():
            inside, ongoing, untimed, _ = _split_by_part(items, part, day, today)
            if not inside and not ongoing and not untimed:
                free.append(day)
                local.append(f"- {_short_date(day)}: free")
            for memory in inside + ongoing + untimed:
                note = (" (started earlier, no end time noted)" if memory in ongoing
                        else " (no time noted)" if memory in untimed and part else "")
                local.append(f"- {_short_date(day)}: {shown(memory)}{note}")
                if memory.id not in withheld:
                    passed_on.append(f"- {_short_date(day)}: {shown(memory)}{note}")
        if len(free) == len(schedule):
            verdict = f"{heading}: free every day, nothing in memory."
        elif not free:
            verdict = f"{heading}: no free day."
        else:
            verdict = f"{heading}: free on {len(free)} of {len(schedule)} days ({', '.join(map(_short_date, free))})."
        local.insert(0, verdict)
    local.append(
        "Checked on this device against the real dates, private items included. "
        + ("Only the routing request left, with its values masked." if routed_by_cloud else "Nothing was sent.")
    )
    lead = f"{heading}, shared items in memory:" if passed_on else f"{heading}: no shared item in memory."
    draft = [lead, *passed_on,
             "Private items are left out of this message: whether one falls in this period shows on your device only."]
    return "\n".join(local), "\n".join(draft)


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
        self.web_search = ""  # live: "ran" or "off", kept with the recording; "" when no web query was made
        # a live answer or brief call failed or was cut off: answered locally, never recorded (a failed route
        # is: the rules route it, in replay as they did live)
        self.failed = False
        self.outbound: list[Outbound] = []
        self.opening_notices = len(notices)  # about the mode (replay, budget, models), set before any call
        self.request_differs = False  # replay: the question was recorded, but for another request
        self.typed_calendar: list[tuple[str, str]] = []  # (kind, text) of the dates and times masked, in order
        self.agent = False  # the message comes from another agent: its words never select a private memory

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
        header: str = "",
        trailer: str = "",
    ) -> str | None:
        """Ask the ``role`` model (live) or read its recorded output (replay).

        The user message passes the guard first and goes out redacted. ``header`` is
        text Confidant wrote from the clock and the request alone (today's date, the
        period asked); it is sent as is, so redacting it cannot hint that a private
        value shares its date. ``trailer`` follows the checked message as is: another
        agent's request, masked by kind and checked for credentials only, since
        redacting the words that match a private value would confirm them to the agent.
        Returns None when the message is blocked or no output is available, so callers
        fall back.
        """
        assistant = self.assistant
        decision = assistant.guard.check_outbound(user, NEBIUS, assistant.store.all_versions(), purpose=purpose)
        self.decisions.append(decision)
        sent = prepared.shared_ids if prepared else []
        withheld = sorted(set((prepared.withheld_ids if prepared else []) + decision.withheld_ids))
        self.withheld_ids.update(withheld)
        if not decision.allowed:
            self.notices.append(f"The guard blocked the {purpose} request, so nothing was sent.")
            return None
        self.sent_ids.update(sent)
        content = (f"{header}\n{decision.redacted_text}" if header else decision.redacted_text) + trailer
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        if self.mode == "replay":
            if self.recorded is None or purpose not in self.recorded.outputs:
                return None
            model = self.recorded.models.get(role, "replay")
            expected = self.recorded.requests.get(purpose)
            if expected and payload_sha256(messages) != expected:  # not the request the recording answered
                self.request_differs = True
                self._log(NEBIUS, model, purpose, messages, sent, withheld,
                          status="no answer (request differs from the recording)")
                self.notices.append(
                    f"This {purpose} request differs from the one recorded live (a memory changed since the "
                    "recording), so no model answered it: replay answers only the exact request it recorded."
                )
                return None
            self._log(NEBIUS, model, purpose, messages, sent, withheld)
            return strip_reasoning(self.recorded.outputs[purpose])
        assert assistant.llm is not None and assistant.models is not None  # guaranteed in live mode
        try:
            result = assistant.llm.chat(
                assistant.models[role], messages, max_tokens=max_tokens, temperature=temperature, purpose=purpose,
                thinking=THINKING[role],
            )
        except ServiceError as exc:
            self._log_unanswered(NEBIUS, assistant.models[role], purpose, messages, sent, withheld, exc.unanswered)
            self.notices.append(f"The live {purpose} call failed ({exc}); a local fallback was used.")
            self.failed = self.failed or purpose != "route"
            return None
        self._log_unanswered(NEBIUS, assistant.models[role], purpose, messages, sent, withheld, result.unanswered)
        self.live_calls += 1
        self.tokens += result.prompt_tokens + result.completion_tokens
        self._log(
            NEBIUS, result.model, purpose, messages, sent, withheld, prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens, latency_ms=result.latency_ms,
            cost=estimate_cost(result.model, result.prompt_tokens, result.completion_tokens),
        )
        if result.truncated:  # unfinished: the text may be the model's reasoning, never an answer
            self.notices.append(
                f"The {purpose} model reached its {max_tokens}-token limit before finishing; a local fallback was used."
            )
            self.failed = self.failed or purpose != "route"
            return None
        if not result.text:
            self.notices.append(f"The {purpose} model returned no answer text; a local fallback was used.")
            self.failed = self.failed or purpose != "route"
            return None
        self.outputs[purpose] = result.text
        return result.text

    def web(self, query: str) -> list[WebResult]:
        """Run a guarded web search (live) or return the recorded results (replay). Without a Tavily key
        the query is still guarded, so the Guard tab shows what a search would send; nothing leaves."""
        assistant = self.assistant
        decision = assistant.guard.check_outbound(
            query, TAVILY, assistant.store.all_versions(), purpose="web search", strict=True
        )
        self.decisions.append(decision)
        self.withheld_ids.update(decision.withheld_ids)
        if not decision.allowed:
            self.notices.append("The guard blocked the web query: it would have revealed private details.")
            return []
        if self.mode != "replay" and assistant.search is None:
            self.notices.append(
                f"Web search is off (no TAVILY_API_KEY), so nothing was searched; the guarded query would have been "
                f"\u201c{decision.redacted_text}\u201d. The answer uses memory only."
            )
            self.decisions[-1] = replace(decision, not_sent="web search is off, so this query was not sent")
            self.web_search = "off"
            return []
        payload = search_payload(decision.redacted_text, WEB_RESULT_COUNT)
        if self.mode == "replay":
            recorded = self.recorded
            results = list(recorded.web_results) if recorded else []
            if results or (recorded is not None and recorded.web_search == "ran"):
                self._log(TAVILY, TAVILY_MODEL, "web_search", payload, [], decision.withheld_ids)
            elif recorded is not None and recorded.web_search == "off":
                self.notices.append(
                    "Web search was off when this turn was recorded, so nothing was searched; the guarded query "
                    f"would have been \u201c{decision.redacted_text}\u201d. The answer uses memory only."
                )
                self.decisions[-1] = replace(decision, not_sent="web search was off when this turn was recorded")
            else:
                self.notices.append("Replay has no recorded web results for this request.")
                self.decisions[-1] = replace(decision, not_sent="the recording holds no web search for this turn")
            return results
        started = time.perf_counter()
        try:
            results = assistant.search.search(decision.redacted_text, max_results=WEB_RESULT_COUNT)
        except ServiceError as exc:  # the query left the device: it is not marked as not sent
            self._log_unanswered(TAVILY, TAVILY_MODEL, "web_search", payload, [], decision.withheld_ids, exc.unanswered)
            self.notices.append(f"Web search failed ({exc}); the answer uses memory only.")
            self.failed = True
            return []
        self._log_unanswered(TAVILY, TAVILY_MODEL, "web_search", payload, [], decision.withheld_ids,
                             getattr(results, "unanswered", ()))
        self.live_calls += 1
        self.web_search = "ran"
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
        status: str = ANSWERED,
    ) -> None:
        """Write one receipt for this turn's mode, and keep the payload with it for this turn only."""
        receipt = self.assistant.receipts.record(
            provider=provider, model=model, purpose=purpose, payload=payload, mode=self.mode,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, latency_ms=latency_ms,
            sent_memory_ids=sent, withheld_memory_ids=withheld, cost_usd_estimate=cost,
            session_id=self.session_id, status=status,
        )
        self.receipts.append(receipt)
        self.outbound.append(Outbound(provider, model, purpose, payload, receipt.payload_sha256, self.mode, status))

    def _log_unanswered(
        self, provider: str, model: str, purpose: str, payload: object, sent: Sequence[int], withheld: Sequence[int],
        reasons: Sequence[str],
    ) -> None:
        """One receipt per request that left and got no answer (a failed call, or the attempts before a retry
        that worked): it carried the same payload, so it is logged like any other call, with no tokens."""
        for reason in reasons:
            self._log(provider, model, purpose, payload, sent, withheld, status=f"no answer ({reason})")
        self.live_calls += len(reasons)


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

    def handle(self, message: str, *, session_id: str, intent: str | None = None, agent: bool = False) -> Turn:
        """Answer one user message end to end.

        ``intent`` (one of INTENTS) skips routing, e.g. "answer" for read-only callers. ``agent`` marks a message
        written by another agent (over MCP) rather than the user: its words never select a private memory, since
        the agent could learn from which ones do what the memory says.
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
        exchange.agent = agent
        if intent is None:
            route, private = self._route(exchange, message)
        else:
            route, private = Route(intent, source="caller"), False
        handlers = {
            "answer": self._answer, "remember": self._remember, "correct": self._correct,
            "search": self._search, "brief": self._brief, "availability": self._availability,
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
        """Late-bind the draft (or the device-only text), settle the budget and recorder, and build the Turn."""
        prepared = outcome.prepared
        local = outcome.local
        if not local and route.intent == "answer":
            ordered = _in_date_order(outcome.draft, self._date_key)  # device only: the order dates the items
            local = ordered if ordered != outcome.draft else ""
        if not local and route.intent == "brief":
            merged = _merged_withheld(outcome.draft, self._date_key)  # device only, likewise
            local = merged if merged != outcome.draft else ""
        if route.intent in ("answer", "brief"):  # device only: which items are over, or in the period, dates them
            today = self.today()
            period = (today, today + timedelta(days=BRIEF_HORIZON_DAYS)) if route.intent == "brief" \
                else period_asked(message, today)
            current = local or outcome.draft
            trimmed = _without_items(current, lambda memory_id: self._is_over(memory_id)
                                     or (period is not None and self._is_outside(memory_id, period)))
            if current.strip() and not trimmed.strip():  # every line stood for an item outside the period
                trimmed = "Nothing in memory falls in the period asked."
            local = trimmed if trimmed != current else local
        if exchange.typed_calendar:  # device only: the dates and times the user typed, masked for Nebius
            current = local or outcome.draft
            restored = _calendar_restored(current, exchange.typed_calendar)
            local = restored if restored != current else local
        shown = local or outcome.draft
        segments = self.guard.segments(shown, prepared)
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
        if live and self.recorder is not None and exchange.outputs and not exchange.failed:
            self.recorder.add(RecordedTurn(
                question=message, outputs=dict(exchange.outputs), web_results=list(exchange.web_results),
                models=dict(self.models or {}), recorded_at=utc_timestamp(), web_search=exchange.web_search,
                requests={call.purpose: call.sha256 for call in exchange.outbound
                          if call.provider == NEBIUS and call.status == ANSWERED and call.purpose in exchange.outputs},
            ))
        report = GuardReport(
            prepared=prepared, decisions=exchange.decisions, kept_local=kept_local,
            late_bound_ids=late_bound, unresolved=unresolved, sent_to_cloud=live,
        )
        return Turn(
            answer=self.guard.late_bind(shown, prepared), citations=cited_memories(shown, prepared),
            guard=report, actions=exchange.actions, receipts=exchange.receipts, mode=exchange.mode,
            intent=route.intent, draft=outcome.draft, notice=" ".join(exchange.notices),
            outbound=list(exchange.outbound), local=local,
        )

    def _date_key(self, memory_id: int) -> tuple[date, int] | None:
        """When a memory happens (its date, then its start minute), for ordering on this device; None when it
        is not a current memory with a date."""
        try:
            memory = self.store.get(memory_id)
        except MemoryNotFound:
            return None
        try:  # ordering is a convenience: a text the date rules cannot read leaves the model's order
            day = event_date(memory, self.today()) if memory.status == "current" else None
            return None if day is None else (day, (item_span(memory) or (24 * 60, 0))[0])
        except (ValueError, OverflowError):
            return None

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
        local = rule_route(message)
        if local.intent != "answer" or has_value_cues(message) or self._asks_about_own_period(message):
            # the rules already know; or the message may carry a date, time or amount in some form the
            # scanners do not read; or it asks about a period of the user's own time: in each case, no need
            # to show it to a router
            return local, False
        raw = exchange.complete("route", "fast", ROUTER_SYSTEM_PROMPT, masked(message))
        route = parse_route(raw) if raw is not None else None
        if route is None:
            if raw is not None:
                exchange.actions.append("router reply was not valid JSON; used keyword rules")
            route = rule_route(message)
        return route, False

    def _asks_about_own_period(self, message: str) -> bool:
        """A question, in the first person, that names a period of time ("What's on my plate next week?").

        It asks about the user's own schedule, which memory answers; live, with the final router prompt, the
        router sent "What's on my plate next week?" to a web search in five calls out of seven. A question
        about the world ("Is there a public holiday next week?") or a statement naming a period ("Next week
        I'm in Paris") still goes to the router: the first may need the web, the second may be news to keep.
        """
        return (
            (message.rstrip().endswith("?") or _QUESTION_START.match(message) is not None)
            and any(word in _FIRST_PERSON for word in words(message))
            and period_asked(message, self.today()) is not None
        )

    def _private_signals(self, message: str) -> tuple[list[str], list[int]]:
        """Why a message must stay local: private values in it, private labels it
        names, or a private topic according to the local rules."""
        current = self.store.all_current()
        # Its dates and times are left out: that one equals a private item's would otherwise change how the
        # message is handled (routed, masked, stored), which Nebius could see. They are masked alike instead.
        decision = self.guard.check_outbound(masked_calendar(message), NEBIUS, self.store.all_versions(),
                                             purpose="route", strict=True)
        reasons, ids = list(decision.reasons), list(decision.withheld_ids)
        for memory in current:
            named = label_mentioned(memory.label, message)
            if named and memory.id not in ids and not self.guard.may_send(memory, NEBIUS):
                reasons.append(f"mentions {memory.cite()} ({memory.sensitivity}: {memory.label})")
                ids.append(memory.id)
        if not reasons and classify_sensitivity(message) == "private":
            reasons.append("looks private to the local rules (health, money, legal, job search or a code)")
        return reasons, ids

    # --- intents ---------------------------------------------------------------

    def _answer(self, exchange: _Exchange, message: str, route: Route, private: bool) -> _Outcome:
        """Answer from memory with the reasoning model."""
        today = self.today()
        window = time_window(message, today)
        # A date or clock time the user typed is masked, and the period note says only that the request names
        # one, whether or not a private item is then: masking the value and dropping the note only when it equals
        # a private value would tell Nebius so (the Guard tab's "Prove it" check, confidant/proof.py, moves every
        # private appointment and compares the requests). The shared events of that period are still listed.
        calendar = not exchange.agent and not private and names_calendar_value(message)
        # A period read from the user's private message could date a private item. Not for another agent's
        # question: it chose the period, and dropping the note only when its date matched a private item
        # would tell it so.
        if window and not exchange.agent and not calendar and (
            private or self.guard.check_outbound(message, NEBIUS, self.store.all_versions(), strict=True).withheld_ids
        ):
            window = None
        if window and calendar:
            notes = [f"Period asked about: the {_days(window)} the request names (masked here); every shared event "
                     "of it is listed below."]
        else:
            notes = [f"Period asked about: {_period(window, today)}; every shared event of it is listed below."] if window else []
        return self._compose(exchange, "answer", message, self._retrieve(message, window, agent=exchange.agent), notes,
                             mask=private, calendar=calendar)

    def _remember(self, exchange: _Exchange, message: str, route: Route, private: bool) -> _Outcome:
        """Store what the user stated, in the user's words when a cloud router chose this intent,
        classified on the device."""
        text = _stated_text(route, message)
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
        if not self.guard.may_send(memory, NEBIUS):
            exchange.withheld_ids.add(memory.id)
        lead = f"Saved as [{memory.cite()}] ({memory.sensitivity} {memory.kind}, “{memory.label}”): "
        return self._confirm(memory, lead)

    def _correct(self, exchange: _Exchange, message: str, route: Route, private: bool) -> _Outcome:
        """Add a new version of the memory the message corrects (or record it if none matches)."""
        target = self._find_target(route.target, message, from_model=route.source == "model")
        if target is None:
            exchange.actions.append("no existing memory matched the correction; recorded it as new")
            return self._remember(exchange, message, route, private)
        text = _stated_text(route, message)
        sensitivity, label = correction_level(target, text, private=private)
        try:
            updated = self.store.correct(
                target.id, text, author="user", source_ref=exchange.session_id, expected_version=target.version,
                sensitivity=sensitivity, label=label,
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
        if not self.guard.may_send(updated, NEBIUS):
            exchange.withheld_ids.add(updated.id)  # kept on the device: counted in the turn's guard summary
        return self._confirm(updated, f"Updated [{updated.cite()}] (“{updated.label}”, was v{target.version}): ")

    def _search(self, exchange: _Exchange, message: str, route: Route, private: bool) -> _Outcome:
        """Search the web with a guarded query, record the results, then answer from them.
        The query is built from the user's words: a cloud router only saw them masked. A message the local
        rules find private is not searched at all: the guard removes stored values from a query, not the
        private topic the user typed ("tips for my final-round interview on Thursday")."""
        if private:
            exchange.actions.append("web search skipped: the message looks private")
            exchange.notices.append(
                "This message looks private, so nothing was searched on the web. Nemotron answers from memory, and "
                "gets the message with its dates, times, amounts, capitalised names, days and months masked."
            )
            return self._answer(exchange, message, route, private)
        if route.source == "model":  # the user did not ask for the web: send only words, none of the values typed
            query = " ".join(masked(web_query_from(message), keep_kind=False).split())
        else:
            query = route.query or web_query_from(message)
        results = exchange.web(query)
        found = record_results(self.store, results)
        if found:
            exchange.actions.append("recorded web sources " + ", ".join(f"{m.cite()} ({m.author})" for m in found))
        found_ids = {memory.id for memory in found}
        related = [m for m in self._retrieve(message, time_window(message, self.today())) if m.id not in found_ids]
        return self._compose(exchange, "answer", message, self._capped(found + related),
                             calendar=not exchange.agent and names_calendar_value(message))

    def _brief(self, exchange: _Exchange, message: str, route: Route, private: bool) -> _Outcome:
        """Events of the coming week, open to-dos and today's routines, plus today's receipts."""
        if exchange.mode == "replay" and exchange.recorded is None and self.replay:
            exchange.recorded = self.replay.lookup(BRIEF_QUESTION)
        today = self.today()
        events = [
            memory for memory in self._events_between(today, today + timedelta(days=BRIEF_HORIZON_DAYS))
            if self.guard.may_send(memory, NEBIUS)
        ] + self._private_schedule()
        event_ids = {memory.id for memory in events}
        # Open items and today's routines are picked by their words and weekday: shared memories only. Private
        # events and routines are all in the list above, whatever their date or weekday, so none is dated.
        extras = [
            memory for memory in self.store.all_current()
            if memory.id not in event_ids and self.guard.may_send(memory, NEBIUS)
            and (is_open_item(memory) or recurs_on(memory, today))
        ]
        request = (
            f"Write my daily brief: the events of the next {BRIEF_HORIZON_DAYS} days, soonest first, then the "
            "withheld ones; my open to-do items; and one practical suggestion for today."
        )
        outcome = self._compose(exchange, "brief", request, events + extras)
        summary = self.receipts.today_summary(exchange.session_id).line()
        return _Outcome(f"{outcome.draft}\n\n{summary}", outcome.prepared)

    def _availability(self, exchange: _Exchange, message: str, route: Route, private: bool) -> _Outcome:
        """Whether a period is free, checked on this device against the real dates, private items included.
        No cloud model may learn when a private item happens, so this intent makes no outbound call. The
        answer shown here places every item in time; the draft, which may be passed on, lists the shared
        items only and says the same about private ones whatever the period holds."""
        today = self.today()
        routed_by_cloud = bool(exchange.receipts)  # a cloud router chose this intent (it saw the masked message)
        del exchange.notices[:exchange.opening_notices]  # replay and budget notices: no answer call is made
        exchange.actions.append(
            "checked availability on this device" + ("" if routed_by_cloud else ": nothing sent")
        )
        pool = self._private_schedule()  # every private item, whatever the period: the context dates none
        exchange.withheld_ids.update(memory.id for memory in pool)
        period = period_asked(message, today)
        if period is None and unread_period(message):
            return _Outcome(
                "Which day should I check? For example: Am I free on Tuesday 3 November? I check up to "
                f"{AVAILABILITY_MAX_DAYS} days at a time.", self.guard.prepare_context(pool, NEBIUS),
            )
        first, last = period or (today, today)
        last = min(last, first + timedelta(days=AVAILABILITY_MAX_DAYS - 1))
        part = part_of_day(message)
        days = [first + timedelta(days=offset) for offset in range((last - first).days + 1)]
        heading = _capitalised(_period((first, last), today))
        pair = two_days_asked(message, today)
        if pair and (pair[1] - pair[0]).days > 1:  # "on 2 Nov and 30 Nov": those two days, not the 29 between
            days = list(pair)
            heading = f"{_capitalised(_long_date(pair[0]))} and {_long_date(pair[1])}"
        heading += f", {part[0]}" if part else ""
        schedule = {day: self._items_on(day) for day in days}
        shared = [memory for items in schedule.values() for memory in items if self.guard.may_send(memory, NEBIUS)]
        prepared = self.guard.prepare_context([*shared, *pool], NEBIUS)
        local, draft = _availability_texts(
            heading, schedule, part, prepared, single=len(days) == 1, today=today, routed_by_cloud=routed_by_cloud,
        )
        return _Outcome(draft, prepared, local=local)

    # --- helpers ---------------------------------------------------------------

    def _classify(self, exchange: _Exchange, text: str, private: bool) -> tuple[str, str, str]:
        """``(sensitivity, kind, label)`` by the local rules only: asking a cloud model whether
        a fact is private would show it the fact. A message judged private stays private."""
        sensitivity = "private" if private else classify_new_fact(text)
        return sensitivity, classify_kind(text), make_label(text, sensitivity)

    def _confirm(self, memory: Memory, lead: str) -> _Outcome:
        """A local confirmation draft; private values appear as their placeholder."""
        prepared = self.guard.prepare_context([memory], NEBIUS)
        if prepared.withheld:
            return _Outcome(f"{lead}{placeholder(memory.id)} Its value stays local; "
                            "cloud models only ever see its label.", prepared)
        return _Outcome(f"{lead}{memory.text}", prepared)

    def _find_target(self, hint: str, message: str, *, from_model: bool = False) -> Memory | None:
        """The memory a correction refers to: one whose label the message (or the router's
        hint) names, most specific label first, else a full-text match sharing at least two
        topic words with the message. A hint from a cloud router never selects a memory it
        may not see."""
        candidates = [memory for memory in self.store.all_current() if memory.source_type != "web"]
        for text in (message, hint):
            if text is hint and from_model:  # a cloud router may name only memories it is allowed to see
                candidates = [memory for memory in candidates if self.guard.may_send(memory, NEBIUS)]
            named = [memory for memory in candidates if text and label_mentioned(memory.label, text)]
            if named:
                return max(named, key=lambda memory: (len(words(memory.label)), -memory.id))
        allowed = {memory.id for memory in candidates}
        topic = {stem(word) for word in words(message) if len(word) > 2 and word not in STOPWORDS}
        for memory in self.store.search(message, limit=3):
            shared = topic & {stem(word) for word in words(memory.text)}
            if memory.id in allowed and len(shared) >= 2:  # one common word ("flat") is not enough to overwrite
                return memory
        return None

    def _events_between(self, first: date, last: date) -> list[Memory]:
        """Current event memories dated within [first, last], soonest first."""
        today = self.today()
        dated = [(event_days(memory, today), memory) for memory in self.store.all_current() if memory.kind == "event"]
        within = [(days, memory) for days, memory in dated if days is not None and days[0] <= last and first <= days[1]]
        return [memory for _, memory in sorted(within, key=lambda pair: (pair[0][0], pair[1].id))]

    def _retrieve(self, message: str, window: tuple[date, date] | None, *, agent: bool = False) -> list[Memory]:
        """For a period: its shared events, then every upcoming withheld event whatever the
        period (so the list dates none of them); then the best full-text matches.

        A message that refers to a time in any way, even one the period note leaves out (a
        weekday, a masked date), gets the same full withheld list, and full-text search may not
        add withheld memories to it: which ones share its words ("Tuesday") would date them.
        Nor may it for another agent's message, whatever it says: the agent chose the words."""
        found: dict[int, Memory] = {}
        timed = window is not None or names_time(message)
        if window:
            for memory in self._events_between(*window):
                if self.guard.may_send(memory, NEBIUS):
                    found[memory.id] = memory
        schedule = self._private_schedule() if timed else []
        # Restricted to what Nebius may see, the search is also ranked without the private memories: searched
        # with them and filtered after, private matches would use up result slots, so which shared memories
        # make the context would tell which words the private ones hold.
        levels = sorted(self.guard.allowed(NEBIUS)) if timed or agent else None
        for memory in self.store.search(message, limit=8, sensitivities=levels):
            if (timed or agent) and not self.guard.may_send(memory, NEBIUS):
                continue
            found.setdefault(memory.id, memory)
        return self._capped([*found.values(), *schedule])

    def _capped(self, memories: list[Memory]) -> list[Memory]:
        """At most :data:`MAX_CONTEXT_MEMORIES` of the memories picked for a message, then every item of the
        private schedule among ``memories`` (up to :data:`MAX_WITHHELD_ITEMS`). A placeholder is one short line;
        counted against the same cap, the private items of a busy month would push out the shared ones, or
        each other, and an answer about a day could lose that day's private item."""
        schedule = {memory.id for memory in self._private_schedule()}
        picked = [memory for memory in memories if memory.id not in schedule][:MAX_CONTEXT_MEMORIES]
        held = list({memory.id: memory for memory in memories if memory.id in schedule}.values())
        return picked + held[:MAX_WITHHELD_ITEMS]

    def _items_on(self, day: date) -> list[Memory]:
        """Current events on ``day`` (one dated that day, or a range of days that holds it) and routines held
        on its weekday ("every Tuesday"), by start time that day."""
        today = self.today()

        def on_day(memory: Memory) -> bool:
            days = event_days(memory, today) if memory.kind == "event" else None
            return (days is not None and days[0] <= day <= days[1]) or recurs_on(memory, day)

        items = [memory for memory in self.store.all_current() if memory.source_type != "web" and on_day(memory)]
        return sorted(items, key=lambda memory: ((item_span_on(memory, day, today) or (24 * 60, 0))[0], memory.id))

    def _private_schedule(self) -> list[Memory]:
        """Every current event or weekly routine the reasoning model may not see, whatever its date, by id.

        Listed whole for any period, a brief or another day: a list cut to a period, or to the days still to
        come, would date its items by which ones are in it, or by the day one drops out. The device leaves out,
        in the answer it shows, those that are over (:meth:`_is_over`)."""
        today = self.today()
        return [
            memory for memory in self.store.all_current()
            if memory.source_type != "web" and not self.guard.may_send(memory, NEBIUS)
            and (memory.kind == "event" or any(recurs_on(memory, today + timedelta(days=n)) for n in range(7)))
        ]

    def _is_outside(self, memory_id: int, period: tuple[date, date]) -> bool:
        """True for a current private event none of whose days falls in ``period``, or a private weekly routine
        held on none of its days: on this device only, for leaving it out of the answer shown."""
        try:
            memory = self.store.get(memory_id)
        except MemoryNotFound:
            return False
        if memory.status != "current" or self.guard.may_send(memory, NEBIUS):
            return False
        first, last = period
        today = self.today()
        if memory.kind == "event":
            try:
                days = event_days(memory, today)
            except (ValueError, OverflowError):
                return False
            if days is not None:
                return days[1] < first or last < days[0]
        span = min((last - first).days, 6)
        weekly = any(recurs_on(memory, today + timedelta(days=n)) for n in range(7))
        return weekly and not any(recurs_on(memory, first + timedelta(days=n)) for n in range(span + 1))

    def _is_over(self, memory_id: int) -> bool:
        """True for a current private event whose last day is before today: on this device only, for leaving it
        out of the answer shown."""
        try:
            memory = self.store.get(memory_id)
        except MemoryNotFound:
            return False
        if memory.status != "current" or memory.kind != "event" or self.guard.may_send(memory, NEBIUS):
            return False
        try:
            days = event_days(memory, self.today())
        except (ValueError, OverflowError):
            return False
        return days is not None and days[1] < self.today()

    def _compose(
        self, exchange: _Exchange, purpose: str, request: str, memories: list[Memory], notes: Sequence[str] = (),
        *, mask: bool = False, calendar: bool = False,
    ) -> _Outcome:
        """Prepare the guarded context, ask the reasoning model, fall back to a local list. ``mask``: the request
        is a message the rules find private, sent with its values, names, days and months masked. ``calendar``:
        its dates, clock times and days of the month are masked, all of them alike."""
        prepared = self.guard.prepare_context(memories, NEBIUS)
        if mask:
            request = masked_private(request)
        elif exchange.agent:  # every value and name masked alike, whatever private memories hold
            request = masked(request)
        elif calendar:  # likewise for the dates and times a user typed; put back here in the answer shown
            exchange.typed_calendar = calendar_values(request)
            request = masked_calendar(request)
        # What the user typed is checked against every stored value, shared ones included. An agent's words are
        # checked for credentials only, and sent after the stored-value check of the context: redacting the ones
        # that match a private value ("11月3日", "half ten", "rossi") would tell the agent which of its guesses
        # are right, and the request Nebius gets would depend on private memories.
        typed = self.guard.check_outbound(request, NEBIUS, () if exchange.agent else self.store.all_versions(),
                                          purpose=purpose, strict=True)
        if typed.withheld_ids or typed.reasons:  # what the user typed: no shared-value exemption
            exchange.decisions.append(typed)
            exchange.withheld_ids.update(typed.withheld_ids)
            request = typed.redacted_text
        header = "\n".join([f"Today is {_long_date(self.today())}.", *notes])
        asked = f"\nRequest: {request}"
        prompt = "\n".join(["", "Memories:", prepared.context_text or "(none relevant)"])
        raw = exchange.complete(
            purpose, "reason", ANSWER_SYSTEM_PROMPT, prompt + ("" if exchange.agent else f"\n{asked}"),
            prepared=prepared, max_tokens=REASON_MAX_TOKENS, temperature=REASON_TEMPERATURE, header=header,
            trailer=f"\n{asked}" if exchange.agent else "",
        )
        return _Outcome(raw if raw is not None else self._fallback(exchange, prepared, memories), prepared)

    @staticmethod
    def _fallback(exchange: _Exchange, prepared: PreparedContext, memories: list[Memory]) -> str:
        """A local answer listing the relevant memories (private ones as placeholders)."""
        if exchange.mode == "replay" and exchange.request_differs:
            lead = "The recording of this question answered a different request, so no model answered it"
        elif exchange.mode == "replay":
            lead = "No recorded turn matches this question, so no model answered it"
            exchange.notices.append(REPLAY_SCOPE)
        else:
            lead = "The cloud model could not answer this time"
        included = {memory.id for memory in [*prepared.shared, *prepared.withheld]}
        withheld = set(prepared.withheld_ids)
        lines: list[str] = []
        for memory in memories:
            if memory.id in included:
                included.discard(memory.id)
                shown = placeholder(memory.id) if memory.id in withheld else f"{memory.text} [{memory.cite()}]"
                lines.append(f"- {shown}")
        if not lines:
            return f"{lead}, and nothing in memory matches its words."
        return f"{lead}. The memories that match it:\n" + "\n".join(lines)
