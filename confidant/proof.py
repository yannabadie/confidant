"""Prove it: what leaves the device does not depend on when private items happen.

For one question (or the daily brief), the same turn is built twice, with no model call and nothing sent: on a
copy of the memory as it is, and on a copy where every private appointment is moved to another day and hour.
Each request the turn would send is hashed (SHA-256 of its exact messages, as its receipt keeps it). Identical
hashes show that Nemotron could not tell when those appointments are, nor whether one falls in the period
asked: a placeholder dated by its neighbours, or left out of the list on some days, would change a request.
The same comparison runs on 30 question pairs in ``tests/test_differential.py``; this runs it on any question
of the visitor's session. A rewrite of a shared memory, which Nemotron does receive, changes the hashes: the
check is not vacuous (``tests/test_proof.py``).
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from .assistant import Assistant
from .budget import Budget
from .guard import NEBIUS, Guard
from .llm import ChatResult
from .memory import Memory, MemoryStore
from .receipts import ReceiptLog
from .rules import event_days
from .scan import find_dates, find_values

SHIFT_DAYS = 9  # another week and another weekday: Tue 3 Nov becomes Thu 12 Nov
SHIFT_MINUTES = 75  # and another hour: 10:30 becomes 11:45
_MODELS = {"reason": "proof", "fast": "proof"}  # names only: a request's hash covers its messages, not the model


@dataclass(frozen=True)
class Move:
    """One private appointment as stored and as moved for the comparison."""

    memory_id: int
    before: str
    after: str


@dataclass(frozen=True)
class Proof:
    """The requests one turn would send, before and after the private appointments moved."""

    question: str
    moves: list[Move] = field(default_factory=list)
    original: list[tuple[str, str]] = field(default_factory=list)  # (purpose, SHA-256), in order
    moved: list[tuple[str, str]] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        """True when every request is byte for byte the same, or when the turn sends none either way."""
        return self.original == self.moved

    @property
    def sent(self) -> bool:
        """True when the turn sends at least one request (False: answered on this device)."""
        return bool(self.original or self.moved)


class _Capture:
    """Stands in for Token Factory: keeps nothing, sends nothing, answers the route as the real turn was routed
    and every other call with no text (the turn then falls back locally; only its requests matter here)."""

    def __init__(self, intent: str) -> None:
        self.route = json.dumps({"intent": intent})

    def chat(self, model: str, messages: Sequence[dict[str, str]], *, max_tokens: int, temperature: float,
             purpose: str, thinking: bool = True) -> ChatResult:
        return ChatResult(text=self.route if purpose == "route" else "", prompt_tokens=0, completion_tokens=0,
                          latency_ms=0, model=model)

    def list_models(self) -> list[str]:
        return sorted(_MODELS.values())


def moved_text(memory: Memory, today: date, guard: Guard) -> str:
    """The text of a private appointment with each date moved by :data:`SHIFT_DAYS` and each clock time by
    :data:`SHIFT_MINUTES`, written in full ("Thursday 12 Nov 2026", "11:45"); any other memory as it is."""
    if memory.kind != "event" or guard.may_send(memory, NEBIUS):
        return memory.text
    try:
        if event_days(memory, today) is None:
            return memory.text
    except (ValueError, OverflowError):
        return memory.text
    edits: list[tuple[int, int, str]] = []
    for found in find_dates(memory.text):
        day = found.resolve(today) + timedelta(days=SHIFT_DAYS)
        edits.append((found.start, found.end, f"{day:%A} {day.day} {day:%b} {day.year}"))
    for value in find_values(memory.text):
        if value.kind == "time" and not any(start <= value.start < end for start, end, _ in edits):
            hour, minute = (int(part) for part in value.folded.split(":"))
            total = (hour * 60 + minute + SHIFT_MINUTES) % (24 * 60)
            edits.append((value.start, value.end, f"{total // 60:02d}:{total % 60:02d}"))
    text = memory.text
    for start, end, new in sorted(edits, reverse=True):
        text = text[:start] + new + text[end:]
    return text


def _requests(store: MemoryStore, question: str, intent: str, brief: bool, guard: Guard) -> list[tuple[str, str]]:
    assistant = Assistant(store, guard, ReceiptLog(":memory:"), Budget(10**9, 10**6), _Capture(intent),
                          models=_MODELS)
    turn = assistant.daily_brief("proof") if brief else assistant.handle(question, session_id="proof")
    return [(call.purpose, call.sha256) for call in turn.outbound]


def prove(
    store: MemoryStore, question: str, *, intent: str = "answer", guard: Guard | None = None,
    rewrite: Callable[[Memory, date], str] | None = None,
) -> Proof:
    """Build ``question``'s turn (the daily brief when ``intent`` is "brief") on two copies of ``store``, as it
    is and with the private appointments moved, and compare the hashes of the requests. ``store`` itself is
    not changed and nothing leaves the device. ``rewrite`` replaces the move, for tests."""
    guard = guard or Guard()
    today = store.as_of() or date.today()
    change = rewrite or (lambda memory, day: moved_text(memory, day, guard))
    original = store.copy()
    moved = store.copy(lambda memory: change(memory, today))
    moves = [Move(memory.id, memory.text, after) for memory in store.all_current()
             if (after := change(memory, today)) != memory.text]
    try:
        brief = intent == "brief"
        return Proof(question, moves, _requests(original, question, intent, brief, guard),
                     _requests(moved, question, intent, brief, guard))
    finally:
        original.close()
        moved.close()
