"""Shared fixtures: the demo persona, scripted fake model and search clients, and a test rig.

Nothing here touches the network: the fakes stand in for Token Factory and Tavily.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from confidant import Assistant, Budget, Guard, MemoryStore, ReceiptLog, ReplayBook, WebResult
from confidant.llm import ChatResult

PERSONA = Path(__file__).resolve().parent.parent / "data" / "demo_persona.json"
MODELS = {"reason": "nvidia/Nemotron-3-Ultra-550b-a55b", "fast": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"}
Messages = Sequence[Mapping[str, str]]
Reply = str | Exception | Callable[[Messages], str]

TAPER_RESULTS = [
    WebResult("Ten-day half marathon taper", "https://running.example.org/taper",
              "Cut weekly volume by 40 to 60 percent over the last ten days and keep short strides.", 0.91),
    WebResult("How to taper for a half", "https://www.coach.example.com/half-taper",
              "Keep one session at race pace, rest two days before the race.", 0.84),
]


def request_line(messages: Messages) -> str:
    """The user's request as it reached the model (the last line starting with "Request:")."""
    content = messages[-1]["content"]
    lines = [line for line in content.splitlines() if line.startswith("Request:")]
    return (lines[-1] if lines else content).lower()


def demo_router(messages: Messages) -> str:
    """Router replies for the demo script, in the shapes a real model produces."""
    message = messages[-1]["content"].lower()
    if "taper" in message:
        return '```json\n{"intent": "search", "query": "10-day half marathon taper plan"}\n```'
    if "birthday" in message:
        return 'Sure! {"intent": "answer"}'
    if "brief" in message:
        return '{"intent": "brief"}'
    if "aisle" in message:
        return '{"intent": "remember", "text": "Prefers aisle seats on long train trips."}'
    return '{"intent": "answer"}'


def demo_answer(messages: Messages) -> str:
    """Reasoning-model replies for the demo script, citing tags and copying placeholders."""
    request = request_line(messages)
    if "plate" in request:
        return ("Next week:\n- ⟦m7⟧\n- Quarterly product review, Wed 4 Nov at 15:00 [m6v1]\n- ⟦m11⟧\n"
                "- Dinner with friends, Fri 6 Nov at 20:00 [m9v1]")
    if "taper" in request:
        return ("Race day is Sun 15 Nov [m3v1]. Over the last ten days cut volume by 40-60% and keep strides "
                "[m26v1]; that matches your coach's advice [m25v1].")
    if "birthday" in request:
        return "Happy 30th, Léa! May your climbs be high and your puns terrible. [m4v1] [m13v1]"
    return "Nothing in memory about that."


DEMO_REPLIES: dict[str, Reply] = {
    "route": demo_router,
    "classify": '{"sensitivity": "personal", "kind": "preference", "label": "seat preference"}',
    "answer": demo_answer,
    "brief": "Today is Friday: weekly review at 17:00 [m8v1].\n- ⟦m7⟧\n- Quarterly product review [m6v1]\n"
             "- ⟦m11⟧\n- To do: book Léa's table [m14v1]",
}


class FakeLLM:
    """Scripted stand-in for TokenFactory: replies by purpose and records every request."""

    def __init__(self, replies: Mapping[str, Reply] | None = None) -> None:
        self.replies = dict(DEMO_REPLIES if replies is None else replies)
        self.calls: list[dict] = []

    def chat(self, model: str, messages: Messages, *, max_tokens: int, temperature: float, purpose: str) -> ChatResult:
        """Return the scripted reply for ``purpose`` (or raise it, if it is an exception)."""
        self.calls.append({"model": model, "purpose": purpose, "max_tokens": max_tokens,
                           "messages": [dict(message) for message in messages]})
        reply = self.replies.get(purpose, "")
        if isinstance(reply, Exception):
            raise reply
        text = reply(messages) if callable(reply) else reply
        return ChatResult(text=text, prompt_tokens=120, completion_tokens=30, latency_ms=7, model=model)

    def list_models(self) -> list[str]:
        """A catalog containing the default models."""
        return sorted(MODELS.values())

    def purposes(self) -> list[str]:
        """Purposes of the calls made so far, in order."""
        return [call["purpose"] for call in self.calls]

    def sent_text(self) -> str:
        """Everything that was sent, as one string, for leak checks."""
        return json.dumps([call["messages"] for call in self.calls], ensure_ascii=False)


class FakeTavily:
    """Scripted stand-in for Tavily that records the queries it receives."""

    def __init__(self, results: Sequence[WebResult] = TAPER_RESULTS) -> None:
        self.results = list(results)
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int = 5) -> list[WebResult]:
        """Return the scripted results."""
        self.queries.append(query)
        return self.results[:max_results]


@dataclass
class Rig:
    """An assistant wired to fakes, with handles on everything a test inspects."""

    assistant: Assistant
    store: MemoryStore
    receipts: ReceiptLog
    llm: FakeLLM | None
    tavily: FakeTavily | None

    def ask(self, message: str, session_id: str = "s1"):
        """Send one message through the assistant."""
        return self.assistant.handle(message, session_id=session_id)


@pytest.fixture
def store() -> MemoryStore:
    """A fresh in-memory store seeded with the demo persona."""
    memory = MemoryStore()
    memory.load_seed(PERSONA)
    return memory


@pytest.fixture
def make_rig(store: MemoryStore) -> Callable[..., Rig]:
    """Factory for rigs: live by default (scripted FakeLLM), optionally with replay or a budget."""

    def build(
        replies: Mapping[str, Reply] | None = None,
        *,
        live: bool = True,
        web: bool = True,
        replay: ReplayBook | None = None,
        budget: Budget | None = None,
        recorder: ReplayBook | None = None,
    ) -> Rig:
        """A rig over the shared seeded store; ``live=False`` means no key, ``web=False`` no Tavily."""
        llm = FakeLLM(replies) if live else None
        tavily = FakeTavily() if web else None
        receipts = ReceiptLog()
        assistant = Assistant(
            store, Guard(), receipts, budget or Budget(1_000_000, 100), llm, search=tavily, replay=replay,
            models=MODELS if live else None, recorder=recorder,
        )
        return Rig(assistant, store, receipts, llm, tavily)

    return build
