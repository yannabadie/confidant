"""Offline replay: recorded model outputs, keyed by normalised question.

Replay is used only when there is no API key, the budget is spent, or no
Nemotron model can be reached. A recorded turn holds the raw outputs of a live
turn's model calls by purpose ("route", "classify", "answer", "brief") plus its
web results, so everything else in the turn (guard checks, memory updates, late
binding, citations) still runs locally, and every replayed call is logged with
mode "replay".

File format (``data/replay/*.json``): a list of turns, ``{"turns": [...]}`` or
a single turn::

    {"question": "What's on my plate next week?",
     "outputs": {"route": "{\\"intent\\": \\"answer\\"}",
                 "answer": "Next week: ⟦m7⟧ ... [m6v1]"},
     "web_results": [{"title": "...", "url": "https://...", "content": "...", "score": 0.9}],
     "models": {"reason": "nvidia/...", "fast": "nvidia/..."},
     "recorded_at": "2026-10-28T10:00:00Z"}

Outputs may also be given at the top level ("route": {...}, "answer": "..."),
and JSON objects are accepted where text is expected.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .scan import fold
from .search import WebResult

PURPOSES = ("route", "classify", "answer", "brief")


def normalize_question(text: str) -> str:
    """Lookup key for a question: case, accents, apostrophes and punctuation ignored."""
    folded = fold(text)[0].replace("'", "").replace("’", "")
    return " ".join(re.findall(r"[^\W_]+", folded))


def _as_text(value: Any) -> str:
    """Recorded output as text; JSON objects and lists are serialised."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True)
class RecordedTurn:
    """The model outputs and web results of one recorded live turn."""

    question: str
    outputs: dict[str, str]
    web_results: list[WebResult] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    recorded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, as written by :meth:`ReplayBook.save`."""
        return {
            "question": self.question,
            "outputs": dict(self.outputs),
            "web_results": [asdict(result) for result in self.web_results],
            "models": dict(self.models),
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecordedTurn:
        """Parse one turn of the replay file format; ValueError if it has no question."""
        question = " ".join(str(data.get("question") or "").split())
        if not question:
            raise ValueError("a recorded turn needs a question")
        outputs = {str(key): _as_text(value) for key, value in dict(data.get("outputs") or {}).items()}
        for purpose in PURPOSES:
            if purpose in data and purpose not in outputs:
                outputs[purpose] = _as_text(data[purpose])
        results = [
            WebResult(
                title=str(item.get("title") or item["url"]), url=str(item["url"]),
                content=str(item.get("content") or ""), score=float(item.get("score") or 0.0),
            )
            for item in data.get("web_results") or []
        ]
        models = {str(role): str(model) for role, model in dict(data.get("models") or {}).items()}
        return cls(question, outputs, results, models, str(data.get("recorded_at") or ""))


class ReplayBook:
    """Recorded turns, looked up by normalised question: exact match first, then a close one."""

    FUZZY_CUTOFF = 0.88

    def __init__(self, turns: Iterable[RecordedTurn] = ()) -> None:
        self._turns: dict[str, RecordedTurn] = {}
        for turn in turns:
            self.add(turn)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ReplayBook:
        """Read one replay file, or every ``*.json`` file of a directory.

        A missing path gives an empty book; a malformed file raises ValueError naming it.
        """
        source = Path(path)
        files = sorted(source.glob("*.json")) if source.is_dir() else [source] if source.is_file() else []
        book = cls()
        for file in files:
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                items = data.get("turns", [data]) if isinstance(data, dict) else data
                for item in items:
                    book.add(RecordedTurn.from_dict(item))
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"{file}: not a valid replay file ({exc})") from None
        return book

    def lookup(self, question: str) -> RecordedTurn | None:
        """The recorded turn for ``question``, or for a near-identical wording, else None."""
        key = normalize_question(question)
        if key in self._turns:
            return self._turns[key]
        close = difflib.get_close_matches(key, list(self._turns), n=1, cutoff=self.FUZZY_CUTOFF)
        return self._turns[close[0]] if close else None

    def add(self, turn: RecordedTurn) -> None:
        """Add or replace the recording for ``turn.question``."""
        self._turns[normalize_question(turn.question)] = turn

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write every recorded turn to ``path`` as ``{"turns": [...]}``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        turns = [turn.to_dict() for turn in self._turns.values()]
        target.write_text(json.dumps({"turns": turns}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def questions(self) -> list[str]:
        """The recorded questions, as originally written."""
        return [turn.question for turn in self._turns.values()]

    def __len__(self) -> int:
        """Number of recorded turns."""
        return len(self._turns)
