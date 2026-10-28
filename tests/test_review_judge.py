"""A zero-context judge of the whole submission (a Claude agent with no history of the project, 28 October 2026,
evening) asked the demo its own time questions, with a scripted model that copies every placeholder the way the
recorded live answers do. Three faults, all in what this device shows, never in what it sends:

- "What's on this weekend?" showed the Tuesday dermatologist appointment and the Thursday interview: Nebius gets
  every private item whatever the period, and the device took out only the items that were over.
- With 13 more private appointments, "What's on my plate next week?" dropped one of next week's: placeholders
  counted against the 10-memory context cap.
- The brief listed the private items apart, after Friday's dinner, instead of in date order.

Each is reproduced here before its fix. What leaves the device must not change: tests/test_differential.py
still checks that.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from confidant import Assistant, Budget, Guard, MemoryStore, ReceiptLog
from confidant.replay import ReplayBook

from conftest import MODELS, PERSONA, FakeLLM, Rig

REPLAY = Path(__file__).resolve().parents[1] / "data" / "replay"
_CONTEXT_LINE = re.compile(r"^\[(m\d+v\d+)\] (.+)$")
_HELD_LINE = re.compile(r"^(⟦m\d+⟧ \(private: [^)]*\))$")


def copy_all(messages: list[dict[str, str]]) -> str:
    """A model that lists every memory it got, shared ones with their citation, withheld ones as their
    placeholder line, as the live answers of 28 October do."""
    lines = []
    for line in messages[-1]["content"].splitlines():
        if match := _CONTEXT_LINE.match(line):
            lines.append(f"- {match.group(2)} [{match.group(1)}]")
        elif match := _HELD_LINE.match(line):
            lines.append(f"- {match.group(1)}")
    return "\n".join(lines)


def nebius_users(llm: FakeLLM) -> str:
    return "\n".join(message["content"] for call in llm.calls for message in call["messages"]
                     if message["role"] == "user")


def test_a_weekend_question_shows_no_private_item_of_another_day(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig({"route": '{"intent": "answer"}', "answer": copy_all})
    turn = rig.ask("What's on this weekend?")
    assert "Dermatologist" not in turn.answer and "interview" not in turn.answer
    assert "⟦m7⟧" in turn.draft and "⟦m11⟧" in turn.draft  # the draft, which may leave, keeps them all


def test_a_question_about_its_day_still_shows_it(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig({"route": '{"intent": "answer"}', "answer": copy_all})
    turn = rig.ask("What's on Tuesday?")
    assert "Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30." in turn.answer
    assert "interview" not in turn.answer


def test_the_replay_fallback_lists_only_the_period_asked() -> None:
    store = MemoryStore()
    store.load_seed(PERSONA)
    assistant = Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), None,
                          replay=ReplayBook.load(REPLAY))
    turn = assistant.handle("Plan my weekend", session_id="s")
    assert turn.mode == "replay"
    assert "Dermatologist" not in turn.answer and "interview" not in turn.answer


MORE_PRIVATE = [
    ("Blood test at the lab on Monday 2 Nov 2026 at 08:00.", "health appointment"),
    *[(f"Physiotherapy session on {day} Nov 2026 at 18:00.", "health appointment")
      for day in ("Monday 16", "Tuesday 17", "Wednesday 18", "Thursday 19", "Friday 20", "Saturday 21")],
    *[(f"Call with the bank about the loan on {day} Nov 2026 at 12:30.", "finances")
      for day in ("Monday 23", "Tuesday 24", "Wednesday 25", "Thursday 26", "Friday 27", "Saturday 28")],
]


def crowded(tmp_path: Path) -> MemoryStore:
    data = json.loads(PERSONA.read_text(encoding="utf-8"))
    next_id = max(memory["id"] for memory in data["memories"]) + 1
    for offset, (text, label) in enumerate(MORE_PRIVATE):
        data["memories"].append({"id": next_id + offset, "text": text, "kind": "event", "sensitivity": "private",
                                 "label": label, "source_type": "user", "source_ref": "test", "author": "alex"})
    path = tmp_path / "crowded.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = MemoryStore()
    store.load_seed(path)
    return store


def test_placeholders_do_not_use_up_the_context(tmp_path: Path) -> None:
    store = crowded(tmp_path)
    llm = FakeLLM({"route": '{"intent": "answer"}', "answer": copy_all})
    assistant = Assistant(store, Guard(), ReceiptLog(), Budget(1_000_000, 100), llm, models=MODELS)
    turn = assistant.handle("What's on my plate next week?", session_id="s")
    sent = nebius_users(llm)
    private_events = [m for m in store.all_current() if m.sensitivity == "private" and m.kind == "event"]
    assert all(f"⟦m{memory.id}⟧" in sent for memory in private_events)
    assert "[m6v1]" in sent and "[m9v1]" in sent  # next week's shared events are still there
    assert "Blood test at the lab on Monday 2 Nov 2026 at 08:00." in turn.answer
    assert "Physiotherapy" not in turn.answer and "loan" not in turn.answer  # after next week


BRIEF = (
    "**Events (next 7 days)**\n"
    "- Today, Fri 30 Oct, 17:00 – Weekly review [m8v1]\n"
    "- Wed 4 Nov, 15:00 – Quarterly product review [m6v1]\n"
    "- Fri 6 Nov, 20:00 – Dinner with friends at the Lebanese place [m9v1]\n"
    "\n"
    "**Withheld**\n"
    "⟦m7⟧ (private: health appointment, value withheld)\n"
    "⟦m11⟧ (private: private appointment, value withheld)\n"
    "\n"
    "**Open to-do items**\n"
    "- Book a table for Léa's birthday dinner by Tue 10 Nov [m14v1]"
)


def test_the_brief_shows_withheld_items_in_date_order(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig({"route": '{"intent": "brief"}', "brief": BRIEF})
    turn = rig.ask("Run my daily brief.")
    shown = turn.answer
    order = [shown.index(text) for text in ("Weekly review", "Dermatologist appointment", "Quarterly product review",
                                            "final-round interview", "Dinner with friends")]
    assert order == sorted(order)
    assert "**Withheld**" not in shown  # its items moved into the list, so its heading went too
    assert "**Withheld**" in turn.draft and turn.draft.index("⟦m7⟧") > turn.draft.index("Dinner")  # unchanged
