"""The command line: seed, chat without keys (replay), brief and receipts."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from confidant.cli import main

from conftest import PERSONA


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A seeded Confidant home, with API keys set to empty strings (which count as unset)."""
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    assert main(["--home", str(tmp_path), "seed", "--persona", str(PERSONA)]) == 0
    return tmp_path


def replay_dir(tmp_path: Path) -> Path:
    """A replay folder with one recorded turn."""
    folder = tmp_path / "replay"
    folder.mkdir()
    (folder / "demo.json").write_text(json.dumps([{
        "question": "What's on my plate next week?",
        "outputs": {"route": {"intent": "answer"}, "answer": "Next week: ⟦m7⟧ and the review [m6v1]."},
    }]), encoding="utf-8")
    return folder


def test_seed_loads_the_persona_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--home", str(tmp_path), "seed", "--persona", str(PERSONA)]) == 0
    out = capsys.readouterr().out
    assert "Seeded 25 memories" in out and "snapshot date 2026-10-30" in out
    assert main(["--home", str(tmp_path), "seed", "--persona", str(PERSONA)]) == 1
    assert "already holds 25 memories" in capsys.readouterr().err


def test_chat_without_keys_says_so_and_uses_replay(home: Path, monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("What's on my plate next week?\n\nRemember my PIN is 4821\nquit\n"))
    assert main(["--home", str(home), "--replay", str(replay_dir(home)), "chat"]) == 0
    captured = capsys.readouterr()
    assert "NEBIUS_API_KEY is not set: answers come from replay (1 recorded turns" in captured.err
    assert "Next week: Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30. and the review [m6v1]." in captured.out
    assert "sources: m7v1 · seed · 2026-10-19 | m6v1 · seed · 2026-10-05" in captured.out
    assert "guard: 2 private memories never left Confidant · mode: replay" in captured.out
    assert "I won't keep or send that" in captured.out


def test_brief_and_receipts(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert main(["--home", str(home), "--replay", str(home / "none"), "brief"]) == 0
    out = capsys.readouterr().out
    assert "Replay has no recorded answer" in out and "Receipts today: 0 outbound calls" in out
    assert main(["--home", str(home), "receipts"]) == 0
    assert "No outbound calls today." in capsys.readouterr().out


def test_brief_delivery_sends_labels_never_private_values(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = home / "brief-replay"
    folder.mkdir()
    (folder / "brief.json").write_text(json.dumps([{
        "question": "Run my daily brief.",
        "outputs": {"route": {"intent": "brief"},
                    "answer": "This week:\n⟦m7⟧\n⟦m11⟧\nThe design review [m6v1]."},
    }]), encoding="utf-8")
    capsys.readouterr()
    assert main(["--home", str(home), "--replay", str(folder), "brief", "--delivery"]) == 0
    out = capsys.readouterr().out
    assert "[private: dermatologist appointment]" in out
    assert "10:30" not in out and "Dermatologist appointment on" not in out
    assert "⟦" not in out
    assert "Private items stay on your device" in out and "replay" in out
    assert main(["--home", str(home), "--replay", str(folder), "brief"]) == 0
    assert "Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30." in capsys.readouterr().out
