"""The 199 attack messages of the leak test, run by the test suite.

Round 10 (29 October 2026): a judge noted that the "0 of 199" of ``docs/leak-test.md`` was enforced by no test, and
that counting leaked values cannot see an oracle, a redaction that happens only when a private value matches (it
counts "[withheld]" as a success). So the suite runs the leak test on the five rounds of messages, and runs the Prove
it comparison on every message: what leaves the device must be the same, byte for byte, when every private
appointment moves to another week, weekday and hour. Both take a few minutes; nothing is sent anywhere.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from confidant import Assistant, Budget, Guard, MemoryStore, ReceiptLog
from confidant.proof import prove

from conftest import MODELS, PERSONA, FakeLLM

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ["leak_prompts.jsonl", "leak_prompts_heldout.jsonl", "leak_prompts_round3.jsonl",
            "leak_prompts_round4.jsonl", "leak_prompts_round9.jsonl"]


def cases(name: str) -> list[dict]:
    lines = (ROOT / "tests" / "fixtures" / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_the_five_rounds_hold_199_messages() -> None:
    assert sum(len(cases(name)) for name in FIXTURES) == 199


@pytest.mark.parametrize("name", FIXTURES)
def test_no_attack_message_leaks_a_private_value(name: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/leak_test.py", "--fixtures", f"tests/fixtures/{name}",
         "--out", str(tmp_path / "results.json"), "--strict", "--quiet"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


def store_after_setup(case: dict) -> MemoryStore:
    """The demo memory after the case's setup turns ("Remember that I was diagnosed with..."), as the leak test runs
    them: until round 11, 44 cases were compared without the private facts their setup stores."""
    store = MemoryStore()
    store.load_seed(PERSONA)
    assistant = Assistant(store, Guard(), ReceiptLog(), Budget(10**9, 10**6),
                          FakeLLM({"route": '{"intent": "answer"}', "answer": "ok"}), models=MODELS)
    for message in case.get("setup", []):
        assistant.handle(message, session_id=f"setup-{case['id']}")
    return store


@pytest.mark.parametrize("name", FIXTURES)
def test_no_attack_message_depends_on_when_private_appointments_happen(name: str) -> None:
    differ = []
    for case in cases(name):
        if not prove(store_after_setup(case), case["message"]).identical:
            differ.append((case["id"], case["message"]))
    assert differ == []
