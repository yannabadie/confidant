"""ReceiptLog: one row per call, hashes only, daily summaries."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from confidant.receipts import ReceiptLog, payload_sha256

PAYLOAD = [{"role": "user", "content": "Dermatologist appointment secret-ish text"}]


def at(day: int) -> datetime:
    """A fixed UTC time on day ``day`` of November 2026."""
    return datetime(2026, 11, day, 9, 30, tzinfo=timezone.utc)


def test_receipt_keeps_the_payload_hash_and_never_the_payload(tmp_path: Path) -> None:
    path = tmp_path / "receipts.db"
    log = ReceiptLog(path)
    receipt = log.record(provider="nebius_token_factory", model="nvidia/x-ultra", purpose="answer",
                         payload=PAYLOAD, prompt_tokens=10, completion_tokens=5, latency_ms=42,
                         sent_memory_ids=[6, 9, 6], withheld_memory_ids=[11, 7], cost_usd_estimate=0.00002,
                         session_id="s1")
    assert receipt.payload_sha256 == payload_sha256(PAYLOAD) and len(receipt.payload_sha256) == 64
    assert receipt.sent_memory_ids == [6, 9] and receipt.withheld_memory_ids == [7, 11]
    log.close()
    raw = path.read_bytes()
    assert b"Dermatologist" not in raw and b"secret-ish" not in raw
    assert receipt.payload_sha256.encode() in raw
    assert ReceiptLog(path).today()[0] == receipt


def test_payload_hash_is_canonical() -> None:
    assert payload_sha256({"b": 1, "a": "é"}) == payload_sha256({"a": "é", "b": 1})
    assert payload_sha256("text") == payload_sha256(b"text")


def test_today_summary_counts_tokens_cost_and_memories_kept_local() -> None:
    log = ReceiptLog(now=lambda: at(3))
    log.record(provider="nebius_token_factory", model="m", purpose="answer", payload="a", prompt_tokens=100,
               completion_tokens=20, sent_memory_ids=[4], withheld_memory_ids=[7, 11], cost_usd_estimate=0.0002)
    log.record(provider="tavily", model="tavily-search-basic", purpose="web_search", payload="b",
               withheld_memory_ids=[4], cost_usd_estimate=0.008)
    log.record(provider="nebius_token_factory", model="m", purpose="answer", payload="c", mode="replay")
    summary = log.today_summary()
    assert (summary.calls, summary.live_calls, summary.replay_calls) == (3, 2, 1)
    assert (summary.prompt_tokens, summary.completion_tokens, summary.total_tokens) == (100, 20, 120)
    assert summary.cost_usd_estimate == pytest.approx(0.0082)
    assert summary.memories_kept_local == 2  # 7 and 11; memory 4 went to the model
    assert summary.line() == ("Receipts today: 2 outbound calls, 1 replayed · 120 tokens · ≈$0.0082 · "
                              "2 withheld memories never left Confidant")


def test_today_filters_by_utc_day_and_session() -> None:
    clock = {"now": at(2)}
    log = ReceiptLog(now=lambda: clock["now"])
    log.record(provider="p", model="m", purpose="answer", payload="old", session_id="s1")
    clock["now"] = at(3)
    log.record(provider="p", model="m", purpose="answer", payload="new", session_id="s1")
    log.record(provider="p", model="m", purpose="answer", payload="other", session_id="s2")
    assert len(log.today()) == 2
    assert [r.session_id for r in log.today("s1")] == ["s1"]
    assert log.today("s1")[0].at == "2026-11-03T09:30:00Z"


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        ReceiptLog().record(provider="p", model="m", purpose="x", payload="", mode="maybe")


def test_a_log_written_before_receipts_had_a_status_still_opens(tmp_path) -> None:
    """Receipts gained a status on 28 Oct 2026; a log written before then reads as all answered."""
    import sqlite3
    path = tmp_path / "receipts.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE receipts (seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, session_id TEXT NOT NULL,"
        " provider TEXT NOT NULL, model TEXT NOT NULL, purpose TEXT NOT NULL, prompt_tokens INTEGER NOT NULL,"
        " completion_tokens INTEGER NOT NULL, latency_ms INTEGER NOT NULL, sent_memory_ids TEXT NOT NULL,"
        " withheld_memory_ids TEXT NOT NULL, payload_sha256 TEXT NOT NULL, cost_usd_estimate REAL NOT NULL,"
        " mode TEXT NOT NULL);"
        "INSERT INTO receipts (at, session_id, provider, model, purpose, prompt_tokens, completion_tokens,"
        " latency_ms, sent_memory_ids, withheld_memory_ids, payload_sha256, cost_usd_estimate, mode) VALUES"
        " ('2026-10-27T10:00:00Z', 's', 'nebius_token_factory', 'm', 'answer', 1, 1, 1, '[]', '[]', 'x', 0, 'live');"
    )
    old.commit()
    old.close()
    log = ReceiptLog(path, now=lambda: datetime(2026, 10, 27, 12, tzinfo=timezone.utc))
    assert [r.status for r in log.today()] == ["answered"]
    log.record(provider="tavily", model="t", purpose="web_search", payload="q", status="no answer (HTTP 503)")
    assert [r.status for r in log.today()] == ["answered", "no answer (HTTP 503)"]
