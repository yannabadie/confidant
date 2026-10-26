"""Receipts: one row per outbound call, holding a hash of the payload and never the payload.

Live calls and replayed ones are both logged, labelled by ``mode``, so the
receipts tab always tells the truth about what reached a cloud provider.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODES = ("live", "replay")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
    at                  TEXT    NOT NULL,
    session_id          TEXT    NOT NULL,
    provider            TEXT    NOT NULL,
    model               TEXT    NOT NULL,
    purpose             TEXT    NOT NULL,
    prompt_tokens       INTEGER NOT NULL,
    completion_tokens   INTEGER NOT NULL,
    latency_ms          INTEGER NOT NULL,
    sent_memory_ids     TEXT    NOT NULL,
    withheld_memory_ids TEXT    NOT NULL,
    payload_sha256      TEXT    NOT NULL,
    cost_usd_estimate   REAL    NOT NULL,
    mode                TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS receipts_by_time ON receipts (at);
"""


def payload_sha256(payload: Any) -> str:
    """SHA-256 hex digest of a payload: bytes as they are, text as UTF-8, anything
    else as canonical JSON (sorted keys, no extra spaces)."""
    if isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Receipt:
    """One outbound call (or one replayed call when ``mode`` is ``"replay"``)."""

    at: str
    provider: str
    model: str
    purpose: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    sent_memory_ids: list[int]
    withheld_memory_ids: list[int]
    payload_sha256: str
    cost_usd_estimate: float
    mode: str
    session_id: str = ""


@dataclass(frozen=True)
class ReceiptSummary:
    """Totals for one day of receipts.

    ``memories_kept_local`` counts memories withheld from at least one call and
    sent in none: they never left Confidant that day.
    """

    calls: int
    live_calls: int
    replay_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd_estimate: float
    memories_kept_local: int

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def line(self) -> str:
        """One-line human summary, e.g. for the daily brief. Replayed calls sent nothing,
        so they are not counted as outbound."""
        calls = f"{self.live_calls} outbound call{'' if self.live_calls == 1 else 's'}"
        if self.replay_calls:
            calls += f", {self.replay_calls} replayed"
        kept = self.memories_kept_local
        return (
            f"Receipts today: {calls} · {self.total_tokens:,} tokens · ≈${self.cost_usd_estimate:.4f} · "
            f"{kept} withheld memor{'y' if kept == 1 else 'ies'} never left Confidant"
        )


def _utc_now() -> datetime:
    """The current UTC time."""
    return datetime.now(timezone.utc)


class ReceiptLog:
    """Append-only SQLite log of outbound calls (``path`` or ``":memory:"``)."""

    def __init__(self, path: str | os.PathLike[str] = ":memory:", *, now: Callable[[], datetime] = _utc_now) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._now = now
        with self._lock:
            self._db.executescript(_SCHEMA)

    def record(
        self,
        *,
        provider: str,
        model: str,
        purpose: str,
        payload: Any,
        mode: str = "live",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: int = 0,
        sent_memory_ids: Iterable[int] = (),
        withheld_memory_ids: Iterable[int] = (),
        cost_usd_estimate: float = 0.0,
        session_id: str = "",
    ) -> Receipt:
        """Log one call. ``payload`` (what was or would have been sent) is hashed and discarded."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
        receipt = Receipt(
            at=self._now().astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            provider=provider, model=model, purpose=purpose,
            prompt_tokens=int(prompt_tokens), completion_tokens=int(completion_tokens),
            latency_ms=int(latency_ms),
            sent_memory_ids=sorted(set(sent_memory_ids)),
            withheld_memory_ids=sorted(set(withheld_memory_ids)),
            payload_sha256=payload_sha256(payload),
            cost_usd_estimate=round(float(cost_usd_estimate), 6), mode=mode, session_id=session_id,
        )
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO receipts (at, session_id, provider, model, purpose, prompt_tokens, "
                "completion_tokens, latency_ms, sent_memory_ids, withheld_memory_ids, payload_sha256, "
                "cost_usd_estimate, mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    receipt.at, receipt.session_id, receipt.provider, receipt.model, receipt.purpose,
                    receipt.prompt_tokens, receipt.completion_tokens, receipt.latency_ms,
                    json.dumps(receipt.sent_memory_ids), json.dumps(receipt.withheld_memory_ids),
                    receipt.payload_sha256, receipt.cost_usd_estimate, receipt.mode,
                ),
            )
        return receipt

    def today(self, session_id: str | None = None) -> list[Receipt]:
        """Today's receipts (UTC), oldest first, optionally for one session only."""
        sql = "SELECT * FROM receipts WHERE substr(at, 1, 10) = ?"
        params: list[Any] = [self._now().astimezone(timezone.utc).date().isoformat()]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY seq", params).fetchall()
        return [_to_receipt(row) for row in rows]

    def today_summary(self, session_id: str | None = None) -> ReceiptSummary:
        """Counts, tokens, estimated cost and memories that never left, for today."""
        rows = self.today(session_id)
        sent = {memory_id for row in rows for memory_id in row.sent_memory_ids}
        withheld = {memory_id for row in rows for memory_id in row.withheld_memory_ids}
        live = sum(row.mode == "live" for row in rows)
        return ReceiptSummary(
            calls=len(rows), live_calls=live, replay_calls=len(rows) - live,
            prompt_tokens=sum(row.prompt_tokens for row in rows),
            completion_tokens=sum(row.completion_tokens for row in rows),
            cost_usd_estimate=round(sum(row.cost_usd_estimate for row in rows), 6),
            memories_kept_local=len(withheld - sent),
        )

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._db.close()


def _to_receipt(row: sqlite3.Row) -> Receipt:
    """Build a Receipt from a database row."""
    return Receipt(
        at=row["at"], provider=row["provider"], model=row["model"], purpose=row["purpose"],
        prompt_tokens=row["prompt_tokens"], completion_tokens=row["completion_tokens"],
        latency_ms=row["latency_ms"], sent_memory_ids=json.loads(row["sent_memory_ids"]),
        withheld_memory_ids=json.loads(row["withheld_memory_ids"]),
        payload_sha256=row["payload_sha256"], cost_usd_estimate=row["cost_usd_estimate"],
        mode=row["mode"], session_id=row["session_id"],
    )
