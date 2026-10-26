"""Budget: daily tokens across sessions, requests per session, env defaults, day rollover."""
from __future__ import annotations

from datetime import date

import pytest

from confidant.budget import Budget


def test_defaults_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONFIDANT_DAILY_TOKENS", raising=False)
    monkeypatch.setenv("CONFIDANT_SESSION_REQUESTS", "")  # empty means unset
    budget = Budget()
    assert (budget.daily_tokens, budget.per_session_requests) == (300_000, 25)
    monkeypatch.setenv("CONFIDANT_DAILY_TOKENS", "5000")
    monkeypatch.setenv("CONFIDANT_SESSION_REQUESTS", " 3 ")
    assert (Budget().daily_tokens, Budget().per_session_requests) == (5000, 3)


def test_invalid_environment_values_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONFIDANT_DAILY_TOKENS", "lots")
    with pytest.raises(ValueError, match="CONFIDANT_DAILY_TOKENS"):
        Budget()


def test_each_session_has_its_own_request_limit() -> None:
    budget = Budget(daily_tokens=10_000, per_session_requests=2)
    for _ in range(2):
        assert budget.check("a") == (True, "")
        budget.consume("a", 10)
    allowed, reason = budget.check("a")
    assert not allowed and "2 live requests" in reason and "replay" in reason
    assert budget.check("b") == (True, "")
    assert budget.remaining("a") == (9_980, 0)


def test_daily_tokens_are_shared_by_all_sessions() -> None:
    budget = Budget(daily_tokens=100, per_session_requests=10)
    budget.consume("a", 60)
    budget.consume("b", 50)
    allowed, reason = budget.check("c")
    assert not allowed and "100 tokens" in reason and "replay" in reason
    with pytest.raises(ValueError):
        budget.consume("c", -1)


def test_counters_reset_when_the_utc_day_changes() -> None:
    day = {"today": date(2026, 11, 3)}
    budget = Budget(daily_tokens=100, per_session_requests=1, today=lambda: day["today"])
    budget.consume("a", 100)
    assert not budget.check("a")[0]
    day["today"] = date(2026, 11, 4)
    assert budget.check("a") == (True, "")
    assert budget.remaining("a") == (100, 1)
