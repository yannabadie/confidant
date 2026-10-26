"""Usage caps for the live models: tokens per day (all sessions) and requests per session.

A request is one user turn that reached the cloud. Counters reset at midnight
UTC. When :meth:`Budget.check` refuses, the assistant switches to replay mode
and tells the user why.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import date, datetime, timezone

from .config import env_int

DEFAULT_DAILY_TOKENS = 300_000
DEFAULT_SESSION_REQUESTS = 25


def _utc_today() -> date:
    """Today's date in UTC."""
    return datetime.now(timezone.utc).date()


class Budget:
    """Thread-safe token and request budget.

    Unset limits come from ``CONFIDANT_DAILY_TOKENS`` (default 300000) and
    ``CONFIDANT_SESSION_REQUESTS`` (default 25).
    """

    def __init__(
        self,
        daily_tokens: int | None = None,
        per_session_requests: int | None = None,
        *,
        today: Callable[[], date] = _utc_today,
    ) -> None:
        self.daily_tokens = (
            env_int("CONFIDANT_DAILY_TOKENS", DEFAULT_DAILY_TOKENS) if daily_tokens is None else daily_tokens
        )
        self.per_session_requests = (
            env_int("CONFIDANT_SESSION_REQUESTS", DEFAULT_SESSION_REQUESTS)
            if per_session_requests is None
            else per_session_requests
        )
        self._today = today
        self._day = today()
        self._tokens = 0
        self._requests: dict[str, int] = {}
        self._lock = threading.Lock()

    def check(self, session_id: str) -> tuple[bool, str]:
        """Whether ``session_id`` may make a live request now, and if not, why."""
        with self._lock:
            self._roll_over()
            if self._tokens >= self.daily_tokens:
                return False, (
                    f"Today's live budget of {self.daily_tokens:,} tokens is used up, "
                    "so Confidant switched to replay."
                )
            if self._requests.get(session_id, 0) >= self.per_session_requests:
                return False, (
                    f"This session has used its {self.per_session_requests} live requests, "
                    "so Confidant switched to replay."
                )
            return True, ""

    def consume(self, session_id: str, tokens: int) -> None:
        """Count one live request of ``tokens`` tokens for ``session_id``."""
        if tokens < 0:
            raise ValueError("tokens must not be negative")
        with self._lock:
            self._roll_over()
            self._tokens += tokens
            self._requests[session_id] = self._requests.get(session_id, 0) + 1

    def remaining(self, session_id: str) -> tuple[int, int]:
        """Tokens left today and live requests left for ``session_id``."""
        with self._lock:
            self._roll_over()
            return (
                max(self.daily_tokens - self._tokens, 0),
                max(self.per_session_requests - self._requests.get(session_id, 0), 0),
            )

    def _roll_over(self) -> None:
        """Reset the counters when the UTC day changes (caller holds the lock)."""
        today = self._today()
        if today != self._day:
            self._day, self._tokens, self._requests = today, 0, {}
