"""Settings from the environment, and the repository's data paths.

An empty or blank variable counts as unset, so ``TAVILY_API_KEY=""`` behaves
exactly like a missing key.
"""
from __future__ import annotations

import os
from pathlib import Path

VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEMO_PERSONA = DATA_DIR / "demo_persona.json"
REPLAY_DIR = DATA_DIR / "replay"


def env(name: str) -> str | None:
    """The stripped value of environment variable ``name``, or None if unset or blank."""
    value = os.environ.get(name, "").strip()
    return value or None


def env_int(name: str, default: int) -> int:
    """A non-negative integer from the environment, ``default`` when unset or blank.

    Raises ValueError with the variable's name when the value is not a whole number.
    """
    raw = env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")
    return value
