"""Check the Nebius Token Factory account from the command line.

Lists the NVIDIA Nemotron models visible to the key, then sends one tiny chat
request to each chosen model and prints latency and token usage. The key is read
from NEBIUS_API_KEY and never printed. Standard library only, so it runs with
any Python 3.9+.

Usage: scripts/with-keys.sh python3 scripts/probe_token_factory.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("CONFIDANT_BASE_URL", "https://api.tokenfactory.nebius.com/v1")


def call(path: str, body: dict | None = None) -> dict:
    """GET (no body) or POST JSON to the Token Factory API and return the JSON reply."""
    key = os.environ.get("NEBIUS_API_KEY", "")
    if not key:
        sys.exit("NEBIUS_API_KEY is not set (run through scripts/with-keys.sh)")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method="POST" if body is not None else "GET",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        sys.exit(f"HTTP {exc.code} on {path}: {detail}")


def main() -> None:
    """List Nemotron models and ping the reasoning, fast and Lightning candidates."""
    ids = sorted(m.get("id", "") for m in call("/models").get("data", []))
    nemotron = [i for i in ids if "nemotron" in i.lower()]
    print(f"{len(ids)} models visible, {len(nemotron)} Nemotron:")
    for model_id in nemotron:
        print("  ", model_id)
    reason = os.environ.get("CONFIDANT_MODEL_REASON") or next(
        (i for i in nemotron if "ultra" in i.lower()), None)
    fast = os.environ.get("CONFIDANT_MODEL_FAST") or next(
        (i for i in nemotron if "nano" in i.lower() or "super" in i.lower()), None)
    lightning = next((i for i in nemotron if "lightning" in i.lower()), None)
    for role, model_id in (("reason", reason), ("fast", fast), ("fast-alt", lightning)):
        if not model_id:
            print(f"{role}: no matching Nemotron model")
            continue
        started = time.monotonic()
        reply = call("/chat/completions", {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
            "max_tokens": 512,
            "temperature": 0,
        })
        elapsed = int((time.monotonic() - started) * 1000)
        usage = reply.get("usage", {})
        message = (reply.get("choices") or [{}])[0].get("message", {})
        text = message.get("content") or ""
        thinking = "yes" if (message.get("reasoning_content") or "<think>" in text) else "no"
        print(f"{role}: {model_id} -> {text.strip()[-60:]!r} in {elapsed} ms, "
              f"{usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out tokens, "
              f"reasoning trace: {thinking}")


if __name__ == "__main__":
    main()
