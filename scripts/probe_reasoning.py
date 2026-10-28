#!/usr/bin/env python3
"""How the Nemotron models on Token Factory spend their token budget, on Confidant's own prompts.

Plays the six demo turns live on a throwaway memory seeded with the synthetic persona, keeps the exact messages
of every model call, then sends the router's and the brief's calls again with variants: the same budget, four
times the budget, and reasoning switched off through the chat template (``enable_thinking``). For each, it
reports how the call ended (finish_reason), the tokens, whether a reasoning trace came back separately
(``reasoning_content``) or inside the text, and how the answer text begins. Only the synthetic persona is sent.

The key comes from the environment only and is never printed.
Usage: scripts/with-keys.sh .venv/bin/python scripts/probe_reasoning.py [--out probe-reasoning.md] [--repeat N]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from confidant.assistant import Assistant  # noqa: E402
from confidant.budget import Budget  # noqa: E402
from confidant.config import DEMO_PERSONA, env  # noqa: E402
from confidant.guard import Guard  # noqa: E402
from confidant.llm import DEFAULT_BASE_URL, TokenFactory  # noqa: E402
from confidant.memory import MemoryStore  # noqa: E402
from confidant.net import ServiceError  # noqa: E402
from confidant.receipts import ReceiptLog  # noqa: E402
from confidant.replay import ReplayBook  # noqa: E402

QUESTIONS = (
    "What's on my plate next week?",
    "My dermatologist appointment moved to Wednesday 4 Nov at 9:00.",
    "Find a 10-day taper plan for my half marathon.",
    "Draft a birthday message for Léa.",
    "Remember my bank PIN is 4821.",
    "Run my daily brief.",
)


class Capturing:
    """A chat client that passes every call to the real one and keeps what was asked and how it ended."""

    def __init__(self, inner: TokenFactory) -> None:
        self.inner = inner
        self.calls: list[dict] = []

    def chat(self, model, messages, *, max_tokens, temperature, purpose, thinking=True):
        result = self.inner.chat(model, messages, max_tokens=max_tokens, temperature=temperature, purpose=purpose,
                                 thinking=thinking)
        self.calls.append({
            "purpose": purpose, "model": model, "messages": [dict(m) for m in messages],
            "max_tokens": max_tokens, "temperature": temperature, "thinking": thinking,
            "finish_reason": result.finish_reason, "completion_tokens": result.completion_tokens,
        })
        return result

    def list_models(self):
        return self.inner.list_models()


def send(client: TokenFactory, call: dict, *, max_tokens: int, thinking_off: bool) -> dict:
    """One raw chat completion with the captured messages; returns what the probe reports."""
    payload = {
        "model": call["model"], "messages": call["messages"], "max_tokens": max_tokens,
        "temperature": call["temperature"],
    }
    if thinking_off:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        data, latency_ms = client._request("POST", "/chat/completions", call["purpose"], payload)
    except ServiceError as exc:
        return {"error": str(exc)[:200]}
    choice = data["choices"][0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    usage = data.get("usage") or {}
    out = {
        "finish": choice.get("finish_reason"), "prompt": usage.get("prompt_tokens"),
        "completion": usage.get("completion_tokens"), "ms": latency_ms, "reasoning_chars": len(reasoning),
        "think_tags": "<think>" in content.lower() or "</think>" in content.lower(),
        "head": content[:140].replace("\n", " / "),
    }
    if call["purpose"] == "route":
        try:
            out["json"] = json.loads(content.strip()).get("intent", "?")
        except (ValueError, AttributeError):
            out["json"] = "invalid"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "probe-reasoning.md")
    parser.add_argument("--repeat", type=int, default=1, help="how many times to send each variant")
    args = parser.parse_args()
    key = env("NEBIUS_API_KEY")
    if not key:
        print("NEBIUS_API_KEY is not set (run through scripts/with-keys.sh)", file=sys.stderr)
        return 2
    client = TokenFactory(key, base_url=env("CONFIDANT_BASE_URL") or DEFAULT_BASE_URL)
    capture = Capturing(client)
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "memory.db")
        store.load_seed(DEMO_PERSONA)
        assistant = Assistant(store, Guard(), ReceiptLog(Path(tmp) / "receipts.db"), Budget(), capture, None,
                              ReplayBook())
        for question in QUESTIONS:
            assistant.handle(question, session_id="probe")
        models = dict(assistant.models or {})
    lines = [
        f"# Reasoning probe, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        f"Models: reason `{models.get('reason')}`, fast `{models.get('fast')}`.",
        "",
        "## Calls made by the six demo turns",
        "",
        "| purpose | model | reasoning | max_tokens | finish | completion tokens |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for call in capture.calls:
        lines.append(f"| {call['purpose']} | {call['model'].rsplit('/', 1)[-1]} | "
                     f"{'on' if call['thinking'] else 'off'} | {call['max_tokens']} | "
                     f"{call['finish_reason']} | {call['completion_tokens']} |")
    lines += ["", "## Variants", "",
              "| call | variant | finish | completion | ms | reasoning_content chars | tags in text | route JSON "
              "| text begins |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    picked = [(f"{c['purpose']} #{i}", c) for i, c in enumerate(capture.calls) if c["purpose"] in ("route", "brief")]
    for label, call in picked:
        budget, sent_off = call["max_tokens"], not call["thinking"]
        variants = [("as sent", budget, sent_off), ("budget x4", budget * 4, sent_off)]
        if not sent_off:
            variants.append(("thinking off", budget, True))
        for variant, max_tokens, off in variants:
            for _ in range(max(1, args.repeat)):
                r = send(client, call, max_tokens=max_tokens, thinking_off=off)
                if "error" in r:
                    lines.append(f"| {label} | {variant} ({max_tokens}) | error: {r['error']} | | | | | | |")
                    continue
                head = r["head"].replace("|", "\\|")
                lines.append(
                    f"| {label} | {variant} ({max_tokens}) | {r['finish']} | {r['completion']} | {r['ms']} | "
                    f"{r['reasoning_chars']} | {'yes' if r['think_tags'] else 'no'} | {r.get('json', '')} | "
                    f"{head} |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
