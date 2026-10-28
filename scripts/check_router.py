#!/usr/bin/env python3
"""The router on fixed cases, live: does the fast model pick the intent its prompt defines?

Each message is sent as Confidant sends it (masked: dates, times, numbers, e-mail addresses and names
replaced by their kind), with the router prompt, the router's budget and reasoning off. The cases were
written before any run. In the app, the router only sees messages the local rules cannot route and that carry
no value; here every case goes to it, so the check covers each intent the prompt describes.

The key comes from the environment only and is never printed.
Usage: scripts/with-keys.sh .venv/bin/python scripts/check_router.py [--out docs/router-check.md]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from confidant.assistant import (  # noqa: E402
    FAST_MAX_TOKENS, FAST_TEMPERATURE, ROUTER_SYSTEM_PROMPT, THINKING, masked, parse_route,
)
from confidant.config import env  # noqa: E402
from confidant.llm import DEFAULT_BASE_URL, TokenFactory, discover_models  # noqa: E402
from confidant.net import ServiceError  # noqa: E402

CASES = (
    ("What's on my plate next week?", "answer"),
    ("Draft a birthday message for Léa.", "answer"),
    ("What did my coach say about race week?", "answer"),
    ("What should I get Léa for her birthday?", "answer"),
    ("Which restaurant did we pick for dinner with friends?", "answer"),
    ("How is my training going?", "answer"),
    ("Look up how to treat a blister before a race.", "search"),
    ("What's the world record for the half marathon?", "search"),
    ("Search the web for carb-loading tips.", "search"),
    ("Find a recipe for a pre-race pasta dinner.", "search"),
    ("Am I free on Tuesday afternoon?", "availability"),
    ("Do I have anything on Saturday morning?", "availability"),
    ("Run my daily brief.", "brief"),
    ("I prefer aisle seats on long train trips.", "remember"),
    ("Léa's favourite flowers are actually tulips, not peonies.", "correct"),
    ("Our weekly review moved to Thursdays.", "correct"),
)


def route_once(client: TokenFactory, model: str, message: str) -> tuple[str, int]:
    """The intent the router picks for ``message`` ("cut off", "invalid" or "error" otherwise) and the latency."""
    try:
        result = client.chat(
            model, [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}, {"role": "user", "content": masked(message)}],
            max_tokens=FAST_MAX_TOKENS, temperature=FAST_TEMPERATURE, purpose="route", thinking=THINKING["fast"],
        )
    except ServiceError:
        return "error", 0
    route = None if result.truncated else parse_route(result.text)
    return (route.intent if route else ("cut off" if result.truncated else "invalid")), result.latency_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "router-check.md")
    parser.add_argument("--runs", type=int, default=3, help="how many times each case is sent (default 3)")
    args = parser.parse_args()
    key = env("NEBIUS_API_KEY")
    if not key:
        print("NEBIUS_API_KEY is not set (run through scripts/with-keys.sh)", file=sys.stderr)
        return 2
    client = TokenFactory(key, base_url=env("CONFIDANT_BASE_URL") or DEFAULT_BASE_URL)
    model = discover_models(client)["fast"]
    runs = max(1, args.runs)
    rows, right, steady, latencies = [], 0, 0, []
    for message, expected in CASES:
        got = [route_once(client, model, message) for _ in range(runs)]
        latencies += [ms for _, ms in got if ms]
        hits = sum(intent == expected for intent, _ in got)
        right += hits
        steady += hits == runs
        mark = "yes" if hits == runs else f"**{hits} of {runs}**"
        rows.append(f"| {message} | {expected} | {', '.join(intent for intent, _ in got)} | {mark} |")
    client.close()
    mean = round(sum(latencies) / len(latencies)) if latencies else 0
    text = "\n".join([
        f"# Router check, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        f"Model `{model}`, reasoning {'on' if THINKING['fast'] else 'off'}, max {FAST_MAX_TOKENS} tokens, "
        f"temperature {FAST_TEMPERATURE}; each case sent {runs} times. Produced by `scripts/check_router.py`; the "
        "cases were written before any run.",
        "",
        f"**{right} of {len(CASES) * runs}** answers as expected; {steady} of {len(CASES)} cases right every "
        f"time; {mean} ms per call on average. The same message can get a different route from one call to "
        "the next, even at temperature 0.",
        "",
        "In the app, the router only sees messages that the local rules cannot route, that carry no date, time or "
        "amount, and that are not questions about a period of the user's time (\"What's on my plate next week?\" "
        "is routed on the device). When it picks a web search for a question about the "
        "user's own life, the answer still draws on memory, and the query, with names and values removed, goes "
        "to the web only if web search is on.",
        "",
        "| message | expected | router, each call | right |",
        "| --- | --- | --- | --- |",
        *rows,
    ]) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
