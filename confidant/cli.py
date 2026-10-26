"""Command line: ``python -m confidant chat|brief|seed|receipts|serve-mcp``.

Keys come only from the environment (``NEBIUS_API_KEY``, ``TAVILY_API_KEY``; an
empty value counts as unset). Without a Nebius key, answers come from the replay
recordings when there are any, and from local memory otherwise.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TextIO

from .assistant import Assistant, Turn
from .budget import Budget
from .config import DEMO_PERSONA, REPLAY_DIR, env
from .guard import Guard
from .llm import DEFAULT_BASE_URL, TokenFactory
from .mcp_server import MCPServer
from .memory import MemoryStore
from .receipts import ReceiptLog
from .replay import ReplayBook
from .search import Tavily

SESSION_ID = "cli"
DEFAULT_HOME = Path.home() / ".confidant"


def _parser() -> argparse.ArgumentParser:
    """The argument parser for all commands."""
    parser = argparse.ArgumentParser(
        prog="python -m confidant",
        description="Confidant: a personal AI that keeps your secrets and shows its receipts.",
    )
    parser.add_argument(
        "--home", type=Path, default=Path(env("CONFIDANT_HOME") or DEFAULT_HOME),
        help="folder holding memory.db and receipts.db (default: $CONFIDANT_HOME or ~/.confidant)",
    )
    parser.add_argument(
        "--replay", type=Path, default=REPLAY_DIR,
        help="replay file or folder used when no live model is available (default: data/replay)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="talk to Confidant, one message per line")
    chat.add_argument("--record", type=Path, metavar="FILE", help="also save live model outputs to this replay file")
    commands.add_parser("brief", help="print the daily brief")
    seed = commands.add_parser("seed", help="load a persona (default: the demo persona) into an empty memory")
    seed.add_argument("--persona", type=Path, default=DEMO_PERSONA)
    commands.add_parser("receipts", help="list today's outbound calls")
    mcp = commands.add_parser("serve-mcp", help="serve the MCP tools over stdio")
    mcp.add_argument("--reveal-private", action="store_true", help="let the MCP client see private values")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit status."""
    args = _parser().parse_args(argv)
    commands = {"chat": _chat, "brief": _brief, "seed": _seed, "receipts": _receipts, "serve-mcp": _serve_mcp}
    try:
        return commands[args.command](args)
    except KeyboardInterrupt:
        return 130


def _build(args: argparse.Namespace, recorder: ReplayBook | None = None) -> Assistant:
    """An assistant over the memory in ``--home``, explaining on stderr what is not live."""
    store = MemoryStore(args.home / "memory.db")
    if not store.all_current():
        print("Memory is empty; run `python -m confidant seed` to load the demo persona.", file=sys.stderr)
    replay = ReplayBook.load(args.replay)
    nebius_key, tavily_key = env("NEBIUS_API_KEY"), env("TAVILY_API_KEY")
    if nebius_key is None:
        where = f"answers come from replay ({len(replay)} recorded turns in {args.replay})" if replay else (
            f"there is no replay data in {args.replay}, so answers are built from local memory only"
        )
        print(f"NEBIUS_API_KEY is not set: {where}.", file=sys.stderr)
    elif tavily_key is None:
        print("TAVILY_API_KEY is not set: web search is off.", file=sys.stderr)
    llm = TokenFactory(nebius_key, base_url=env("CONFIDANT_BASE_URL") or DEFAULT_BASE_URL) if nebius_key else None
    search = Tavily(tavily_key) if tavily_key else None
    receipts = ReceiptLog(args.home / "receipts.db")
    return Assistant(store, Guard(), receipts, Budget(), llm, search, replay, recorder=recorder)


def _print_turn(turn: Turn, out: TextIO) -> None:
    """Show an answer with its sources, guard summary, actions and notices."""
    out.write(f"\n{turn.answer}\n")
    if turn.citations:
        sources = " | ".join(f"{m.cite()} · {m.source_type} · {m.created_at[:10]}" for m in turn.citations)
        out.write(f"  sources: {sources}\n")
    out.write(f"  guard: {turn.guard.summary()} · mode: {turn.mode}\n")
    for action in turn.actions:
        out.write(f"  did: {action}\n")
    if turn.notice:
        out.write(f"  note: {turn.notice}\n")
    out.flush()


def _messages(stream: TextIO) -> Iterator[str]:
    """Non-empty input lines until end of input or "quit"; prompts on stderr when interactive."""
    interactive = stream.isatty()
    while True:
        if interactive:
            print("you> ", end="", file=sys.stderr, flush=True)
        line = stream.readline()
        if not line or line.strip().lower() in ("quit", "exit"):
            return
        if line.strip():
            yield line.strip()


def _chat(args: argparse.Namespace) -> int:
    """Interactive (or piped) conversation."""
    recorder = ReplayBook.load(args.record) if args.record else None
    assistant = _build(args, recorder)
    for message in _messages(sys.stdin):
        _print_turn(assistant.handle(message, session_id=SESSION_ID), sys.stdout)
    if recorder is not None:
        recorder.save(args.record)
        print(f"Saved {len(recorder)} recorded turns to {args.record}.", file=sys.stderr)
    return 0


def _brief(args: argparse.Namespace) -> int:
    """Print the daily brief."""
    _print_turn(_build(args).daily_brief(SESSION_ID), sys.stdout)
    return 0


def _seed(args: argparse.Namespace) -> int:
    """Load the persona into an empty memory database."""
    store = MemoryStore(args.home / "memory.db")
    existing = store.all_current()
    if existing:
        print(f"{args.home / 'memory.db'} already holds {len(existing)} memories; "
              "use another --home to seed a fresh one.", file=sys.stderr)
        return 1
    loaded = store.load_seed(args.persona)
    as_of = store.as_of()
    print(f"Seeded {len(loaded)} memories from {args.persona}" + (f" (snapshot date {as_of})." if as_of else "."))
    return 0


def _receipts(args: argparse.Namespace) -> int:
    """List today's outbound calls (hashes, never payloads) and the day's totals."""
    log = ReceiptLog(args.home / "receipts.db")
    rows = log.today()
    if not rows:
        print("No outbound calls today.")
    for row in rows:
        print(
            f"{row.at[11:19]}  {row.mode:<6}  {row.provider:<20}  {row.purpose:<10}  "
            f"{row.prompt_tokens:>6}+{row.completion_tokens:<5} tok  {row.latency_ms:>6} ms  "
            f"sha256 {row.payload_sha256[:12]}  sent {row.sent_memory_ids}  withheld {row.withheld_memory_ids}  "
            f"{row.model}"
        )
    print(log.today_summary().line())
    return 0


def _serve_mcp(args: argparse.Namespace) -> int:
    """Serve MCP over stdio (stdout carries the protocol; notices go to stderr)."""
    MCPServer(_build(args), reveal_private=args.reveal_private).serve()
    return 0
