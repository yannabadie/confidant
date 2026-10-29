"""Confidant as an MCP server over stdio: JSON-RPC 2.0, one message per line, no SDK.

Tools: ``memory_search``, ``memory_record``, ``memory_correct``,
``memory_history``, ``guarded_answer`` and ``daily_brief``, so Hermes Agent or any
MCP client can use Confidant's memory. Private memories reach the client only as placeholders
with their label, unless the server is started with ``--reveal-private``: an
MCP client may itself be backed by a cloud model.

Client configuration (stdio transport), from a checkout of the repository::

    {
      "mcpServers": {
        "confidant": {
          "command": "python3",
          "args": ["-m", "confidant", "serve-mcp"],
          "cwd": "/path/to/confidant"
        }
      }
    }

``guarded_answer`` calls Nemotron live when NEBIUS_API_KEY is in the server's
environment (``scripts/with-keys.sh`` provides it from the macOS keychain) and
uses replay otherwise.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, TextIO

from .assistant import Assistant, cited_memories, correction_level, delivery_text
from .config import VERSION
from .forms import warm_up
from .guard import PLACEHOLDER_RE, placeholder
from .memory import KINDS, SENSITIVITIES, Memory
from .rules import (
    classify_kind, classify_new_fact, classify_sensitivity, is_availability_question, make_label, stricter,
)
from .scan import find_values

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
# What guarded_answer says about the guard, whatever the question: a count of the private memories a question
# touched would tell a client which of its words a private memory holds.
AGENT_GUARD_NOTE = ("Private memories stay in Confidant: an agent sees at most a placeholder and its category, "
                    "and only in answers about a period of time.")
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = (
    -32700, -32600, -32601, -32602, -32603,
)
_log = logging.getLogger(__name__)


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """A JSON Schema for an object with ``properties``."""
    return {"type": "object", "properties": properties, "required": required}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_search",
        "description": "Full-text search over the user's current memories that may be shared. Private "
                       "memories are not searched: which words match one would tell a client what it says.",
        "inputSchema": _schema({
            "query": {"type": "string", "description": "Words to look for."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 8},
        }, ["query"]),
    },
    {
        "name": "memory_record",
        "description": "Store a new memory with provenance. Credentials (passwords, PINs, card numbers, "
                       "API keys) are refused. Sensitivity, kind and label are inferred locally if omitted; a "
                       "client may raise the sensitivity the device infers, never lower it, and a private "
                       "memory always gets a category label (\"health appointment\").",
        "inputSchema": _schema({
            "text": {"type": "string"},
            "sensitivity": {"type": "string", "enum": list(SENSITIVITIES)},
            "kind": {"type": "string", "enum": [kind for kind in KINDS if kind != "source"]},
            "label": {"type": "string", "description": "2 to 4 words naming the topic; no names, numbers or dates."},
        }, ["text"]),
    },
    {
        "name": "memory_correct",
        "description": "Correct a memory: adds a new version and keeps the old one in its history. "
                       "expected_version must be the version you read, or the call fails with a conflict.",
        "inputSchema": _schema({
            "memory_id": {"type": "integer"},
            "new_text": {"type": "string"},
            "expected_version": {"type": "integer"},
        }, ["memory_id", "new_text", "expected_version"]),
    },
    {
        "name": "memory_history",
        "description": "Every version of a memory with its provenance, oldest first.",
        "inputSchema": _schema({"memory_id": {"type": "integer"}}, ["memory_id"]),
    },
    {
        "name": "guarded_answer",
        "description": "Answer a question from the user's memories through Confidant's privacy guard: "
                       "the cloud model sees placeholders, never private values. The words of a question never "
                       "pick a private memory (private items appear only in time-based answers, where every "
                       "upcoming one is listed, undated). Availability questions (\"Am I free on Tuesday "
                       "morning?\") are checked on the device and list shared items only, since placing a private "
                       "item in time would date it. Read-only.",
        "inputSchema": _schema({"question": {"type": "string"}}, ["question"]),
    },
    {
        "name": "daily_brief",
        "description": "The user's daily brief (the week ahead and open items), safe to deliver: private "
                       "items appear only as their label, e.g. [private: health appointment]. Read-only.",
        "inputSchema": _schema({}, []),
    },
]


class _RpcError(Exception):
    """A JSON-RPC error to send back to the client."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """A JSON-RPC error response."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _string(arguments: Mapping[str, Any], name: str, default: str | None = None) -> str:
    """A non-empty string argument (or ``default`` when it is absent)."""
    value = arguments.get(name)
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{name}' must be a non-empty string.")
    return value


def _integer(arguments: Mapping[str, Any], name: str, default: int | None = None) -> int:
    """An integer argument (or ``default`` when it is absent)."""
    value = arguments.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"'{name}' must be an integer.")
    return value


class MCPServer:
    """Serves Confidant's tools to one MCP client over a line-delimited JSON-RPC stream."""

    def __init__(self, assistant: Assistant, *, reveal_private: bool = False) -> None:
        self.assistant = assistant
        self.reveal_private = reveal_private
        self.client = "mcp-client"
        self._tools: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
            "memory_search": self._memory_search,
            "memory_record": self._memory_record,
            "memory_correct": self._memory_correct,
            "memory_history": self._memory_history,
            "guarded_answer": self._guarded_answer,
            "daily_brief": self._daily_brief,
        }

    def serve(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        """Answer requests line by line until the input ends. The guard's written forms of every date and time
        are generated in the background meanwhile (about ten seconds), so the first tool call does not wait."""
        threading.Thread(target=warm_up, name="forms-warm-up", daemon=True).start()
        source = sys.stdin if stdin is None else stdin
        sink = sys.stdout if stdout is None else stdout
        for line in source:
            if not line.strip():
                continue
            response = self.handle_line(line)
            if response is not None:
                sink.write(json.dumps(response) + "\n")
                sink.flush()

    def handle_line(self, line: str) -> dict[str, Any] | None:
        """Process one line of input; returns the response, or None when none is due."""
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return _error(None, PARSE_ERROR, "Parse error: the line is not valid JSON.")
        return self.handle_message(message)

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        """Process one JSON-RPC message; notifications and stray responses get no reply."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            request_id = message.get("id") if isinstance(message, dict) else None
            return _error(request_id, INVALID_REQUEST, "Invalid request: expected a JSON-RPC 2.0 object.")
        if not isinstance(message.get("method"), str):
            if "result" in message or "error" in message:
                return None
            return _error(message.get("id"), INVALID_REQUEST, "Invalid request: 'method' is missing.")
        if "id" not in message:
            return None
        try:
            result = self._dispatch(message["method"], message.get("params") or {})
        except _RpcError as exc:
            return _error(message["id"], exc.code, str(exc))
        except Exception:  # keep serving: one bad request must not end the session
            _log.exception("MCP request %r failed", message["method"])
            return _error(message["id"], INTERNAL_ERROR, "Internal error; see the server log.")
        return {"jsonrpc": "2.0", "id": message["id"], "result": result}

    def _dispatch(self, method: str, params: Any) -> dict[str, Any]:
        """Run one request method."""
        if not isinstance(params, dict):
            raise _RpcError(INVALID_PARAMS, "'params' must be an object.")
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            return self._call_tool(params)
        raise _RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

    def _initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Negotiate the protocol version and remember the client's name."""
        requested = params.get("protocolVersion")
        info = params.get("clientInfo")
        if isinstance(info, dict) and isinstance(info.get("name"), str) and info["name"].strip():
            self.client = info["name"].strip()[:60]
        return {
            "protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "confidant", "version": VERSION},
            "instructions": "Confidant is the user's private memory. Search before answering personal "
                            "questions, cite memories as m12v2, and treat ⟦m7⟧ placeholders as values "
                            "you are not allowed to see.",
        }

    def _call_tool(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Run a tool; failures the caller can fix come back as an ``isError`` result."""
        name, arguments = params.get("name"), params.get("arguments") or {}
        if name not in self._tools:
            raise _RpcError(INVALID_PARAMS, f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise _RpcError(INVALID_PARAMS, "Tool arguments must be an object.")
        try:
            payload = self._tools[name](arguments)
        except (ValueError, LookupError, RuntimeError) as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        text = json.dumps(payload, ensure_ascii=False)
        return {"content": [{"type": "text", "text": text}], "structuredContent": payload, "isError": False}

    def _view(self, memory: Memory) -> dict[str, Any]:
        """A memory as the client may see it: private text and notes withheld unless revealed."""
        view = memory.to_dict()
        if memory.sensitivity == "private" and not self.reveal_private:
            hidden = f"{placeholder(memory.id)} (private: {memory.label}, value withheld)"
            view.update(text=hidden, note="", withheld=True)
        return view

    def _memory_search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Tool: full-text search."""
        limit = _integer(arguments, "limit", 8)
        if not 1 <= limit <= 25:
            raise ValueError("'limit' must be between 1 and 25.")
        levels = None if self.reveal_private else [level for level in SENSITIVITIES if level != "private"]
        found = self.assistant.store.search(_string(arguments, "query"), limit=limit, sensitivities=levels)
        return {"memories": [self._view(memory) for memory in found]}

    def _memory_record(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Tool: store a new memory. The device decides how private it is: a client may only make it stricter.
        A private memory gets a category label, since its label is sent wherever its value is not; a label the
        client gives another memory is kept only if it holds no value (no date, time, number or name)."""
        text = _string(arguments, "text")
        inferred = classify_new_fact(text)
        offered = _string(arguments, "label", "")
        if offered and classify_sensitivity(offered) == "private":  # the client's own label says what it is:
            inferred = "private"  # "doctor visit" for "Seeing Okafor after my run on Monday" (round 11)
        sensitivity = stricter(_string(arguments, "sensitivity", inferred), inferred)
        label = make_label(text, sensitivity)
        offered = offered or label
        if sensitivity != "private" and not find_values(offered) and len(offered.split()) <= 4:
            label = offered
        memory = self.assistant.store.record(
            text, kind=_string(arguments, "kind", classify_kind(text)), sensitivity=sensitivity,
            source_type="user", source_ref=f"mcp:{self.client}", author=self.client, label=label,
        )
        return {"memory": self._view(memory)}

    def _memory_correct(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Tool: correct a memory, guarded by its expected version. The sensitivity follows the same rule as a
        correction typed in the chat: never lower, and private when the new words look private."""
        memory_id, text = _integer(arguments, "memory_id"), _string(arguments, "new_text")
        sensitivity, label = correction_level(self.assistant.store.get(memory_id), text)
        memory = self.assistant.store.correct(
            memory_id, text, author=self.client, source_ref=f"mcp:{self.client}",
            expected_version=_integer(arguments, "expected_version"), sensitivity=sensitivity, label=label,
        )
        return {"memory": self._view(memory)}

    def _memory_history(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Tool: all versions of a memory. When its latest version is private, every version is withheld: the
        text it had before a correction made it private ("Partner Sam is vegetarian") would otherwise sit next
        to "(private: health)"."""
        versions = self.assistant.store.history(_integer(arguments, "memory_id"))
        latest = versions[-1] if versions else None
        if latest is not None and latest.sensitivity == "private" and not self.reveal_private:
            versions = [replace(memory, sensitivity="private", label=latest.label) for memory in versions]
        return {"versions": [self._view(memory) for memory in versions]}

    def _daily_brief(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Tool: the daily brief as it may leave the device (labels, never private values)."""
        turn = self.assistant.daily_brief(f"mcp:{self.client}")
        return {
            "brief": turn.answer if self.reveal_private else delivery_text(turn, self.assistant.store),
            "citations": [memory.cite() for memory in turn.citations],
            "guard": turn.guard.summary(),
            "mode": turn.mode,
            "notice": turn.notice,
        }

    def _guarded_answer(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Tool: a guarded, read-only answer from memory (an availability check, checked on the device,
        or an answer). Without --reveal-private the client gets the draft, and only the placeholders and
        citations that appear in it."""
        question = _string(arguments, "question")
        intent = "availability" if is_availability_question(question) else "answer"
        turn = self.assistant.handle(question, session_id=f"mcp:{self.client}", intent=intent,
                                     agent=not self.reveal_private)
        prepared = turn.guard.prepared
        withheld = prepared.withheld if prepared else []
        cited = turn.citations if self.reveal_private or prepared is None else cited_memories(turn.draft, prepared)
        in_draft = {int(match.group(1)) for match in PLACEHOLDER_RE.finditer(turn.draft)}
        return {
            "answer": turn.answer if self.reveal_private else turn.draft,
            "placeholders": {} if self.reveal_private else {
                placeholder(memory.id): f"{memory.sensitivity}: {memory.label}" for memory in withheld
                if memory.id in in_draft
            },
            "citations": [memory.cite() for memory in cited],
            "guard": turn.guard.summary() if self.reveal_private else AGENT_GUARD_NOTE,
            "mode": turn.mode,
            "notice": turn.notice,
        }
