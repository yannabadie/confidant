"""MCP server over stdio: protocol handshake, tools, errors, and a real subprocess."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from confidant.mcp_server import TOOLS, MCPServer

from conftest import PERSONA, Rig

ROOT = Path(__file__).resolve().parent.parent


def rpc(request_id: int | None, method: str, **params: Any) -> str:
    """One JSON-RPC line (a notification when ``request_id`` is None)."""
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
    if request_id is not None:
        message["id"] = request_id
    return json.dumps(message)


def call(request_id: int, tool: str, **arguments: Any) -> str:
    """A tools/call line."""
    return rpc(request_id, "tools/call", name=tool, arguments=arguments)


def converse(server: MCPServer, *lines: str) -> dict[Any, dict]:
    """Feed lines through in-memory pipes and return the responses by id."""
    stdin, stdout = io.StringIO("\n".join(lines) + "\n"), io.StringIO()
    server.serve(stdin, stdout)
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return {response.get("id"): response for response in responses}


def payload(response: dict) -> dict:
    """The structured result of a successful tool call."""
    result = response["result"]
    assert result["isError"] is False, result
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
    return result["structuredContent"]


def test_handshake_and_tool_listing(make_rig: Callable[..., Rig]) -> None:
    server = MCPServer(make_rig().assistant)
    responses = converse(
        server,
        rpc(1, "initialize", protocolVersion="2025-06-18", capabilities={}, clientInfo={"name": "hermes-agent"}),
        rpc(None, "notifications/initialized"),
        rpc(2, "tools/list"),
        rpc(3, "ping"),
    )
    assert set(responses) == {1, 2, 3}
    init = responses[1]["result"]
    assert init["protocolVersion"] == "2025-06-18" and init["serverInfo"]["name"] == "confidant"
    assert init["capabilities"] == {"tools": {"listChanged": False}} and server.client == "hermes-agent"
    names = [tool["name"] for tool in responses[2]["result"]["tools"]]
    assert names == ["memory_search", "memory_record", "memory_correct", "memory_history", "guarded_answer",
                     "daily_brief"]
    assert all(tool["inputSchema"]["type"] == "object" for tool in TOOLS)
    assert responses[3]["result"] == {}


def test_memory_tools_keep_private_values_from_the_client(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    responses = converse(
        MCPServer(rig.assistant),
        call(1, "memory_search", query="dermatologist appointment"),
        call(2, "memory_record", text="Prefers oat milk in tea."),
        call(3, "memory_correct", memory_id=7, new_text="Dermatologist appointment on Wednesday 4 Nov 2026 at 9:00.",
             expected_version=1),
        call(4, "memory_history", memory_id=7),
    )
    found = payload(responses[1])["memories"]
    assert found[0]["cite"] == "m7v1" and found[0]["withheld"] is True
    assert found[0]["text"] == "⟦m7⟧ (private: health appointment, value withheld)"
    recorded = payload(responses[2])["memory"]
    assert (recorded["cite"], recorded["sensitivity"]) == ("m26v1", "personal")
    assert recorded["source_ref"] == "mcp:mcp-client"
    assert payload(responses[3])["memory"]["cite"] == "m7v2"
    versions = payload(responses[4])["versions"]
    assert [v["status"] for v in versions] == ["superseded", "current"]
    assert all("10:30" not in v["text"] and "9:00" not in v["text"] for v in versions)
    assert rig.store.get(7).text.endswith("at 9:00.")


def test_tool_errors_come_back_as_results_the_client_can_read(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    responses = converse(
        MCPServer(rig.assistant),
        call(1, "memory_record", text="Remember my bank PIN is 4821"),
        call(2, "memory_correct", memory_id=7, new_text="Moved.", expected_version=3),
        call(3, "memory_history", memory_id=999),
        call(4, "memory_search", query="x", limit=500),
        call(5, "memory_correct", memory_id="seven", new_text="Moved.", expected_version=1),
    )
    for request_id in range(1, 6):
        assert responses[request_id]["result"]["isError"] is True
    assert "PIN" in responses[1]["result"]["content"][0]["text"]
    assert "version 1, not 3" in responses[2]["result"]["content"][0]["text"]
    assert "m999" in responses[3]["result"]["content"][0]["text"]
    assert len(rig.store.all_current()) == 25


def test_protocol_errors(make_rig: Callable[..., Rig]) -> None:
    responses = converse(
        MCPServer(make_rig().assistant),
        "{not json",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "resources/list"}),
        json.dumps({"jsonrpc": "1.0", "id": 3, "method": "ping"}),
        call(4, "no_such_tool"),
        json.dumps({"jsonrpc": "2.0", "id": 5, "result": {}}),
    )
    assert responses[None]["error"]["code"] == -32700
    assert responses[2]["error"]["code"] == -32601
    assert responses[3]["error"]["code"] == -32600
    assert responses[4]["error"]["code"] == -32602
    assert 5 not in responses


def test_guarded_answer_returns_placeholders_unless_revealed(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    question = call(1, "guarded_answer", question="What's on my plate next week?")
    hidden = payload(converse(MCPServer(rig.assistant), question)[1])
    assert "⟦m7⟧" in hidden["answer"] and "10:30" not in hidden["answer"]
    assert hidden["placeholders"]["⟦m7⟧"] == "private: health appointment"
    assert hidden["citations"] == ["m7v1", "m6v1", "m11v1", "m9v1"] and hidden["mode"] == "live"
    assert hidden["guard"] == "2 private memories never left Confidant"
    assert rig.llm.purposes() == ["answer"]
    shown = payload(converse(MCPServer(rig.assistant, reveal_private=True), question)[1])
    assert "Dermatologist appointment on Tuesday 3 Nov 2026 at 10:30." in shown["answer"]
    assert shown["placeholders"] == {}


def test_daily_brief_tool_returns_the_brief_with_labels_only(make_rig: Callable[..., Rig]) -> None:
    rig = make_rig()
    brief = payload(converse(MCPServer(rig.assistant), call(1, "daily_brief"))[1])
    assert rig.llm.purposes() == ["brief"], "the brief intent runs, not a plain answer"
    assert "[private: health appointment]" in brief["brief"] and "⟦m7⟧" not in brief["brief"]
    assert not any(value in brief["brief"] for value in ("10:30", "Dermatologist", "lead designer"))
    assert brief["brief"].splitlines()[-1].startswith("Private items stay on your device")


def test_serve_mcp_runs_over_real_stdio(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("NEBIUS_API_KEY", "TAVILY_API_KEY")}
    base = [sys.executable, "-m", "confidant", "--home", str(tmp_path), "--replay", str(tmp_path / "none")]
    seeded = subprocess.run([*base, "seed", "--persona", str(PERSONA)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert seeded.returncode == 0, seeded.stderr
    lines = [rpc(1, "initialize", protocolVersion="2024-11-05", clientInfo={"name": "probe"}),
             rpc(None, "notifications/initialized"), call(2, "memory_search", query="half marathon")]
    served = subprocess.run([*base, "serve-mcp"], cwd=ROOT, env=env, input="\n".join(lines) + "\n",
                            capture_output=True, text=True, timeout=60)
    assert served.returncode == 0, served.stderr
    responses = [json.loads(line) for line in served.stdout.splitlines()]
    assert [r["id"] for r in responses] == [1, 2]
    assert responses[0]["result"]["protocolVersion"] == "2024-11-05"
    assert responses[1]["result"]["structuredContent"]["memories"][0]["cite"] == "m3v1"
    assert "NEBIUS_API_KEY is not set" in served.stderr
