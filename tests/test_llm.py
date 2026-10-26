"""TokenFactory over a mocked transport, reasoning-trace stripping and model selection."""
from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from confidant.llm import (
    DEFAULT_BASE_URL,
    ModelSelectionError,
    TokenFactory,
    discover_models,
    estimate_cost,
    pick_models,
    strip_reasoning,
)
from confidant.net import ServiceError

KEY = "test-key"
MESSAGES = [{"role": "user", "content": "Reply with: ready"}]
# The NVIDIA text models shown in the Token Factory catalog (October 2026).
CATALOG = [
    "nvidia/Nemotron-3-Ultra-550b-a55b", "nvidia/Nemotron-3-Ultra-550b-a55b-serverless",
    "nvidia/Nemotron-3-Super-120b-a12b", "nvidia/Nemotron-3-Super-120b-a12b-serverless",
    "nvidia/Nemotron-3-Nano-30B-A3B", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "nvidia/Nemotron-3-Nano-Omni",
    "nvidia/Nemotron-3.5-Lightning", "nvidia/Nemotron-3.5-Lightning-serverless",
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
]


def completion(content: str | None, **message: str) -> dict:
    """An OpenAI-style chat completion body."""
    reply = {"role": "assistant", "content": content, **message}
    return {
        "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
        "choices": [{"index": 0, "finish_reason": "stop", "message": reply}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 340, "total_tokens": 352},
    }


def client(handler: Callable[[httpx.Request], httpx.Response]) -> TokenFactory:
    """A TokenFactory whose HTTP calls go to ``handler`` (no retry delay)."""
    return TokenFactory(KEY, http=httpx.Client(transport=httpx.MockTransport(handler)), retry_delay=0)


@pytest.fixture(autouse=True)
def no_model_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests start without model overrides in the environment."""
    monkeypatch.delenv("CONFIDANT_MODEL_REASON", raising=False)
    monkeypatch.delenv("CONFIDANT_MODEL_FAST", raising=False)


def test_chat_sends_an_openai_compatible_request_and_reads_usage() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion("ready"))

    result = client(handler).chat("nvidia/Nemotron-3-Ultra-550b-a55b", MESSAGES, max_tokens=64,
                                  temperature=0.0, purpose="probe")
    request = seen[0]
    assert str(request.url) == f"{DEFAULT_BASE_URL}/chat/completions" and request.method == "POST"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content) == {"model": "nvidia/Nemotron-3-Ultra-550b-a55b", "messages": MESSAGES,
                                           "max_tokens": 64, "temperature": 0.0}
    assert (result.text, result.prompt_tokens, result.completion_tokens) == ("ready", 12, 340)
    assert result.model == "nvidia/Nemotron-3-Ultra-550b-a55b" and result.latency_ms >= 0


def test_reasoning_content_field_is_never_surfaced() -> None:
    body = completion("The answer.", reasoning_content="Private chain of thought about ⟦m7⟧.")
    result = client(lambda request: httpx.Response(200, json=body)).chat("m", MESSAGES)
    assert result.text == "The answer."
    assert result.completion_tokens == 340  # reasoning tokens still count


def test_inline_think_block_is_stripped() -> None:
    body = completion("<think>\nThe user wants a word.\n</think>\n\nready")
    assert client(lambda request: httpx.Response(200, json=body)).chat("m", MESSAGES).text == "ready"


@pytest.mark.parametrize(
    ("raw", "answer"),
    [
        ("<think>a</think>answer", "answer"),
        ("<THINK>a</THINK> <think>b</think> answer", "answer"),
        ("reasoning with no opening tag</think>answer", "answer"),
        ("<think>outer <think>inner</think> still thinking</think> answer", "answer"),
        ("<think>cut off before any answer", ""),
        ("plain answer", "plain answer"),
    ],
)
def test_strip_reasoning_variants(raw: str, answer: str) -> None:
    assert strip_reasoning(raw) == answer


def test_null_content_gives_an_empty_answer_with_the_finish_reason() -> None:
    body = completion(None, reasoning_content="long thoughts")
    body["choices"][0]["finish_reason"] = "length"
    result = client(lambda request: httpx.Response(200, json=body)).chat("m", MESSAGES)
    assert (result.text, result.finish_reason) == ("", "length")


def test_one_retry_on_rate_limit_then_success() -> None:
    replies = iter([httpx.Response(429, json={"error": "slow down"}), httpx.Response(200, json=completion("ok"))])
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return next(replies)

    assert client(handler).chat("m", MESSAGES).text == "ok"
    assert len(calls) == 2


def test_server_errors_fail_after_one_retry() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="unavailable")

    with pytest.raises(ServiceError, match=r"server error \(HTTP 503\).*after one retry"):
        client(handler).chat("m", MESSAGES, purpose="answer")
    assert len(calls) == 2


def test_auth_errors_name_the_variable_and_never_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"bad key {KEY}"}})

    with pytest.raises(ServiceError) as caught:
        client(handler).chat("m", MESSAGES)
    assert "NEBIUS_API_KEY" in str(caught.value) and KEY not in str(caught.value)


def test_other_client_errors_show_a_short_detail_without_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": f"max_tokens too large for {KEY}"}})

    with pytest.raises(ServiceError, match=r"HTTP 400\): max_tokens too large for \*\*\*"):
        client(handler).chat("m", MESSAGES)


def test_network_failures_are_retried_then_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(ServiceError, match="could not be reached"):
        client(handler).chat("m", MESSAGES)


def test_unexpected_response_shape_is_a_clear_error() -> None:
    with pytest.raises(ServiceError, match="unexpected response shape"):
        client(lambda request: httpx.Response(200, json={"choices": []})).chat("m", MESSAGES)


def test_list_models_reads_the_catalog() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.path.endswith("/v1/models")
        return httpx.Response(200, json={"object": "list", "data": [{"id": i} for i in reversed(CATALOG)]})

    assert client(handler).list_models() == sorted(CATALOG)


def test_pick_models_ranks_the_live_catalog() -> None:
    chosen = pick_models(CATALOG)
    assert chosen["reason"] == "nvidia/Nemotron-3-Ultra-550b-a55b"
    assert chosen["fast"] in {"nvidia/Nemotron-3-Nano-30B-A3B", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"}
    assert TokenFactory.pick_models(CATALOG) == chosen


def test_pick_models_falls_back_through_the_preferences() -> None:
    no_ultra_nano = [i for i in CATALOG if "Ultra-550b" not in i and "Nano-30B" not in i]
    assert pick_models(no_ultra_nano) == {"reason": "nvidia/Nemotron-3-Super-120b-a12b",
                                          "fast": "nvidia/Nemotron-3.5-Lightning"}
    assert pick_models(["nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", "nvidia/Nemotron-3-Super-120b-a12b-serverless"]) == {
        "reason": "nvidia/Nemotron-3-Super-120b-a12b-serverless",
        "fast": "nvidia/Nemotron-3-Super-120b-a12b-serverless",
    }


def test_pick_models_skips_omni_vl_and_kvoff_and_ignores_the_prefix() -> None:
    ids = ["Nemotron-3-Ultra-550b-a55b-kvoff", "nemotron-3-ultra-550b-a55b", "Nemotron-3.5-Super-VL",
           "Nemotron-3-Nano-Omni", "nemotron-3-nano-30b-a3b"]
    assert pick_models(ids) == {"reason": "nemotron-3-ultra-550b-a55b", "fast": "nemotron-3-nano-30b-a3b"}


def test_environment_overrides_win_and_empty_values_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONFIDANT_MODEL_REASON", "custom/reasoner")
    monkeypatch.setenv("CONFIDANT_MODEL_FAST", "")
    assert pick_models(CATALOG)["reason"] == "custom/reasoner"
    assert "Nano-30B" in pick_models(CATALOG)["fast"]


def test_no_matching_model_lists_what_is_available() -> None:
    with pytest.raises(ModelSelectionError) as caught:
        pick_models(["meta/llama-3.3-70b", "nvidia/Nemotron-3-Nano-Omni"])
    message = str(caught.value)
    assert "reason role" in message and "meta/llama-3.3-70b" in message and "CONFIDANT_MODEL_REASON" in message


def test_discover_models_lists_only_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    class Catalog:
        """A model catalog that counts how often it is listed."""

        calls = 0

        def list_models(self) -> list[str]:
            """The test catalog."""
            Catalog.calls += 1
            return CATALOG

    assert discover_models(Catalog())["reason"] == "nvidia/Nemotron-3-Ultra-550b-a55b"
    monkeypatch.setenv("CONFIDANT_MODEL_REASON", "a/ultra")
    monkeypatch.setenv("CONFIDANT_MODEL_FAST", "a/nano")
    assert discover_models(Catalog()) == {"reason": "a/ultra", "fast": "a/nano"}
    assert Catalog.calls == 1


def test_cost_estimates_follow_the_model_tier() -> None:
    assert estimate_cost("nvidia/Nemotron-3-Ultra-550b-a55b", 1_000_000, 0) == 1.0
    assert estimate_cost("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", 1_000_000, 1_000_000) == pytest.approx(0.30)
    assert estimate_cost("nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", 0, 1_000_000) == pytest.approx(1.80)
    assert estimate_cost("unknown/model", 1_000_000, 0) == 1.0


def test_a_key_is_required() -> None:
    with pytest.raises(ValueError, match="NEBIUS_API_KEY"):
        TokenFactory("")
