"""Nebius Token Factory client (OpenAI-compatible) and Nemotron model selection.

``TokenFactory.chat`` returns only the final answer: Nemotron reasoning traces,
whether in ``reasoning_content`` or inline ``<think>`` blocks, never reach the
user, while their tokens still count in the reported usage. Receipts for these
calls are written by the assistant, which knows which memories each one carried.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .config import env
from .net import ServiceError, request_json

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
SERVICE = "Nebius Token Factory"
KEY_ENV = "NEBIUS_API_KEY"

# Model choice per role: the first preference with a usable match wins (matched
# case-insensitively on the part of the id after the last "/").
REASON_PREFERENCE = ("nemotron-3-ultra", "nemotron-3-super", "llama-3_1-nemotron-ultra")
FAST_PREFERENCE = ("nemotron-3-nano-30b", "nemotron-3.5-lightning", "nemotron-3-super")
SKIPPED_MARKERS = ("omni", "vl", "kvoff")
SERVERLESS_SUFFIX = "-serverless"
ROLE_ENV = {"reason": "CONFIDANT_MODEL_REASON", "fast": "CONFIDANT_MODEL_FAST"}

# Approximate Token Factory list prices in USD per million input / output tokens
# (October 2026), matched on the model name. Receipts show them as estimates only.
PRICES_PER_MILLION = (
    ("llama-3_1-nemotron-ultra", 0.60, 1.80),
    ("ultra", 1.00, 3.00),
    ("super", 0.30, 0.90),
    ("lightning", 0.06, 0.24),
    ("nano", 0.06, 0.24),
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class ModelSelectionError(ValueError):
    """No model in the catalog fits a role; the message lists what is available."""


class ModelCatalog(Protocol):
    """Anything that can list model ids (TokenFactory, or a test double)."""

    def list_models(self) -> list[str]:
        """Model ids available to the caller."""
        ...


@dataclass(frozen=True)
class ChatResult:
    """One chat completion: final answer text (no reasoning trace) and usage."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model: str
    finish_reason: str = ""


def strip_reasoning(text: str) -> str:
    """The final answer without its reasoning trace.

    Removes ``<think>…</think>`` blocks; when a closing ``</think>`` is left over
    (an unterminated or untagged leading trace), keeps only what follows the last
    one; a trace that never closes leaves no answer at all.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    lowered = cleaned.lower()
    if "</think>" in lowered:
        cleaned = cleaned[lowered.rindex("</think>") + len("</think>"):]
    elif lowered.lstrip().startswith("<think>"):
        return ""
    return cleaned.strip()


def _name(model_id: str) -> str:
    """Lower-cased model name: the part of the id after the last "/"."""
    return model_id.rsplit("/", 1)[-1].lower()


def _first_match(ids: Sequence[str], preferences: Sequence[str]) -> str | None:
    """The best id for the first preference that matches any of ``ids``.

    The single tie-break: when both "X" and "X-serverless" match, "X" wins;
    otherwise the first match in sorted order.
    """
    for keyword in preferences:
        matches = [model_id for model_id in ids if keyword in _name(model_id)]
        if matches:
            names = {_name(model_id) for model_id in matches}
            preferred = [
                model_id for model_id in matches
                if not (
                    _name(model_id).endswith(SERVERLESS_SUFFIX)
                    and _name(model_id)[: -len(SERVERLESS_SUFFIX)] in names
                )
            ]
            return preferred[0]
    return None


def pick_models(ids: Iterable[str]) -> dict[str, str]:
    """Choose ``{"reason": id, "fast": id}`` from a model catalog.

    reason: first available of REASON_PREFERENCE; fast: first available of
    FAST_PREFERENCE. Ids containing "omni", "vl" or "kvoff" are skipped. The
    environment variables CONFIDANT_MODEL_REASON and CONFIDANT_MODEL_FAST win
    over the catalog. Raises ModelSelectionError listing the available ids when
    a role has no match.
    """
    catalog = sorted(set(ids))
    usable = [model_id for model_id in catalog if not any(m in _name(model_id) for m in SKIPPED_MARKERS)]
    chosen: dict[str, str] = {}
    for role, preferences in (("reason", REASON_PREFERENCE), ("fast", FAST_PREFERENCE)):
        model_id = env(ROLE_ENV[role]) or _first_match(usable, preferences)
        if model_id is None:
            raise ModelSelectionError(
                f"No Nemotron model fits the {role} role (wanted one of: {', '.join(preferences)}). "
                f"Available: {', '.join(catalog) or 'none'}. Set {ROLE_ENV[role]} to choose one."
            )
        chosen[role] = model_id
    return chosen


def discover_models(catalog: ModelCatalog) -> dict[str, str]:
    """:func:`pick_models` over ``catalog``, without listing when both overrides are set."""
    if all(env(name) for name in ROLE_ENV.values()):
        return pick_models([])
    return pick_models(catalog.list_models())


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost of a call from PRICES_PER_MILLION (unknown models: the dearest tier)."""
    name = _name(model)
    price_in, price_out = next(
        ((cost_in, cost_out) for fragment, cost_in, cost_out in PRICES_PER_MILLION if fragment in name),
        (1.00, 3.00),
    )
    return round((prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000, 6)


class TokenFactory:
    """Client for Nebius Token Factory's OpenAI-compatible API.

    ``http`` lets callers inject an ``httpx.Client`` (tests use ``httpx.MockTransport``).
    The key only ever goes into the Authorization header.
    """

    pick_models = staticmethod(pick_models)

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        http: httpx.Client | None = None,
        *,
        timeout: float = 60.0,
        retry_delay: float = 1.0,
    ) -> None:
        if not api_key:
            raise ValueError(f"TokenFactory needs an API key ({KEY_ENV}).")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_delay = retry_delay
        self._http = http if http is not None else httpx.Client()
        self._owns_http = http is None

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        purpose: str = "chat",
    ) -> ChatResult:
        """Run one chat completion (POST /chat/completions) and return the final answer.

        ``purpose`` labels error messages. Raises ServiceError on failure.
        """
        payload = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data, latency_ms = self._request("POST", "/chat/completions", purpose, payload)
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ServiceError(f"{SERVICE} ({purpose}) sent an unexpected response shape.") from None
        usage = data.get("usage") or {}
        return ChatResult(
            text=strip_reasoning(str(content)),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
            model=str(data.get("model") or model),
            finish_reason=str(choice.get("finish_reason") or ""),
        )

    def list_models(self) -> list[str]:
        """Model ids visible to this key (GET /models), sorted."""
        data, _ = self._request("GET", "/models", "model list", None)
        try:
            return sorted({str(item["id"]) for item in data["data"]})
        except (KeyError, TypeError):
            raise ServiceError(f"{SERVICE} (model list) sent an unexpected response shape.") from None

    def close(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._owns_http:
            self._http.close()

    def _request(
        self, method: str, path: str, purpose: str, payload: Mapping[str, object] | None
    ) -> tuple[Any, int]:
        """Send one request to the API with this client's settings."""
        return request_json(
            self._http, method, self.base_url + path, service=f"{SERVICE} ({purpose})",
            api_key=self._api_key, key_env=KEY_ENV, timeout=self.timeout,
            retry_delay=self.retry_delay, payload=payload,
        )
