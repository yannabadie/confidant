"""JSON over HTTPS for the cloud clients: Bearer auth, timeouts, one retry, readable errors."""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_WAIT_S = 10.0


# Failures that happen before any byte of the request leaves: nothing was sent, so no receipt is due.
NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.UnsupportedProtocol, httpx.ProxyError)


class ServiceError(RuntimeError):
    """An outbound call failed. The message is safe to show: it never contains the API key.

    ``unanswered`` holds one reason ("HTTP 503", "timeout"...) per request that left and got no usable
    answer, so each can have its receipt."""

    def __init__(self, message: str, *, unanswered: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.unanswered = tuple(unanswered)


def request_json(
    http: httpx.Client,
    method: str,
    url: str,
    *,
    service: str,
    api_key: str,
    key_env: str,
    timeout: float,
    retry_delay: float,
    payload: Mapping[str, Any] | None = None,
    unanswered: list[str] | None = None,
) -> tuple[Any, int]:
    """Send one JSON request and return ``(decoded body, latency in ms)``.

    Retries once on HTTP 429/5xx (honouring ``Retry-After`` up to 10 s) or when
    the connection fails; a timeout fails at once. Raises ServiceError with a
    message naming ``service`` and, for auth failures, the ``key_env`` variable.
    ``unanswered`` (when given) receives one reason per request that left and got
    no usable answer, also on success after a retry; the ServiceError carries them too.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    failed = unanswered if unanswered is not None else []
    started = time.perf_counter()
    problem = ""
    for attempt in (1, 2):
        try:
            response = http.request(method, url, json=payload, headers=headers, timeout=timeout)
        except NOT_SENT as exc:
            if isinstance(exc, httpx.TimeoutException):
                raise ServiceError(f"{service} did not answer within {timeout:g} s.", unanswered=failed) from None
            problem, wait = f"{service} could not be reached ({type(exc).__name__}).", retry_delay
        except httpx.TimeoutException:
            failed.append("timeout")
            raise ServiceError(f"{service} did not answer within {timeout:g} s.", unanswered=failed) from None
        except httpx.RequestError as exc:  # sent, then the connection broke or the reply could not be read
            failed.append("connection lost")
            problem, wait = f"{service} could not be reached ({type(exc).__name__}).", retry_delay
        else:
            if response.status_code < 400:
                try:
                    return _decode(response, service), round((time.perf_counter() - started) * 1000)
                except ServiceError as exc:
                    failed.append("unreadable response")
                    raise ServiceError(str(exc), unanswered=failed) from None
            failed.append(f"HTTP {response.status_code}")
            problem = _describe(response, service, key_env, api_key)
            if response.status_code not in RETRY_STATUSES:
                raise ServiceError(problem, unanswered=failed)
            wait = _retry_wait(response, retry_delay)
        if attempt == 1:
            time.sleep(wait)
    raise ServiceError(f"{problem} Gave up after one retry.", unanswered=failed)


def _decode(response: httpx.Response, service: str) -> Any:
    """The JSON body of a successful response."""
    try:
        return response.json()
    except ValueError:
        raise ServiceError(f"{service} sent a response that is not JSON.") from None


def _retry_wait(response: httpx.Response, default: float) -> float:
    """Seconds to wait before retrying: the server's Retry-After if given (capped), else ``default``."""
    try:
        return min(max(float(response.headers.get("Retry-After", "")), default), MAX_RETRY_WAIT_S)
    except ValueError:
        return default


def _describe(response: httpx.Response, service: str, key_env: str, api_key: str) -> str:
    """A short, key-free explanation of an HTTP error."""
    status = response.status_code
    if status in (401, 403):
        return f"{service} rejected the API key (HTTP {status}); check {key_env}."
    if status == 404:
        return f"{service} returned HTTP 404 for {response.request.url.path}; check the model id and base URL."
    if status == 429:
        return f"{service} is rate-limiting requests (HTTP 429)."
    if status >= 500:
        return f"{service} had a server error (HTTP {status})."
    try:
        body = response.json()
        error = body.get("error", body) if isinstance(body, dict) else body
        detail = str(error.get("message") or error.get("detail") or error) if isinstance(error, dict) else str(error)
    except ValueError:
        detail = response.text
    detail = " ".join(detail.split())[:200]
    if api_key:
        detail = detail.replace(api_key, "***")
    return f"{service} refused the request (HTTP {status}): {detail}"
