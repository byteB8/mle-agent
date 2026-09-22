"""Minimal client for an OpenAI-compatible /v1/chat/completions endpoint (e.g. vLLM).

Stdlib only: the point is to own the request/response handling, retries and token
accounting rather than hide them behind an SDK.
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    def __init__(self, msg: str, retryable: bool = False):
        super().__init__(msg)
        self.retryable = retryable


@dataclass
class Completion:
    message: dict            # assistant message, normalised to {role, content, tool_calls?}
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency: float


def _normalise(msg: dict) -> dict:
    """Keep only the fields we send back to the server; drop reasoning/extra keys."""
    out = {"role": "assistant", "content": msg.get("content") or ""}
    calls = []
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args or {})
        calls.append({"id": c.get("id") or f"call_{len(calls)}", "type": "function",
                      "function": {"name": fn.get("name", ""), "arguments": args}})
    if calls:
        out["tool_calls"] = calls
    return out


class LLM:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 timeout: float = 900.0, max_retries: int = 5):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.7, top_p: float = 0.8, max_tokens: int = 4096,
             seed: int | None = None) -> Completion:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "top_p": top_p, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if seed is not None:
            body["seed"] = seed

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                data = self._post("/chat/completions", body)
            except LLMError as e:
                if not e.retryable or attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            try:
                choice = data["choices"][0]
            except (KeyError, IndexError) as e:
                raise LLMError(f"malformed response: {str(data)[:500]}") from e
            usage = data.get("usage") or {}
            return Completion(message=_normalise(choice.get("message") or {}),
                              finish_reason=choice.get("finish_reason") or "",
                              prompt_tokens=int(usage.get("prompt_tokens") or 0),
                              completion_tokens=int(usage.get("completion_tokens") or 0),
                              latency=time.monotonic() - t0)
        raise AssertionError("unreachable")

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:1000]
            raise LLMError(f"HTTP {e.code}: {detail}", retryable=e.code in RETRYABLE_HTTP) from e
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            raise LLMError(f"connection error: {e}", retryable=True) from e
