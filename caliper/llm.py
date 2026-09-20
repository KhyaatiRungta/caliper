"""OpenRouter-backed LLM client.

Caliper only needs an LLM for one thing: the ``llm_judge`` grader. Everything
else in the harness is deterministic and runs with no API key. This file is
deliberately a standalone ~200 lines rather than a dependency, because the repo
must clone-and-run.

Key resolution order: ``OPENROUTER_API_KEY`` env var, then a ``.env`` file in
the repo root. The key is never required at import time -- only when a call is
actually attempted.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
FAST_MODEL = "anthropic/claude-3.5-haiku"
JUDGE_MODEL = "anthropic/claude-3.5-sonnet"

MAX_ATTEMPTS = 4

# Fallback per-million-token prices used only when OpenRouter does not report
# usage accounting. Deliberately conservative and clearly marked as estimates.
_FALLBACK_PRICES = {
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3.5-haiku": (0.80, 4.00),
}
_DEFAULT_PRICE = (1.00, 3.00)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader. Does not overwrite existing environment values."""
    path = path or _repo_root() / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str:
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise LLMConfigError(
            "OPENROUTER_API_KEY is not set.\n"
            "Caliper only needs it for the llm_judge grader and for live agent runs.\n"
            "Fix by either:\n"
            "  export OPENROUTER_API_KEY=sk-or-...\n"
            "  cp .env.example .env   # then fill in the key\n"
            "Or re-run with --offline to use the recorded fixtures instead."
        )
    return key


class LLMConfigError(RuntimeError):
    """Raised when a live call is attempted without usable configuration."""


class LLMCallError(RuntimeError):
    """Raised when every retry attempt of a live call failed."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_s=self.latency_s + other.latency_s,
        )


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: dict = field(default_factory=dict)


def _estimate_cost(model: str, pin: int, pout: int) -> float:
    price_in, price_out = _FALLBACK_PRICES.get(model, _DEFAULT_PRICE)
    return (pin * price_in + pout * price_out) / 1_000_000.0


def _normalise_tool_calls(message: Any) -> list[dict]:
    calls = getattr(message, "tool_calls", None) or []
    out: list[dict] = []
    for c in calls:
        fn = getattr(c, "function", None)
        if fn is None:
            continue
        try:
            args = json.loads(fn.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {"_raw": fn.arguments}
        out.append({"id": getattr(c, "id", ""), "name": fn.name, "arguments": args})
    return out


class LLMClient:
    """Live OpenRouter client. Retries 429 and 5xx with jittered backoff."""

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        self.model = model or os.environ.get("CALIPER_JUDGE_MODEL") or JUDGE_MODEL
        self.temperature = temperature
        self.total_usage = Usage()
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - env dependent
                raise LLMConfigError(
                    "The 'openai' package is required for live LLM calls.\n"
                    "Install it with: pip install -r requirements.txt"
                ) from exc
            self._client = OpenAI(base_url=BASE_URL, api_key=get_api_key())
        return self._client

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> LLMResponse:
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "extra_headers": {
                "HTTP-Referer": "https://github.com/caliper-eval/caliper",
                "X-Title": "Caliper",
            },
            "extra_body": {"usage": {"include": True}},
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        started = time.time()
        last_exc: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - SDK raises many shapes
                last_exc = exc
                if not _is_retryable(exc) or attempt == MAX_ATTEMPTS:
                    raise LLMCallError(
                        f"OpenRouter call failed after {attempt} attempt(s): {exc}"
                    ) from exc
                delay = min(2.0 ** (attempt - 1), 8.0) * (0.5 + random.random())
                print(
                    f"caliper: llm retry {attempt}/{MAX_ATTEMPTS - 1} "
                    f"in {delay:.1f}s ({type(exc).__name__})",
                    file=sys.stderr,
                )
                time.sleep(delay)
        else:  # pragma: no cover - loop always breaks or raises
            raise LLMCallError(str(last_exc))

        latency = time.time() - started
        raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        u = raw.get("usage") or {}
        pin = int(u.get("prompt_tokens") or 0)
        pout = int(u.get("completion_tokens") or 0)
        cost = u.get("cost")
        cost_usd = float(cost) if cost is not None else _estimate_cost(self.model, pin, pout)

        msg = resp.choices[0].message
        usage = Usage(pin, pout, cost_usd, latency)
        self.total_usage = self.total_usage + usage
        return LLMResponse(
            text=(msg.content or ""),
            tool_calls=_normalise_tool_calls(msg),
            usage=usage,
            model=raw.get("model", self.model),
            raw=raw,
        )


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    name = type(exc).__name__
    return name in {"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"}


class OfflineLLMClient:
    """Replays canned responses from a fixtures directory. Same interface.

    Lookup is by a stable key derived from the last user message. A missing
    fixture is not an error: the client returns a neutral, clearly-labelled
    response so that an offline run completes rather than half-failing. Tests
    that care assert on the fixture path directly.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        fixtures_dir: Path | str | None = None,
    ):
        self.model = model or "offline/fixture"
        self.temperature = temperature
        self.total_usage = Usage()
        self.fixtures_dir = Path(fixtures_dir or Path(__file__).parent / "fixtures" / "offline")
        self._cache: dict[str, dict] | None = None

    def _load(self) -> dict[str, dict]:
        if self._cache is None:
            self._cache = {}
            if self.fixtures_dir.exists():
                for p in sorted(self.fixtures_dir.glob("*.json")):
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        self._cache.update(data)
        return self._cache

    @staticmethod
    def key_for(messages: list[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return str(m.get("content", ""))[:200].strip()
        return ""

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> LLMResponse:
        key = self.key_for(messages)
        fixture = self._load().get(key)
        if fixture is None:
            fixture = {
                "text": json.dumps(
                    {"score": 0.0, "reasoning": "offline: no fixture recorded for this prompt"}
                ),
                "tool_calls": [],
            }
        usage = Usage(
            prompt_tokens=int(fixture.get("prompt_tokens", 0)),
            completion_tokens=int(fixture.get("completion_tokens", 0)),
            cost_usd=float(fixture.get("cost_usd", 0.0)),
            latency_s=float(fixture.get("latency_s", 0.0)),
        )
        self.total_usage = self.total_usage + usage
        return LLMResponse(
            text=fixture.get("text", ""),
            tool_calls=list(fixture.get("tool_calls", [])),
            usage=usage,
            model=self.model,
            raw={"offline": True, "key": key},
        )
