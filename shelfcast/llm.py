"""One small client for any OpenAI-compatible chat endpoint, plus a fake for tests.

It only needs the standard library, so the app's light requirements stay light. It works
with OpenAI, Groq, Together, OpenRouter, a local Ollama server (``http://localhost:11434/v1``)
and any other service that speaks the ``/chat/completions`` API with tool calling.

Settings come from the environment (or ``.env``):
    SHELFCAST_LLM_API_KEY   the key (OPENAI_API_KEY is also accepted)
    SHELFCAST_LLM_MODEL     the model name, e.g. gpt-4o-mini or llama-3.1-8b-instant
    SHELFCAST_LLM_BASE_URL  the API root, default https://api.openai.com/v1
    SHELFCAST_LLM_MIN_INTERVAL  optional pause in seconds between calls, for free tiers with low rate limits
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import load_dotenv

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    seconds: float = 0.0
    raw_message: dict | None = None  # the assistant message as the API returned it, to echo back


class LLMError(RuntimeError):
    """The endpoint could not be reached or answered with an error."""


@dataclass
class LLMSettings:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.2
    timeout: float = 60.0
    reasoning_effort: str | None = None  # e.g. "none" or "low"; only sent when set
    min_interval: float = 0.0  # seconds to leave between calls (free tiers allow only a few per minute)

    @classmethod
    def from_env(cls, model: str | None = None) -> "LLMSettings | None":
        """None when no key is set: the app then uses its built-in summary instead."""
        load_dotenv()
        key = os.environ.get("SHELFCAST_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        return cls(
            api_key=key,
            model=model or os.environ.get("SHELFCAST_LLM_MODEL") or DEFAULT_MODEL,
            base_url=(os.environ.get("SHELFCAST_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            reasoning_effort=os.environ.get("SHELFCAST_LLM_REASONING_EFFORT") or None,
            timeout=float(os.environ.get("SHELFCAST_LLM_TIMEOUT") or 60.0),
            min_interval=float(os.environ.get("SHELFCAST_LLM_MIN_INTERVAL") or 0.0),
        )


class OpenAICompatibleClient:
    """Calls ``POST {base_url}/chat/completions``. One retry on a 5xx; a 429 (rate limit) waits for
    the delay the service asks for and tries again, up to four times in all."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self._last_call = 0.0

    @property
    def model(self) -> str:
        return self.settings.model

    def complete(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1500) -> LLMResponse:
        body: dict = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens,
        }
        if self.settings.reasoning_effort:
            body["reasoning_effort"] = self.settings.reasoning_effort
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.settings.api_key}"},
            method="POST",
        )
        started = time.perf_counter()
        wait = self.settings.min_interval - (time.monotonic() - self._last_call)
        if self._last_call and wait > 0:
            time.sleep(wait)
        for attempt in (1, 2, 3, 4):
            try:
                self._last_call = time.monotonic()
                with urllib.request.urlopen(request, timeout=self.settings.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1200]
                if exc.code == 429 and attempt < 4:
                    time.sleep(_rate_limit_wait(detail, exc.headers.get("Retry-After") if exc.headers else None))
                    continue
                if exc.code >= 500 and attempt == 1:
                    time.sleep(2.0)
                    continue
                raise LLMError(f"{self.settings.model} answered HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                raise LLMError(f"Could not reach {self.settings.base_url}: {exc}") from exc
        return parse_chat_completion(payload, self.settings.model, time.perf_counter() - started)


def _rate_limit_wait(detail: str, retry_after: str | None) -> float:
    """Seconds to wait after a 429: the service's own hint when it gives one, else 20, at most 70."""
    seconds = 20.0
    if retry_after and retry_after.strip().replace(".", "", 1).isdigit():
        seconds = float(retry_after)
    else:
        hint = re.search(r'retry(?:Delay)?[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)\s*s', detail, re.IGNORECASE)
        if hint:
            seconds = float(hint.group(1))
    return min(max(seconds, 2.0) + 1.0, 70.0)


def parse_chat_completion(payload: dict, model: str, seconds: float = 0.0) -> LLMResponse:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected answer from the model: {str(payload)[:200]}") from exc
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(id=str(call.get("id", "")), name=str(fn.get("name", "")), arguments=args))
    if payload["choices"][0].get("finish_reason") == "length" and not calls:
        # Models that "think" first can spend the whole budget before the answer starts, and
        # what is left is a fragment. Better to say so than to hand a fragment to the checks.
        raise LLMError(f"{model} ran out of tokens before finishing its answer (finish_reason=length). "
                       "Set SHELFCAST_LLM_REASONING_EFFORT=low (or none) or choose a model that does not think first.")
    text = message.get("content") or ""
    if isinstance(text, list):  # some servers return content parts
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    return LLMResponse(text=text.strip(), tool_calls=calls, model=payload.get("model") or model, seconds=seconds, raw_message=message)


class FakeClient:
    """A scripted model for tests and offline demos.

    ``script`` is a list of responses, used in order. Each item is either a string (the
    model's final text) or a list of ``(tool_name, arguments)`` pairs (one round of tool calls).
    A callable instead of a list is called with the messages and must return such an item,
    which lets a test answer from the tool results it sees.
    """

    def __init__(self, script, model: str = "fake-model"):
        self.script = script
        self.model = model
        self.calls: list[dict] = []
        self._step = 0

    def complete(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1500) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if callable(self.script):
            item = self.script(messages)
        else:
            if self._step >= len(self.script):
                raise LLMError("FakeClient ran out of scripted answers")
            item = self.script[self._step]
            self._step += 1
        if isinstance(item, str):
            return LLMResponse(text=item, model=self.model, raw_message={"role": "assistant", "content": item})
        calls = [ToolCall(id=f"call_{i}", name=name, arguments=args) for i, (name, args) in enumerate(item)]
        raw = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in calls
            ],
        }
        return LLMResponse(text="", tool_calls=calls, model=self.model, raw_message=raw)


def client_from_env(model: str | None = None) -> OpenAICompatibleClient | None:
    settings = LLMSettings.from_env(model)
    return OpenAICompatibleClient(settings) if settings else None
