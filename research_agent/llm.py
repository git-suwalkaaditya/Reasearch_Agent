"""LLM client: plain completions, validated structured outputs, and native tool calling.

Works with any OpenAI-compatible Chat Completions endpoint (OpenAI, Groq, Gemini's
OpenAI-compatible endpoint, Ollama, OpenRouter, vLLM, ...), using only ``requests``.

Key ideas you can learn from this file
--------------------------------------
* Structured outputs: we show the model a JSON schema, ask for JSON only, then
  validate it with pydantic. If validation fails we send the error back and ask the
  model to repair its own output ("self-repair loop").
* Tool calling: we pass OpenAI-format tool definitions; the model replies with
  ``tool_calls`` that our agent executes (see ``nodes.search_node``).
* Failure recovery: HTTP retries with exponential backoff, and graceful
  degradation when an endpoint does not support JSON mode or tools.
"""
from __future__ import annotations

import json
from typing import Any, Optional, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from .config import Settings
from .utils import extract_json, logger, retry

M = TypeVar("M", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class ToolCallingUnsupported(LLMError):
    pass


class BaseLLM:
    """Interface shared by the real client and the offline mock."""

    name = "base"

    def __init__(self) -> None:
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "repairs": 0}

    # -- to implement -------------------------------------------------------
    def complete(self, messages: list[dict], task: str = "", context: Optional[dict] = None, **kw) -> str:
        raise NotImplementedError

    def structured(
        self, messages: list[dict], schema: Type[M], task: str = "", context: Optional[dict] = None, max_repairs: int = 2
    ) -> M:
        raise NotImplementedError

    def chat_with_tools(
        self, messages: list[dict], tools: list[dict], task: str = "", context: Optional[dict] = None
    ) -> dict:
        """Return {"content": str|None, "tool_calls": [{"id", "name", "arguments": dict}]}."""
        raise NotImplementedError

    @property
    def supports_tools(self) -> bool:
        return True


def schema_instructions(schema: Type[BaseModel]) -> str:
    js = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    return (
        "Respond with a single JSON object that conforms to this JSON Schema. "
        "Output JSON only - no markdown fences, no commentary.\n"
        f"JSON Schema: {js}"
    )


class OpenAICompatLLM(BaseLLM):
    name = "openai-compatible"

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.s = settings
        self.url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        self._json_mode = settings.use_json_mode
        self._tools_ok = settings.use_tool_calling

    @property
    def supports_tools(self) -> bool:
        return self._tools_ok

    # -- transport ----------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.s.llm_api_key:
            headers["Authorization"] = f"Bearer {self.s.llm_api_key}"

        def call() -> dict:
            resp = requests.post(self.url, headers=headers, json=payload, timeout=self.s.llm_timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")  # retryable
            if resp.status_code >= 400:
                # Non-retryable (bad request / auth). Surface the body for diagnosis.
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            return resp.json()

        data = retry(call, attempts=4, base_delay=2.0, exceptions=(LLMError, requests.RequestException), what="LLM call")
        self.usage["calls"] += 1
        usage = data.get("usage") or {}
        self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
        self.usage["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        return data

    def _payload(self, messages: list[dict], **extra) -> dict:
        payload = {"model": self.s.llm_model, "messages": messages, "temperature": self.s.llm_temperature}
        payload.update(extra)
        return payload

    # -- API ----------------------------------------------------------------
    def complete(self, messages: list[dict], task: str = "", context: Optional[dict] = None, **kw) -> str:
        data = self._post(self._payload(messages, **kw))
        return (data["choices"][0]["message"].get("content") or "").strip()

    def structured(
        self, messages: list[dict], schema: Type[M], task: str = "", context: Optional[dict] = None, max_repairs: int = 2
    ) -> M:
        msgs = list(messages)
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + schema_instructions(schema)}
        last_error = ""
        for attempt in range(max_repairs + 1):
            extra = {"response_format": {"type": "json_object"}} if self._json_mode else {}
            try:
                raw = self.complete(msgs, **extra)
            except ValueError as exc:
                err = str(exc).lower()
                if self._json_mode and ("response_format" in err or "json_object" in err or "json mode" in err):
                    logger.warning("Endpoint rejected JSON mode; falling back to prompt-only JSON.")
                    self._json_mode = False
                    raw = self.complete(msgs)
                else:
                    raise
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:1500]
                self.usage["repairs"] += 1
                logger.warning("[%s] structured output invalid (attempt %d): %s", task, attempt + 1, last_error[:200])
                msgs = msgs + [
                    {"role": "assistant", "content": raw[:6000]},
                    {
                        "role": "user",
                        "content": "Your previous reply was not valid for the required schema.\n"
                        f"Error: {last_error}\nReturn the corrected JSON object only.",
                    },
                ]
        raise LLMError(f"[{task}] could not obtain valid structured output: {last_error}")

    def chat_with_tools(
        self, messages: list[dict], tools: list[dict], task: str = "", context: Optional[dict] = None
    ) -> dict:
        if not self._tools_ok:
            raise ToolCallingUnsupported("tool calling disabled")
        try:
            data = self._post(self._payload(messages, tools=tools, tool_choice="auto"))
        except ValueError as exc:
            if "tool" in str(exc).lower():
                self._tools_ok = False
                raise ToolCallingUnsupported(str(exc)) from exc
            raise
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args})
        return {"content": msg.get("content"), "tool_calls": calls, "raw_message": msg}


def make_llm(settings: Settings) -> BaseLLM:
    if settings.offline:
        from .mock import MockLLM

        return MockLLM()
    problems = settings.validate()
    if problems:
        raise LLMError("Configuration problem: " + " ".join(problems))
    return OpenAICompatLLM(settings)
