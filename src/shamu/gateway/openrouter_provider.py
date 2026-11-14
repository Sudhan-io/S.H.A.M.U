"""Cloud OpenRouter Model Provider with automatic streaming failover."""

import json
from typing import AsyncIterator, Dict, List, Optional
import httpx
from shamu.types import ChatMessage, ToolDefinition, ChatChunk, ChatResponse, ToolCall
from shamu.config import get_api_key
from shamu.protocols import (
    ModelProvider,
    ProviderUnavailableError,
    ProviderAuthError,
    ShamuError,
)


class OpenRouterProvider(ModelProvider):
    """Integrates with OpenRouter API for cloud model fallback and heavy tasks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-flash",
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key or get_api_key("openrouter")
        self.model = model
        self.timeout = timeout
        self.base_url = "https://openrouter.ai/api/v1"
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or getattr(self._client, "is_closed", False):
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        """Closes the persistent HTTP client connection pool."""
        if self._client is not None:
            if hasattr(self._client, "aclose"):
                await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    def _get_headers(self) -> dict:
        if not self.api_key:
            raise ProviderAuthError(
                "OpenRouter API key not configured. Set via 'set_api_key(\"openrouter\", ...)' or keyring.",
                provider="openrouter",
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/shamu-assistant/shamu",
            "X-Title": "SHAMU AI Desktop Assistant",
            "Content-Type": "application/json",
        }

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatChunk]:
        """Streams response tokens and tool call deltas from OpenRouter."""
        headers = self._get_headers()
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [t.to_openai_dict() for t in tools]

        client = self._get_client()
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code == 401:
                        raise ProviderAuthError("OpenRouter authentication failed (HTTP 401). Invalid API key.", provider="openrouter")
                    if response.status_code == 429:
                        raise ProviderUnavailableError("OpenRouter rate limit reached (HTTP 429).", provider="openrouter")
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise ShamuError(f"OpenRouter returned HTTP {response.status_code}: {error_text.decode('utf-8', errors='replace')}")

                    tool_call_builders: Dict[int, Dict[str, str]] = {}

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            yield ChatChunk(finish_reason="stop")
                            break

                        try:
                            chunk_json = json.loads(data_str)
                            if "error" in chunk_json:
                                err = chunk_json["error"]
                                is_429 = False
                                if isinstance(err, dict):
                                    code = err.get("code") or err.get("status")
                                    if code == 429 or str(code) == "429":
                                        is_429 = True
                                    elif "429" in str(err.get("message", "")) or "rate limit" in str(err.get("message", "")).lower():
                                        is_429 = True
                                elif isinstance(err, str):
                                    if "429" in err or "rate limit" in err.lower():
                                        is_429 = True
                                elif err == 429:
                                    is_429 = True

                                if is_429:
                                    raise ProviderUnavailableError(f"OpenRouter rate limit reached mid-stream (HTTP 429): {err}", provider="openrouter")
                                raise ShamuError(f"OpenRouter mid-stream error: {err}")

                            choices = chunk_json.get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0]
                            delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                            finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None

                            text_delta = delta.get("content") or ""
                            raw_tcs = delta.get("tool_calls") or []

                            if raw_tcs:
                                for raw_tc in raw_tcs:
                                    idx = raw_tc.get("index", 0)
                                    if idx not in tool_call_builders:
                                        tool_call_builders[idx] = {
                                            "id": raw_tc.get("id") or "",
                                            "name": "",
                                            "arguments": "",
                                        }
                                    else:
                                        if raw_tc.get("id"):
                                            tool_call_builders[idx]["id"] = raw_tc["id"]

                                    fn = raw_tc.get("function") or {}
                                    if fn.get("name"):
                                        tool_call_builders[idx]["name"] = fn["name"]
                                    if fn.get("arguments"):
                                        tool_call_builders[idx]["arguments"] += fn["arguments"]

                                    tc_delta = ToolCall(
                                        id=tool_call_builders[idx]["id"],
                                        name=tool_call_builders[idx]["name"],
                                        arguments=tool_call_builders[idx]["arguments"],
                                    )
                                    yield ChatChunk(
                                        text_delta=text_delta,
                                        tool_call_delta=tc_delta,
                                        finish_reason=finish_reason,
                                    )
                                    text_delta = ""
                            else:
                                yield ChatChunk(
                                    text_delta=text_delta,
                                    tool_call_delta=None,
                                    finish_reason=finish_reason,
                                )
                        except json.JSONDecodeError:
                            continue

        except (httpx.ConnectError, httpx.TransportError, httpx.ProtocolError, httpx.HTTPError) as e:
            if isinstance(e, httpx.ConnectError):
                raise ProviderUnavailableError("OpenRouter service unreachable (Network disconnected).", provider="openrouter") from e
            if isinstance(e, httpx.TimeoutException):
                raise ProviderUnavailableError(f"OpenRouter request timed out after {self.timeout}s.", provider="openrouter") from e
            raise ProviderUnavailableError(f"OpenRouter streaming HTTP error: {e}", provider="openrouter") from e

    async def complete_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Executes non-streaming completion via OpenRouter."""
        headers = self._get_headers()
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [t.to_openai_dict() for t in tools]

        client = self._get_client()
        try:
            res = await client.post(url, json=payload, headers=headers)
            if res.status_code == 401:
                raise ProviderAuthError("OpenRouter authentication failed (HTTP 401).", provider="openrouter")
            if res.status_code == 429:
                raise ProviderUnavailableError("OpenRouter rate limit reached (HTTP 429).", provider="openrouter")
            if res.status_code != 200:
                raise ShamuError(f"OpenRouter error {res.status_code}: {res.text}")

            data = res.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content")
            finish_reason = choice.get("finish_reason")

            tool_calls = None
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", "{}"),
                    )
                    for tc in msg["tool_calls"]
                ]

            return ChatResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=data.get("usage"),
            )
        except httpx.ConnectError as e:
            raise ProviderUnavailableError("OpenRouter service unreachable.", provider="openrouter") from e
        except httpx.TimeoutException as e:
            raise ProviderUnavailableError(f"OpenRouter timed out after {self.timeout}s.", provider="openrouter") from e

