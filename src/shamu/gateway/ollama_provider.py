"""Local Ollama Model Provider with streaming and tool-calling support."""

import json
from typing import AsyncIterator, Dict, List, Optional
import httpx
from shamu.config import InferenceConfig
from shamu.types import ChatMessage, ToolDefinition, ChatChunk, ChatResponse, ToolCall
from shamu.protocols import (
    ModelProvider,
    ProviderUnavailableError,
    ContextWindowExceededError,
    ShamuError,
)


class OllamaProvider(ModelProvider):
    """Integrates with local Ollama instance via OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct-q4_K_M",
        timeout: float = 30.0,
        inference_config: Optional[InferenceConfig] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.inference_config = inference_config or InferenceConfig()
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

    def _build_options(self, temperature: float, include_advanced: bool = True) -> dict:
        opts: dict = {
            "num_ctx": self.inference_config.num_ctx,
            "keep_alive": self.inference_config.keep_alive,
            "temperature": temperature,
        }
        if include_advanced:
            if self.inference_config.flash_attention:
                opts["flash_attention"] = True
            if self.inference_config.cache_type_k and self.inference_config.cache_type_k != "f16":
                opts["cache_type_k"] = self.inference_config.cache_type_k
            if self.inference_config.cache_type_v and self.inference_config.cache_type_v != "f16":
                opts["cache_type_v"] = self.inference_config.cache_type_v
        return opts

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatChunk]:
        """Streams response tokens and tool call deltas from Ollama."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "stream": True,
            "options": self._build_options(temperature),
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [t.to_openai_dict() for t in tools]

        client = self._get_client()
        try:
            async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        decoded_error = error_text.decode("utf-8", errors="replace")
                        if "out of memory" in decoded_error.lower() or "cuda" in decoded_error.lower():
                            raise ContextWindowExceededError(f"Ollama GPU Out of Memory: {decoded_error}")
                        if response.status_code >= 500:
                            raise ProviderUnavailableError(f"Ollama server error (HTTP {response.status_code}): {decoded_error}", provider="ollama")
                        raise ShamuError(f"Ollama returned HTTP {response.status_code}: {decoded_error}")

                    # Buffer for assembling fragmented tool call JSON per call index
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
                                    raise ProviderUnavailableError(f"Ollama rate limit mid-stream: {err}", provider="ollama")
                                raise ShamuError(f"Ollama mid-stream error: {err}")

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
                raise ProviderUnavailableError(
                    f"Ollama server unreachable at {self.base_url}. Ensure 'ollama serve' is active.",
                    provider="ollama",
                ) from e
            if isinstance(e, httpx.TimeoutException):
                raise ProviderUnavailableError(
                    f"Ollama request timed out after {self.timeout}s.",
                    provider="ollama",
                ) from e
            raise ProviderUnavailableError(
                f"Ollama streaming HTTP error: {e}",
                provider="ollama",
            ) from e

    async def complete_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Executes non-streaming completion."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "stream": False,
            "options": self._build_options(temperature),
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [t.to_openai_dict() for t in tools]

        client = self._get_client()
        try:
            res = await client.post(url, json=payload)
            if res.status_code != 200:
                text = res.text
                if "out of memory" in text.lower() or "cuda" in text.lower():
                    raise ContextWindowExceededError(f"Ollama GPU Out of Memory: {text}")
                if res.status_code >= 500:
                    raise ProviderUnavailableError(f"Ollama server error (HTTP {res.status_code}): {text}", provider="ollama")
                raise ShamuError(f"Ollama error {res.status_code}: {text}")

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
            raise ProviderUnavailableError(
                f"Ollama server unreachable at {self.base_url}",
                provider="ollama",
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderUnavailableError(
                f"Ollama request timed out after {self.timeout}s",
                provider="ollama",
            ) from e

