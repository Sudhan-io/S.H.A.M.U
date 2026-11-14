"""Unified Model Gateway and dynamic failover router."""

from enum import Enum
from typing import AsyncIterator, Callable, List, Optional, Any
from shamu.types import ChatMessage, ToolDefinition, ChatChunk, ChatResponse
from shamu.protocols import (
    ModelProvider,
    ProviderUnavailableError,
    ContextWindowExceededError,
    ShamuError,
)
from shamu.gateway.ollama_provider import OllamaProvider
from shamu.gateway.openrouter_provider import OpenRouterProvider
from shamu.config import get_config


class RoutingPolicy(str, Enum):
    LOCAL_ONLY = "local_only"
    AUTO_FALLBACK = "auto_fallback"
    CLOUD_ONLY = "cloud_only"


class ModelGateway(ModelProvider):
    """Central gateway that orchestrates local inference and cloud fallback."""

    def __init__(
        self,
        ollama_provider: Optional[OllamaProvider] = None,
        openrouter_provider: Optional[OpenRouterProvider] = None,
        policy: RoutingPolicy = RoutingPolicy.AUTO_FALLBACK,
        on_fallback: Optional[Callable[[str, Exception], None]] = None,
        ui_bridge: Optional[Any] = None,
        emit_notice: bool = True,
    ):
        cfg = get_config()
        self.ollama = ollama_provider or OllamaProvider(
            base_url=cfg.ollama_url,
            model=cfg.local_model,
            timeout=float(cfg.timeout_seconds),
            inference_config=cfg.inference,
        )
        self.openrouter = openrouter_provider or OpenRouterProvider(
            model=cfg.fallback_model,
            timeout=float(cfg.timeout_seconds),
        )
        self.policy = policy
        self.on_fallback = on_fallback
        self.ui_bridge = ui_bridge
        self.emit_notice = emit_notice

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatChunk]:
        """Streams chat completions adhering to the configured RoutingPolicy."""
        if self.policy == RoutingPolicy.CLOUD_ONLY:
            async for chunk in self.openrouter.stream_chat(messages, tools, temperature, max_tokens):
                yield chunk
            return

        if self.policy == RoutingPolicy.LOCAL_ONLY:
            async for chunk in self.ollama.stream_chat(messages, tools, temperature, max_tokens):
                yield chunk
            return

        # AUTO_FALLBACK mode: Attempt local first
        yielded_any = False
        try:
            async for chunk in self.ollama.stream_chat(messages, tools, temperature, max_tokens):
                if chunk.text_delta or chunk.tool_call_delta:
                    yielded_any = True
                yield chunk
        except (ProviderUnavailableError, ContextWindowExceededError, ShamuError, Exception) as exc:
            if self.on_fallback:
                self.on_fallback("Local Ollama failed; failing over to OpenRouter.", exc)

            if yielded_any and self.ui_bridge:
                if hasattr(self.ui_bridge, "turn_reset"):
                    self.ui_bridge.turn_reset.emit()
                if hasattr(self.ui_bridge, "clear_partial"):
                    self.ui_bridge.clear_partial.emit()
                if hasattr(self.ui_bridge, "status_updated"):
                    self.ui_bridge.status_updated.emit("Failing over to cloud...")

            notice_text = (
                f"\n[Notice: Local model unavailable ({exc}). Failing over to cloud provider...]\n\n"
                if self.emit_notice
                else ""
            )

            yield ChatChunk(
                reset_turn=yielded_any,
                text_delta=notice_text,
            )

            async for chunk in self.openrouter.stream_chat(messages, tools, temperature, max_tokens):
                yield chunk

    async def complete_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Executes a full non-streaming turn with fallback."""
        if self.policy == RoutingPolicy.CLOUD_ONLY:
            return await self.openrouter.complete_chat(messages, tools, temperature, max_tokens)

        if self.policy == RoutingPolicy.LOCAL_ONLY:
            return await self.ollama.complete_chat(messages, tools, temperature, max_tokens)

        try:
            return await self.ollama.complete_chat(messages, tools, temperature, max_tokens)
        except (ProviderUnavailableError, ContextWindowExceededError, ShamuError, Exception) as exc:
            if self.on_fallback:
                self.on_fallback("Local Ollama failed; failing over to OpenRouter.", exc)
            return await self.openrouter.complete_chat(messages, tools, temperature, max_tokens)

    async def aclose(self) -> None:
        """Closes both primary and fallback provider connection pools."""
        if hasattr(self.ollama, "aclose"):
            await self.ollama.aclose()
        if hasattr(self.openrouter, "aclose"):
            await self.openrouter.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()


