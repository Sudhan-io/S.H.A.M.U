import pytest
from typing import AsyncIterator, List, Optional
from shamu.types import ChatMessage, Role, ChatChunk, ChatResponse, ToolDefinition
from shamu.protocols import ModelProvider, ProviderUnavailableError
from shamu.gateway.router import ModelGateway, RoutingPolicy


class MockProvider(ModelProvider):
    def __init__(self, response_text: str = "ok", should_fail: bool = False):
        self.response_text = response_text
        self.should_fail = should_fail
        self.call_count = 0

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatChunk]:
        self.call_count += 1
        if self.should_fail:
            raise ProviderUnavailableError("Provider offline", provider="mock")
        yield ChatChunk(text_delta=self.response_text, finish_reason="stop")

    async def complete_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        self.call_count += 1
        if self.should_fail:
            raise ProviderUnavailableError("Provider offline", provider="mock")
        return ChatResponse(content=self.response_text, finish_reason="stop")


@pytest.mark.asyncio
async def test_router_local_only():
    ollama = MockProvider(response_text="local reply")
    openrouter = MockProvider(response_text="cloud reply")
    router = ModelGateway(
        ollama_provider=ollama,
        openrouter_provider=openrouter,
        policy=RoutingPolicy.LOCAL_ONLY,
    )

    chunks = []
    async for c in router.stream_chat([ChatMessage(role=Role.USER, content="hi")]):
        chunks.append(c)

    assert "local reply" in [c.text_delta for c in chunks]
    assert ollama.call_count == 1
    assert openrouter.call_count == 0


@pytest.mark.asyncio
async def test_router_auto_fallback_success():
    ollama = MockProvider(response_text="", should_fail=True)
    openrouter = MockProvider(response_text="cloud fallback")
    fallback_events = []

    def handle_fallback(msg, exc):
        fallback_events.append((msg, exc))

    router = ModelGateway(
        ollama_provider=ollama,
        openrouter_provider=openrouter,
        policy=RoutingPolicy.AUTO_FALLBACK,
        on_fallback=handle_fallback,
    )

    chunks = []
    async for c in router.stream_chat([ChatMessage(role=Role.USER, content="hi")]):
        chunks.append(c)

    text = "".join(c.text_delta for c in chunks)
    assert "cloud fallback" in text
    assert ollama.call_count == 1
    assert openrouter.call_count == 1
    assert len(fallback_events) == 1
    assert "failing over" in fallback_events[0][0].lower()


@pytest.mark.asyncio
async def test_router_mid_stream_failover_signals_reset():
    """ARCH-01: When failover occurs mid-stream, router emits reset_turn=True on fallback chunk."""
    class FailingMidStreamProvider(ModelProvider):
        async def stream_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
            yield ChatChunk(text_delta="Partial text that should be reset")
            raise ProviderUnavailableError("Lost connection mid-turn", provider="ollama")

        async def complete_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
            raise ProviderUnavailableError("Lost connection", provider="ollama")

    ollama = FailingMidStreamProvider()
    openrouter = MockProvider(response_text="Complete answer from cloud")

    router = ModelGateway(
        ollama_provider=ollama,
        openrouter_provider=openrouter,
        policy=RoutingPolicy.AUTO_FALLBACK,
    )

    chunks = []
    async for c in router.stream_chat([ChatMessage(role=Role.USER, content="Hello")]):
        chunks.append(c)

    # First chunk is partial
    assert chunks[0].text_delta == "Partial text that should be reset"
    assert chunks[0].reset_turn is False

    # Second chunk is the failover notification, which MUST signal reset_turn=True
    assert chunks[1].reset_turn is True
    assert "Failing over" in chunks[1].text_delta

    # Third chunk is cloud answer
    assert chunks[2].text_delta == "Complete answer from cloud"


