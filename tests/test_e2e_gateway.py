import argparse
import pytest
from shamu.types import ChatMessage, Role, ToolDefinition, ToolCall, ChatChunk
from shamu.protocols import ModelProvider, ProviderUnavailableError
from shamu.gateway.router import ModelGateway, RoutingPolicy
from shamu.cli import async_main


class MockEndToEndProvider(ModelProvider):
    def __init__(self, name: str = "mock_provider"):
        self.name = name
        self.history_received = []

    async def stream_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
        self.history_received = list(messages)
        last_msg = messages[-1].content
        if "weather" in last_msg.lower():
            # Return a tool call delta
            yield ChatChunk(
                tool_call_delta=ToolCall(
                    id="call_weather_1",
                    name="get_weather",
                    arguments='{"city": "San Francisco"}',
                )
            )
            yield ChatChunk(finish_reason="tool_calls")
        else:
            yield ChatChunk(text_delta=f"Echo from {self.name}: {last_msg}")
            yield ChatChunk(finish_reason="stop")

    async def complete_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
        raise NotImplementedError()


@pytest.mark.asyncio
async def test_e2e_multi_turn_conversation():
    provider = MockEndToEndProvider()
    gateway = ModelGateway(ollama_provider=provider, policy=RoutingPolicy.LOCAL_ONLY)

    # Turn 1
    t1_chunks = []
    async for c in gateway.stream_chat([ChatMessage(role=Role.USER, content="Hello SHAMU")]):
        t1_chunks.append(c)

    text_t1 = "".join(c.text_delta for c in t1_chunks)
    assert "Echo from mock_provider: Hello SHAMU" in text_t1

    # Turn 2 with history
    history = [
        ChatMessage(role=Role.USER, content="Hello SHAMU"),
        ChatMessage(role=Role.ASSISTANT, content=text_t1),
        ChatMessage(role=Role.USER, content="How are you?"),
    ]
    t2_chunks = []
    async for c in gateway.stream_chat(history):
        t2_chunks.append(c)

    text_t2 = "".join(c.text_delta for c in t2_chunks)
    assert "Echo from mock_provider: How are you?" in text_t2
    assert len(provider.history_received) == 3


@pytest.mark.asyncio
async def test_e2e_tool_call_stream():
    provider = MockEndToEndProvider()
    gateway = ModelGateway(ollama_provider=provider, policy=RoutingPolicy.LOCAL_ONLY)

    chunks = []
    async for c in gateway.stream_chat([ChatMessage(role=Role.USER, content="Check weather")]):
        chunks.append(c)

    tool_chunks = [c for c in chunks if c.is_tool_call]
    assert len(tool_chunks) == 1
    tc = tool_chunks[0].tool_call_delta
    assert tc.name == "get_weather"
    assert "San Francisco" in tc.arguments


@pytest.mark.asyncio
async def test_cli_single_prompt_mode(capsys, monkeypatch):
    class MockCLIProvider(ModelProvider):
        async def stream_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
            yield ChatChunk(text_delta="CLI Response Output")
            yield ChatChunk(finish_reason="stop")

        async def complete_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
            pass

    monkeypatch.setattr(
        "shamu.cli.OllamaProvider",
        lambda *args, **kwargs: MockCLIProvider()
    )

    args = argparse.Namespace(
        model="mock",
        fallback=None,
        offline=True,
        cloud=False,
        prompt="Test prompt",
    )
    await async_main(args)

