"""Regression tests for CANON-P1-05: Mid-Stream Model Failover & UI Synchronization."""

import pytest
import asyncio
from typing import AsyncIterator, List, Optional
from PySide6.QtWidgets import QApplication

from shamu.types import ChatMessage, Role, ChatChunk, ToolCall, ToolDefinition
from shamu.protocols import ModelProvider, ProviderUnavailableError, ContextWindowExceededError
from shamu.gateway.router import ModelGateway, RoutingPolicy
from shamu.ui.window import MainWindow, UIBridge, ChatCard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class MockStreamingProvider(ModelProvider):
    """Configurable mock provider that streams text and optional tool calls."""

    def __init__(
        self,
        chunks: Optional[List[ChatChunk]] = None,
        fail_mid_stream: bool = False,
        fail_after_chunks: int = 1,
        exc_to_raise: Optional[Exception] = None,
    ):
        self.chunks = chunks or []
        self.fail_mid_stream = fail_mid_stream
        self.fail_after_chunks = fail_after_chunks
        self.exc_to_raise = exc_to_raise or ProviderUnavailableError("Simulated Ollama crash", provider="ollama")
        self.call_count = 0

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatChunk]:
        self.call_count += 1
        for idx, chunk in enumerate(self.chunks):
            if self.fail_mid_stream and idx >= self.fail_after_chunks:
                raise self.exc_to_raise
            yield chunk

        if self.fail_mid_stream and len(self.chunks) <= self.fail_after_chunks:
            raise self.exc_to_raise

    async def complete_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
        raise NotImplementedError()


def test_ui_window_clears_partial_tokens_on_turn_reset_signal(qapp):
    """UI bridge turn_reset / clear_partial clears partial tokens from failed model."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)

    # 1. Simulate partial tokens rendered from a failing provider
    window._on_chunk("The secret database password is: admin123")
    assert window.current_assistant_card is not None
    assert "admin123" in window.current_assistant_card.label.text()

    # 2. Provider fails mid-stream -> trigger turn_reset
    bridge.turn_reset.emit()
    assert window.current_assistant_card.label.text() == ""
    assert window.current_assistant_card._raw_text == ""

    # 3. Fallback provider streams clean text
    window._on_chunk("I cannot disclose private credentials.")
    window._on_stream_finished()

    # 4. Confirm corrupted partial text was completely purged and only clean final output remains
    assert "admin123" not in window.chat_container.findChild(ChatCard).label.text()
    assert "I cannot disclose private credentials." in window.chat_container.findChild(ChatCard).label.text()


def test_ui_window_clear_partial_cancels_pending_debounced_render(qapp):
    """clear_partial cancels active debouncing render timers on ChatCard."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)

    window._on_chunk("Starting **markdown** table...")
    card = window.current_assistant_card
    assert card is not None
    assert card._pending_markdown is True

    # Emit clear_partial
    bridge.clear_partial.emit()
    assert card._pending_markdown is False
    assert not card._render_timer.isActive()
    assert card.label.text() == ""
    assert card._raw_text == ""


@pytest.mark.asyncio
async def test_gateway_stream_chat_emits_bridge_signals_on_mid_stream_failover():
    """ModelGateway automatically notifies UIBridge with turn_reset / clear_partial on mid-stream failure."""
    failing_primary = MockStreamingProvider(
        chunks=[
            ChatChunk(text_delta="Incomplete sentence..."),
        ],
        fail_mid_stream=True,
        fail_after_chunks=1,
    )
    fallback_cloud = MockStreamingProvider(
        chunks=[
            ChatChunk(text_delta="Clean sentence from cloud.", finish_reason="stop"),
        ]
    )

    bridge = UIBridge()
    reset_events = []
    clear_events = []
    status_events = []

    bridge.turn_reset.connect(lambda: reset_events.append(True))
    bridge.clear_partial.connect(lambda: clear_events.append(True))
    bridge.status_updated.connect(lambda s: status_events.append(s))

    gateway = ModelGateway(
        ollama_provider=failing_primary,
        openrouter_provider=fallback_cloud,
        policy=RoutingPolicy.AUTO_FALLBACK,
        ui_bridge=bridge,
        emit_notice=False,
    )

    received_chunks: List[ChatChunk] = []
    async for chunk in gateway.stream_chat([ChatMessage(role=Role.USER, content="Hello")]):
        received_chunks.append(chunk)

    # Bridge signals must have fired
    assert len(reset_events) >= 1
    assert len(clear_events) >= 1
    assert any("Failing over" in s for s in status_events)

    # Chunk reset_turn must be True
    reset_chunks = [c for c in received_chunks if c.reset_turn]
    assert len(reset_chunks) == 1

    # Final chunk contains clean response
    assert received_chunks[-1].text_delta == "Clean sentence from cloud."


@pytest.mark.asyncio
async def test_streaming_failover_preserves_clean_tool_call_state():
    """Failover mid-stream purges partial/corrupted tool call deltas across boundary."""
    # Primary model emits partial tool call, then crashes mid-stream
    failing_primary = MockStreamingProvider(
        chunks=[
            ChatChunk(
                tool_call_delta=ToolCall(
                    id="call_corrupt_1",
                    name="dangerous_exec",
                    arguments='{"command": "rm -rf /',
                )
            ),
        ],
        fail_mid_stream=True,
        fail_after_chunks=1,
    )
    # Cloud fallback emits clean, distinct tool call
    fallback_cloud = MockStreamingProvider(
        chunks=[
            ChatChunk(
                tool_call_delta=ToolCall(
                    id="call_safe_2",
                    name="read_file",
                    arguments='{"path": "safe.txt"}',
                ),
                finish_reason="tool_calls",
            ),
        ]
    )

    gateway = ModelGateway(
        ollama_provider=failing_primary,
        openrouter_provider=fallback_cloud,
        policy=RoutingPolicy.AUTO_FALLBACK,
        emit_notice=False,
    )

    completed_tool_calls: List[ToolCall] = []
    current_streaming_call: Optional[ToolCall] = None
    full_reply: List[str] = []

    async for chunk in gateway.stream_chat([ChatMessage(role=Role.USER, content="Run task")]):
        if chunk.reset_turn:
            full_reply.clear()
            completed_tool_calls.clear()
            current_streaming_call = None

        if chunk.text_delta:
            full_reply.append(chunk.text_delta)

        if chunk.is_tool_call and chunk.tool_call_delta:
            tc = chunk.tool_call_delta
            if current_streaming_call is not None:
                completed_tool_calls.append(current_streaming_call)
            current_streaming_call = tc.model_copy()

    if current_streaming_call is not None:
        completed_tool_calls.append(current_streaming_call)
        current_streaming_call = None

    # Corrupted tool call from Ollama must NOT exist in completed tool calls
    assert len(completed_tool_calls) == 1
    assert completed_tool_calls[0].id == "call_safe_2"
    assert completed_tool_calls[0].name == "read_file"
    assert completed_tool_calls[0].arguments == '{"path": "safe.txt"}'


@pytest.mark.asyncio
async def test_regression_simulated_local_failure_cloud_failover_leaves_clean_ui_output(qapp):
    """CANON-P1-05 Regression: Simulated local model crash mid-stream leaves only clean cloud output in UI."""
    # Step 1: Prepare providers
    primary_ollama = MockStreamingProvider(
        chunks=[
            ChatChunk(text_delta="This is hallucinated or partial text that crashed"),
        ],
        fail_mid_stream=True,
        fail_after_chunks=1,
        exc_to_raise=ProviderUnavailableError("Ollama connection reset by peer", provider="ollama"),
    )
    fallback_cloud = MockStreamingProvider(
        chunks=[
            ChatChunk(text_delta="Paris is the capital of France.", finish_reason="stop"),
        ]
    )

    bridge = UIBridge()
    window = MainWindow(bridge=bridge)

    gateway = ModelGateway(
        ollama_provider=primary_ollama,
        openrouter_provider=fallback_cloud,
        policy=RoutingPolicy.AUTO_FALLBACK,
        ui_bridge=bridge,
        emit_notice=False,
    )

    # Step 2: Stream through gateway and bridge to window
    async for chunk in gateway.stream_chat([ChatMessage(role=Role.USER, content="What is the capital of France?")]):
        if chunk.reset_turn:
            # Emitted explicitly via bridge or handler
            bridge.clear_partial.emit()

        if chunk.text_delta:
            bridge.chunk_received.emit(chunk.text_delta)

    bridge.stream_finished.emit()

    # Step 3: Verify the UI state
    card = window.current_assistant_card or window.chat_container.findChildren(ChatCard)[-1]
    card_text = card.label.text()

    # Assert no remnant of the failed local response
    assert "hallucinated" not in card_text
    assert "partial text" not in card_text
    assert "crashed" not in card_text

    # Assert clean final output is present
    assert card_text == "Paris is the capital of France."
