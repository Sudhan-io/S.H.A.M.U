"""Comprehensive integration tests for Milestone 1 repairs:
- M1.1: ExecutionFirewall memory injection in app.py and cli.py (DEF-01)
- M1.2: CLI HybridRetriever integration and prompt mode retrieval (DEF-05)
- M1.3: Active conversation session persistence across restarts (DEF-04)
- M1.4: GUI TTS streaming playback in AgentWorker (DEF-02)
- M1.5: CLI voice mode orchestration and tool execution (DEF-03)
"""

import asyncio
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from shamu.config import MemoryConfig, ShamuConfig
from shamu.types import ChatChunk, ChatMessage, MemoryRecord, Role, ToolCall
from shamu.security.firewall import ExecutionFirewall
from shamu.memory.sqlite_store import SQLiteMemoryStore
from shamu.memory.hybrid import HybridRetriever
from shamu.speech.pipeline import VoicePipeline
from shamu.app import AgentWorker, ShamuApp, UIBridge
from shamu.cli import ShamuREPL, async_main
from shamu.agent.orchestrator import (
    AgentOrchestrator,
    TokenChunk,
    ToolExecuted,
    ToolRequested,
    TurnReset,
)


# -----------------------------------------------------------------------------
# M1.1 Tests: Firewall Memory Injection (DEF-01)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m1_1_firewall_memory_injection_cli(tmp_path: Path):
    """M1.1: Ensure ShamuREPL injects memory_store into ExecutionFirewall."""
    cfg = MemoryConfig(db_path=str(tmp_path / "test_m1_1_cli.db"), fts_enabled=True, embedding_enabled=False)
    store = SQLiteMemoryStore(config=cfg)

    # Store a test memory record
    rec = MemoryRecord(
        id="mem-test-01",
        content="The user preferred dark theme and Python 3.12.",
        source="user_preference",
        category="preferences",
        metadata={},
    )
    await store.store(rec)

    mock_gateway = MagicMock()
    console = Console(record=True)
    fw = ExecutionFirewall(workspace_root=tmp_path)

    repl = ShamuREPL(
        gateway=mock_gateway,
        console=console,
        firewall=fw,
        memory_store=store,
    )

    # Verify firewall and executor have memory_store injected
    assert repl.firewall.memory_store is store
    assert repl.firewall.executor.memory_store is store

    # Verify search_local_history executes cleanly without error
    result = await repl.firewall.execute_tool(
        tool_name="search_local_history",
        arguments={"query": "dark theme"},
    )
    assert isinstance(result, list)
    assert len(result) >= 1
    assert "dark theme" in result[0].content


@pytest.mark.asyncio
async def test_m1_1_firewall_memory_injection_gui(tmp_path: Path):
    """M1.1: Ensure AgentWorker injects memory_store into ExecutionFirewall."""
    cfg = MemoryConfig(db_path=str(tmp_path / "test_m1_1_gui.db"), fts_enabled=True, embedding_enabled=False)
    store = SQLiteMemoryStore(config=cfg)

    rec = MemoryRecord(
        id="mem-test-02",
        content="Primary project workspace is located at D:/PROJECTS/S.H.A.M.U.",
        source="system_spec",
        category="facts",
        metadata={},
    )
    await store.store(rec)

    mock_gateway = MagicMock()
    bridge = UIBridge()
    fw = ExecutionFirewall(workspace_root=tmp_path)

    worker = AgentWorker(
        gateway=mock_gateway,
        bridge=bridge,
        firewall=fw,
        memory_store=store,
    )

    # Verify injection
    assert worker.firewall.memory_store is store
    assert worker.firewall.executor.memory_store is store

    # Verify search_local_history execution
    result = await worker.firewall.execute_tool(
        tool_name="search_local_history",
        arguments={"query": "workspace"},
    )
    assert isinstance(result, list)
    assert len(result) >= 1
    assert "S.H.A.M.U." in result[0].content


# -----------------------------------------------------------------------------
# M1.2 Tests: CLI HybridRetriever Integration (DEF-05)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m1_2_cli_hybrid_retrieval_turn_injection():
    """M1.2: Ensure ShamuREPL retrieves memories and injects untrusted context envelope."""
    mock_gateway = MagicMock()

    async def mock_stream_chat(history, tools=None):
        yield ChatChunk(text_delta="I see your GPU is RTX 3050.")

    mock_gateway.stream_chat = mock_stream_chat
    console = Console(record=True)

    # Mock retriever returning curated facts
    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=[
        MemoryRecord(
            id="gpu-fact",
            content="Local workstation has NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)",
            source="hardware_discovery",
            category="hardware",
            metadata={},
        )
    ])

    repl = ShamuREPL(
        gateway=mock_gateway,
        console=console,
        retriever=mock_retriever,
    )

    await repl.execute_turn("What GPU do I have?")

    # Verify retriever was invoked with limit=3
    mock_retriever.retrieve.assert_called_once_with("What GPU do I have?", limit=3)

    # Verify retrieved context was injected into the USER message as untrusted reference, NOT into Role.SYSTEM
    assert len(repl.history) == 2
    user_msg = repl.history[0]
    assert user_msg.role == Role.USER
    assert '<retrieved_context integrity="untrusted">' in user_msg.content
    assert "RTX 3050" in user_msg.content
    assert "What GPU do I have?" in user_msg.content
    assert repl.history[1].role == Role.ASSISTANT
    # Verify no Role.SYSTEM message was introduced or corrupted
    assert not any(m.role == Role.SYSTEM for m in repl.history)


# -----------------------------------------------------------------------------
# M1.3 Tests: Active Session Persistence Across Restarts (DEF-04)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m1_3_sqlite_active_session_persistence(tmp_path: Path):
    """M1.3: Ensure SQLiteMemoryStore saves, loads, and clears active sessions with tool calls."""
    cfg = MemoryConfig(db_path=str(tmp_path / "session_test.db"), fts_enabled=False, embedding_enabled=False)
    store = SQLiteMemoryStore(config=cfg)

    session_id = "test-session-42"
    messages = [
        ChatMessage(role=Role.USER, content="Please list files in the current directory."),
        ChatMessage(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="tc-101", name="list_directory", arguments='{"path": "."}')],
        ),
        ChatMessage(role=Role.TOOL, content="['README.md', 'pyproject.toml']", tool_call_id="tc-101"),
        ChatMessage(role=Role.ASSISTANT, content="The directory contains README.md and pyproject.toml."),
    ]

    # Save session
    await store.save_session(session_id, messages)

    # Load session
    loaded = await store.load_session(session_id)
    assert len(loaded) == 4
    assert loaded[0].role == Role.USER
    assert loaded[0].content == "Please list files in the current directory."

    assert loaded[1].role == Role.ASSISTANT
    assert loaded[1].tool_calls is not None
    assert len(loaded[1].tool_calls) == 1
    assert loaded[1].tool_calls[0].name == "list_directory"
    assert loaded[1].tool_calls[0].id == "tc-101"

    assert loaded[2].role == Role.TOOL
    assert loaded[2].tool_call_id == "tc-101"

    assert loaded[3].role == Role.ASSISTANT
    assert "README.md" in loaded[3].content

    # Clear session
    await store.clear_session(session_id)
    after_clear = await store.load_session(session_id)
    assert len(after_clear) == 0


@pytest.mark.asyncio
async def test_m1_3_cli_session_restoration_across_restarts(tmp_path: Path):
    """M1.3: Ensure ShamuREPL restores history across restart and clears on /clear."""
    cfg = MemoryConfig(db_path=str(tmp_path / "cli_restart.db"), fts_enabled=False, embedding_enabled=False)
    store = SQLiteMemoryStore(config=cfg)

    mock_gateway = MagicMock()

    async def mock_stream_chat(history, tools=None):
        yield ChatChunk(text_delta="Session reply acknowledged.")

    mock_gateway.stream_chat = mock_stream_chat
    console = Console(record=True)

    # First session instance: run turn
    repl1 = ShamuREPL(
        gateway=mock_gateway,
        console=console,
        memory_store=store,
        session_id="persisted_repl",
    )
    await repl1.execute_turn("First turn query.")
    assert len(repl1.history) >= 2

    # Second session instance (simulating restart with same SQLite store)
    repl2 = ShamuREPL(
        gateway=mock_gateway,
        console=console,
        memory_store=store,
        session_id="persisted_repl",
    )
    # History should have been restored on startup
    assert len(repl2.history) >= 2
    assert repl2.history[0].role == Role.USER
    assert repl2.history[0].content == "First turn query."
    assert repl2.history[1].role == Role.ASSISTANT

    # Execute /clear
    should_exit = repl2.handle_slash_command("/clear")
    assert not should_exit
    assert len(repl2.history) == 0

    # Third session instance: confirm cleared in database
    repl3 = ShamuREPL(
        gateway=mock_gateway,
        console=console,
        memory_store=store,
        session_id="persisted_repl",
    )
    assert len(repl3.history) == 0


# -----------------------------------------------------------------------------
# M1.4 Tests: GUI TTS Streaming Playback (DEF-02)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m1_4_gui_tts_streaming_playback():
    """M1.4: Ensure AgentWorker streams sentences to TTS during chat turn."""
    mock_gateway = MagicMock()

    # Stream full sentence across tokens
    async def mock_stream_chat(history, tools=None):
        yield ChatChunk(text_delta="Good morning! ")
        yield ChatChunk(text_delta="The workstation is operating within nominal parameters. ")
        yield ChatChunk(text_delta="All services are green.")

    mock_gateway.stream_chat = mock_stream_chat
    bridge = UIBridge()

    # Mock VoicePipeline and TextToSpeech
    mock_voice = MagicMock(spec=VoicePipeline)
    mock_voice.tts = MagicMock()
    synthesized_sentences = []

    async def mock_synthesize_stream(text: str):
        synthesized_sentences.append(text)
        yield b"RIFF_FAKE_AUDIO_DATA"

    mock_voice.tts.synthesize_stream = mock_synthesize_stream
    mock_voice.enqueue_audio = AsyncMock()
    mock_voice.stop_playback = AsyncMock()

    worker = AgentWorker(
        gateway=mock_gateway,
        bridge=bridge,
        voice_pipeline=mock_voice,
    )

    await worker._process_chat("Status check")

    # Verify that TTS was called with completed sentences/clauses
    assert len(synthesized_sentences) >= 2
    assert mock_voice.enqueue_audio.call_count >= 2


@pytest.mark.asyncio
async def test_m1_4_gui_tts_turn_reset_barge_in():
    """M1.4: Ensure AgentWorker halts audio playback when TurnReset occurs."""
    mock_gateway = MagicMock()

    reset_emitted = False

    async def mock_stream_chat(history, tools=None):
        nonlocal reset_emitted
        if not reset_emitted:
            reset_emitted = True
            yield TurnReset()
        yield ChatChunk(text_delta="Fresh answer after retry.")

    mock_gateway.stream_chat = mock_stream_chat
    bridge = UIBridge()

    mock_voice = MagicMock(spec=VoicePipeline)
    mock_voice.tts = MagicMock()

    async def mock_synthesize_stream(text: str):
        yield b"RIFF_AUDIO"

    mock_voice.tts.synthesize_stream = mock_synthesize_stream
    mock_voice.enqueue_audio = AsyncMock()
    mock_voice.stop_playback = AsyncMock()

    worker = AgentWorker(
        gateway=mock_gateway,
        bridge=bridge,
        voice_pipeline=mock_voice,
    )

    await worker._process_chat("Trigger reset test")

    # Verify stop_playback was called when TurnReset occurred
    assert mock_voice.stop_playback.call_count >= 1


# -----------------------------------------------------------------------------
# M1.5 Tests: CLI Voice Mode Orchestration & Tools (DEF-03)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m1_5_cli_voice_mode_routes_through_orchestrator():
    """M1.5: Ensure CLI voice mode routes through AgentOrchestrator and executes tools."""
    mock_gateway = MagicMock()
    mock_firewall = MagicMock(spec=ExecutionFirewall)
    mock_firewall.approval_manager = MagicMock()
    mock_firewall.tool_registry = MagicMock()
    mock_firewall.tool_registry.get_tier.return_value = 0
    mock_firewall.get_tool_definitions.return_value = [{"name": "get_system_metrics"}]
    mock_firewall.execute_tool = AsyncMock(return_value={"cpu_percent": 12.5, "ram_percent": 45.0})

    call_count = 0

    async def mock_stream_chat(history, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ChatChunk(
                tool_call_delta=ToolCall(id="call_v1", name="get_system_metrics", arguments="{}")
            )
        else:
            yield ChatChunk(text_delta="System metrics: CPU is 12.5% and RAM is 45.0%.")

    mock_gateway.stream_chat = mock_stream_chat
    console = Console(record=True)

    # Mock voice pipeline
    mock_voice = MagicMock(spec=VoicePipeline)
    mock_voice.tts = MagicMock()
    spoken_audio = []

    async def mock_synthesize_stream(text: str):
        spoken_audio.append(text)
        yield b"VOICE_AUDIO_BYTES"

    mock_voice.tts.synthesize_stream = mock_synthesize_stream
    mock_voice.enqueue_audio = AsyncMock()
    mock_voice.wait_for_playback = AsyncMock()
    mock_voice.stop_playback = AsyncMock()

    repl = ShamuREPL(
        gateway=mock_gateway,
        console=console,
        voice_mode=True,
        firewall=mock_firewall,
    )
    repl.voice_pipeline = mock_voice

    await repl.execute_turn("Check system health")

    # 1. Verify firewall executed the tool during voice mode (DEF-03 fixed!)
    mock_firewall.execute_tool.assert_called_once_with(
        tool_name="get_system_metrics",
        arguments={},
    )

    # 2. Verify TTS synthesized the final answer
    assert mock_voice.enqueue_audio.call_count >= 1
    assert mock_voice.wait_for_playback.call_count >= 1

    # 3. Verify history integrity
    assert len(repl.history) == 4
    assert repl.history[0].role == Role.USER
    assert repl.history[1].role == Role.ASSISTANT
    assert repl.history[1].tool_calls[0].name == "get_system_metrics"
    assert repl.history[2].role == Role.TOOL
    assert repl.history[3].role == Role.ASSISTANT
    assert "System metrics" in repl.history[3].content
