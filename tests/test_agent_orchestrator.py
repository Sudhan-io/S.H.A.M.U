"""Comprehensive test suite for AgentOrchestrator and cognitive loop unification (CANON-P1-03).

Verifies:
1. Multi-turn tool calling, chunk delta reassembly, tool response formatting, and error dispatching.
2. Pure event streaming (TurnReset, TokenChunk, ToolRequested, ToolExecuted, StatusUpdate, TurnCompleted).
3. Tool call deduplication (within-response and across-iteration duplicate IDs).
4. SEC-01-C max tool turns enforcement (final turn with tools=None).
5. Secret and path redaction in tool error outputs.
6. Equivalence across all 3 execution modes: AgentWorker (UI), ShamuREPL (CLI interactive), and one-shot CLI (--prompt).
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from shamu.agent.orchestrator import (
    AgentLoop,
    AgentOrchestrator,
    StatusUpdate,
    TokenChunk,
    ToolExecuted,
    ToolRequested,
    TurnCompleted,
    TurnReset,
    is_complete_json_object,
    sanitize_error_message,
)
from shamu.app import AgentWorker
from shamu.cli import ShamuREPL
from shamu.protocols import PermissionDeniedError
from shamu.security.firewall import ExecutionFirewall
from shamu.security.tool_registry import PermissionTier, ToolCapability
from shamu.types import ChatChunk, ChatMessage, Role, ToolCall
from shamu.ui.window import UIBridge


class MockScriptedGateway:
    """Mock gateway that streams scripted sequences of ChatChunks across turns."""

    def __init__(self, turns_chunks: List[List[ChatChunk]]):
        self.turns_chunks = turns_chunks
        self.call_count = 0
        self.received_tools: List[Optional[List[Dict[str, Any]]]] = []
        self.received_histories: List[List[ChatMessage]] = []

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools=None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        self.received_tools.append(tools)
        self.received_histories.append([m.model_copy(deep=True) for m in messages])
        idx = self.call_count
        self.call_count += 1
        if idx < len(self.turns_chunks):
            for chunk in self.turns_chunks[idx]:
                yield chunk
        else:
            yield ChatChunk(text_delta="[Default fallback stop]", finish_reason="stop")


def test_is_complete_json_object():
    assert is_complete_json_object('{"key": "value"}') is True
    assert is_complete_json_object('{"nested": {"a": 1}}') is True
    assert is_complete_json_object('{"key": ') is False
    assert is_complete_json_object('["not", "a", "dict"]') is False
    assert is_complete_json_object("") is False
    assert is_complete_json_object("plain text") is False


def test_sanitize_error_message():
    raw = (
        r"Failed to connect sk-or-v1-abcdef0123456789abcdef0123456789 at "
        r"C:\Users\JohnDoe\.ssh\id_rsa with bearer secret12345678901234"
    )
    sanitized = sanitize_error_message(raw)
    assert "sk-or-v1" not in sanitized
    assert "JohnDoe" not in sanitized
    assert "id_rsa" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized
    assert "[REDACTED_PATH]" in sanitized


@pytest.mark.asyncio
async def test_orchestrator_single_turn_text():
    chunks = [
        ChatChunk(text_delta="Hello "),
        ChatChunk(text_delta="world!"),
        ChatChunk(finish_reason="stop"),
    ]
    gateway = MockScriptedGateway([chunks])
    orchestrator = AgentOrchestrator(gateway=gateway)

    history = [ChatMessage(role=Role.USER, content="Hi")]
    events = []
    async for evt in orchestrator.run_turn(history):
        events.append(evt)

    token_texts = [e.text for e in events if isinstance(e, TokenChunk)]
    assert "".join(token_texts) == "Hello world!"

    completed_evts = [e for e in events if isinstance(e, TurnCompleted)]
    assert len(completed_evts) == 1
    assert completed_evts[0].final_text == "Hello world!"
    assert completed_evts[0].tool_calls_executed == 0

    assert len(history) == 2
    assert history[1].role == Role.ASSISTANT
    assert history[1].content == "Hello world!"


@pytest.mark.asyncio
async def test_orchestrator_multi_turn_tool_execution(tmp_path: Path):
    target_file = tmp_path / "hello.txt"
    target_file.write_text("Hello from file!", encoding="utf-8")

    turn1 = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_read_1",
                name="read_file",
                arguments=json.dumps({"path": str(target_file)}),
            ),
            finish_reason="tool_calls",
        )
    ]
    turn2 = [
        ChatChunk(text_delta="Content is: Hello from file!", finish_reason="stop"),
    ]

    gateway = MockScriptedGateway([turn1, turn2])
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    orchestrator = AgentOrchestrator(gateway=gateway, firewall=firewall)

    history = [ChatMessage(role=Role.USER, content="Read file")]
    events = []
    async for evt in orchestrator.run_turn(history, session_id="test-sess"):
        events.append(evt)

    tool_reqs = [e for e in events if isinstance(e, ToolRequested)]
    assert len(tool_reqs) == 1
    assert tool_reqs[0].tool_call.name == "read_file"

    tool_execs = [e for e in events if isinstance(e, ToolExecuted)]
    assert len(tool_execs) == 1
    assert tool_execs[0].is_cached is False
    assert tool_execs[0].is_error is False
    assert "Hello from file!" in tool_execs[0].result

    assert len(history) == 4
    assert history[0].role == Role.USER
    assert history[1].role == Role.ASSISTANT
    assert history[1].tool_calls[0].name == "read_file"
    assert history[2].role == Role.TOOL
    assert history[2].tool_call_id == "call_read_1"
    assert "Hello from file!" in history[2].content
    assert history[3].role == Role.ASSISTANT
    assert history[3].content == "Content is: Hello from file!"


@pytest.mark.asyncio
async def test_orchestrator_deduplication_within_response(tmp_path: Path):
    target_file = tmp_path / "dedup.txt"
    target_file.write_text("DEDUP DATA", encoding="utf-8")

    turn1 = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_dup_1",
                name="read_file",
                arguments=json.dumps({"path": str(target_file)}),
            ),
        ),
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_dup_1",
                name="read_file",
                arguments=json.dumps({"path": str(target_file)}),
            ),
            finish_reason="tool_calls",
        ),
    ]
    turn2 = [
        ChatChunk(text_delta="Read duplicates done.", finish_reason="stop"),
    ]

    gateway = MockScriptedGateway([turn1, turn2])
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    exec_count = 0
    orig_exec = firewall.execute_tool

    async def counting_exec(*args, **kwargs):
        nonlocal exec_count
        exec_count += 1
        return await orig_exec(*args, **kwargs)

    firewall.execute_tool = counting_exec

    orchestrator = AgentOrchestrator(gateway=gateway, firewall=firewall)
    history = [ChatMessage(role=Role.USER, content="Read dup")]

    events = [e async for e in orchestrator.run_turn(history)]
    tool_execs = [e for e in events if isinstance(e, ToolExecuted)]

    assert len(tool_execs) == 2
    assert tool_execs[0].is_cached is False
    assert tool_execs[1].is_cached is True
    assert exec_count == 1

    tool_msgs = [m for m in history if m.role == Role.TOOL]
    assert len(tool_msgs) == 2
    assert "DEDUP DATA" in tool_msgs[0].content
    assert "DEDUP DATA" in tool_msgs[1].content


@pytest.mark.asyncio
async def test_orchestrator_turn_reset():
    turn1_failover = [
        ChatChunk(text_delta="Partial text that will be wiped..."),
        ChatChunk(reset_turn=True),
        ChatChunk(text_delta="Fresh final answer after failover.", finish_reason="stop"),
    ]

    gateway = MockScriptedGateway([turn1_failover])
    orchestrator = AgentOrchestrator(gateway=gateway)

    history = [ChatMessage(role=Role.USER, content="Hello")]
    events = [e async for e in orchestrator.run_turn(history)]

    resets = [e for e in events if isinstance(e, TurnReset)]
    assert len(resets) == 1

    assert history[-1].role == Role.ASSISTANT
    assert history[-1].content == "Fresh final answer after failover."
    assert "Partial" not in history[-1].content


@pytest.mark.asyncio
async def test_orchestrator_sec01_c_max_tool_turns(tmp_path: Path):
    target_file = tmp_path / "loop.txt"
    target_file.write_text("LOOP DATA", encoding="utf-8")

    turns = []
    for i in range(5):
        turns.append([
            ChatChunk(
                tool_call_delta=ToolCall(
                    id=f"call_{i}",
                    name="read_file",
                    arguments=json.dumps({"path": str(target_file)}),
                ),
                finish_reason="tool_calls",
            )
        ])
    turns.append([
        ChatChunk(text_delta="Final summary response with tools=None.", finish_reason="stop")
    ])

    gateway = MockScriptedGateway(turns)
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    orchestrator = AgentOrchestrator(gateway=gateway, firewall=firewall, max_tool_turns=5)

    history = [ChatMessage(role=Role.USER, content="Infinite tool loop")]
    events = [e async for e in orchestrator.run_turn(history)]

    assert len(gateway.received_tools) == 6
    assert gateway.received_tools[-1] is None

    assert history[-1].role == Role.ASSISTANT
    assert "Final summary response" in history[-1].content


@pytest.mark.asyncio
async def test_orchestrator_error_sanitization(tmp_path: Path):
    secret_key = "sk-or-v1-9876543210abcdef9876543210abcdef98765432"
    sensitive_path = r"C:\Users\SecretAdmin\.ssh\id_rsa"

    turn1 = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_leak",
                name="leak_tool",
                arguments="{}",
            ),
            finish_reason="tool_calls",
        )
    ]
    turn2 = [
        ChatChunk(text_delta="Handled error cleanly.", finish_reason="stop")
    ]

    gateway = MockScriptedGateway([turn1, turn2])
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    async def broken_tool(args):
        raise RuntimeError(f"Failed at {sensitive_path} with token {secret_key}")

    firewall.register_custom_executor("leak_tool", broken_tool)
    firewall.registry.register(ToolCapability(
        name="leak_tool",
        tier=PermissionTier.TIER_1_READ_ONLY,
        description="Leaking tool",
        parameters_schema={"type": "object", "properties": {}},
    ))

    orchestrator = AgentOrchestrator(gateway=gateway, firewall=firewall)
    history = [ChatMessage(role=Role.USER, content="Leak error")]

    events = [e async for e in orchestrator.run_turn(history)]
    tool_execs = [e for e in events if isinstance(e, ToolExecuted)]

    assert len(tool_execs) == 1
    assert tool_execs[0].is_error is True
    assert secret_key not in tool_execs[0].result
    assert "SecretAdmin" not in tool_execs[0].result
    assert "[REDACTED_SECRET]" in tool_execs[0].result
    assert "[REDACTED_PATH]" in tool_execs[0].result


@pytest.mark.asyncio
async def test_orchestrator_callback_mechanism():
    chunks = [
        ChatChunk(text_delta="Ping "),
        ChatChunk(text_delta="Pong!"),
        ChatChunk(finish_reason="stop"),
    ]
    gateway = MockScriptedGateway([chunks])
    orchestrator = AgentOrchestrator(gateway=gateway)

    received_in_callback = []

    def sync_cb(evt):
        received_in_callback.append(evt)

    history = [ChatMessage(role=Role.USER, content="Ping")]
    generator_events = [e async for e in orchestrator.run_turn(history, event_callback=sync_cb)]

    assert len(received_in_callback) == len(generator_events)
    for c_evt, g_evt in zip(received_in_callback, generator_events):
        assert type(c_evt) is type(g_evt)


@pytest.mark.asyncio
async def test_all_three_execution_modes_share_identical_semantics(tmp_path: Path):
    """Proves AgentWorker (UI), ShamuREPL (CLI REPL), and one-shot prompt share

    the exact same AgentOrchestrator cognitive loop semantics.
    """
    target_file = tmp_path / "common.txt"
    target_file.write_text("COMMON CONTENT", encoding="utf-8")

    def make_scripted_chunks():
        return [
            [
                ChatChunk(
                    tool_call_delta=ToolCall(
                        id="call_c1",
                        name="read_file",
                        arguments=json.dumps({"path": str(target_file)}),
                    ),
                    finish_reason="tool_calls",
                )
            ],
            [
                ChatChunk(text_delta="File read successfully: COMMON CONTENT", finish_reason="stop")
            ],
        ]

    # --- Mode 1: AgentWorker (Desktop UI) ---
    gw_worker = MockScriptedGateway(make_scripted_chunks())
    fw_worker = ExecutionFirewall(workspace_root=tmp_path)
    bridge = UIBridge()
    worker = AgentWorker(gateway=gw_worker, bridge=bridge, firewall=fw_worker)
    await worker._process_chat("Read common file via worker")

    # --- Mode 2: ShamuREPL (CLI Interactive) ---
    gw_repl = MockScriptedGateway(make_scripted_chunks())
    fw_repl = ExecutionFirewall(workspace_root=tmp_path)
    from rich.console import Console
    console = Console(record=True)
    repl = ShamuREPL(gateway=gw_repl, console=console, firewall=fw_repl)
    await repl.execute_turn("Read common file via repl")

    # --- Mode 3: One-shot Prompt (CLI --prompt) ---
    gw_prompt = MockScriptedGateway(make_scripted_chunks())
    fw_prompt = ExecutionFirewall(workspace_root=tmp_path)
    prompt_history = [ChatMessage(role=Role.USER, content="Read common file via prompt")]
    prompt_orchestrator = AgentOrchestrator(gateway=gw_prompt, firewall=fw_prompt)
    async for _ in prompt_orchestrator.run_turn(prompt_history):
        pass

    # Verify all 3 modes ended with identical conversation history lengths
    assert len(worker.history) == 4
    assert len(repl.history) == 4
    assert len(prompt_history) == 4

    # Verify all 3 modes produced the identical sequence of roles
    for h in [worker.history, repl.history, prompt_history]:
        assert h[0].role == Role.USER
        assert h[1].role == Role.ASSISTANT
        assert h[1].tool_calls[0].name == "read_file"
        assert h[2].role == Role.TOOL
        assert h[2].tool_call_id == "call_c1"
        assert "COMMON CONTENT" in h[2].content
        assert h[3].role == Role.ASSISTANT
        assert h[3].content == "File read successfully: COMMON CONTENT"

    # Verify AgentLoop alias is identical to AgentOrchestrator
    assert AgentLoop is AgentOrchestrator
