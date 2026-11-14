import asyncio
import json
from pathlib import Path
from typing import List, Optional
import pytest
from PySide6.QtWidgets import QApplication

from shamu.app import AgentWorker, sanitize_error_message
from shamu.protocols import PermissionDeniedError
from shamu.security.approval_manager import ApprovalRequest, ApprovalResponse
from shamu.security.firewall import ExecutionFirewall
from shamu.types import ChatChunk, ChatMessage, Role, ToolCall
from shamu.ui.window import UIBridge


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class MockStreamingGateway:
    """Mock gateway that returns scripted sequences of ChatChunks across turns."""

    def __init__(self, turns_chunks: List[List[ChatChunk]]):
        self.turns_chunks = turns_chunks
        self.call_count = 0
        self.received_tools = []
        self.received_histories = []

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools=None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        self.received_histories.append([m.model_copy(deep=True) for m in messages])
        self.received_tools.append(tools)

        if self.call_count < len(self.turns_chunks):
            chunks = self.turns_chunks[self.call_count]
            self.call_count += 1
            for c in chunks:
                yield c
        else:
            yield ChatChunk(text_delta="Done", finish_reason="stop")


@pytest.mark.asyncio
async def test_ordinary_chat_without_tools(qapp, tmp_path: Path):
    """Ordinary conversation without tools should function normally with chunks emitted to UI."""
    bridge = UIBridge()
    received_ui_chunks = []
    bridge.chunk_received.connect(lambda text: received_ui_chunks.append(text))

    chunks = [
        ChatChunk(text_delta="Hello! "),
        ChatChunk(text_delta="How can I help you today?"),
        ChatChunk(finish_reason="stop"),
    ]
    gateway = MockStreamingGateway([chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("Hello SHAMU")

    # UI received tokens
    assert "".join(received_ui_chunks) == "Hello! How can I help you today?"
    # History has exactly user and assistant turns
    assert len(worker.history) == 2
    assert worker.history[0].role == Role.USER
    assert worker.history[0].content == "Hello SHAMU"
    assert worker.history[1].role == Role.ASSISTANT
    assert worker.history[1].content == "Hello! How can I help you today?"
    # Firewall audit ledger has 0 events (no tools invoked)
    assert firewall.audit_ledger.verify_integrity().verified_records == 0


@pytest.mark.asyncio
async def test_permitted_tools_passed_to_gateway(qapp, tmp_path: Path):
    """Gateway must be supplied with authorized tool definitions, excluding prohibited tools."""
    bridge = UIBridge()
    chunks = [ChatChunk(text_delta="Hi", finish_reason="stop")]
    gateway = MockStreamingGateway([chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("Hi")

    assert len(gateway.received_tools) > 0
    passed_tools = gateway.received_tools[0]
    assert passed_tools is not None
    assert len(passed_tools) > 0

    tool_names = [t.name for t in passed_tools]
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "powershell_exec" in tool_names

    # Hard-prohibited tools MUST NEVER be passed to the model
    assert "format_disk" not in tool_names
    assert "reg_delete" not in tool_names
    assert "disable_firewall" not in tool_names
    assert "elevate_uac" not in tool_names


@pytest.mark.asyncio
async def test_valid_tool_call_executed_through_firewall(qapp, tmp_path: Path):
    """Model tool call must be intercepted and executed via ExecutionFirewall, feeding result back."""
    target_file = tmp_path / "secret.txt"
    target_file.write_text("TOP SECRET CONTENT", encoding="utf-8")

    bridge = UIBridge()
    # Turn 1: model requests read_file
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_read_1",
                name="read_file",
                arguments=json.dumps({"path": str(target_file)}),
            ),
            finish_reason="tool_calls",
        )
    ]
    # Turn 2: model uses tool result to produce final answer
    turn2_chunks = [
        ChatChunk(text_delta="The secret file says: TOP SECRET CONTENT"),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("What is in secret.txt?")

    # Verify history progression: USER -> ASSISTANT(tool_calls) -> TOOL(result) -> ASSISTANT(final)
    assert len(worker.history) == 4
    assert worker.history[0].role == Role.USER
    assert worker.history[1].role == Role.ASSISTANT
    assert worker.history[1].tool_calls is not None
    assert worker.history[1].tool_calls[0].name == "read_file"
    assert worker.history[2].role == Role.TOOL
    assert worker.history[2].content == "TOP SECRET CONTENT"
    assert worker.history[2].name == "read_file"
    assert worker.history[3].role == Role.ASSISTANT
    assert "TOP SECRET CONTENT" in worker.history[3].content

    # Verify audit ledger recorded the tool execution (intent + completion)
    ledger_res = firewall.audit_ledger.verify_integrity()
    assert ledger_res.is_valid is True
    assert ledger_res.verified_records == 2


@pytest.mark.asyncio
async def test_unknown_tool_rejected_and_audited(qapp, tmp_path: Path):
    """Unknown or unapproved tool calls must fail closed, return error safely, and be logged to audit."""
    bridge = UIBridge()
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_bad_1",
                name="unregistered_hack_tool",
                arguments=json.dumps({"target": "system"}),
            ),
            finish_reason="tool_calls",
        )
    ]
    turn2_chunks = [
        ChatChunk(text_delta="I could not execute that tool."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("Run hack")

    assert len(worker.history) == 4
    tool_msg = worker.history[2]
    assert tool_msg.role == Role.TOOL
    assert "Permission denied" in tool_msg.content or "not registered" in tool_msg.content

    # Audit ledger recorded DENY
    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records == 1


@pytest.mark.asyncio
async def test_malformed_arguments_rejected(qapp, tmp_path: Path):
    """Malformed non-JSON arguments must not crash the worker or execute any tool."""
    bridge = UIBridge()
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_malformed",
                name="read_file",
                arguments="NOT_A_VALID_JSON{",
            ),
            finish_reason="tool_calls",
        )
    ]
    turn2_chunks = [
        ChatChunk(text_delta="Failed to call tool due to malformed args."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("Corrupt args call")

    assert len(worker.history) == 4
    tool_msg = worker.history[2]
    assert tool_msg.role == Role.TOOL
    assert "JSON" in tool_msg.content or "Error" in tool_msg.content or "validation" in tool_msg.content.lower()


@pytest.mark.asyncio
async def test_denied_approval_aborts_execution(qapp, tmp_path: Path):
    """When human user or policy denies an approval request, tool must not execute."""
    dest_file = tmp_path / "never_write.txt"
    bridge = UIBridge()

    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_write_1",
                name="write_file",
                arguments=json.dumps({"path": str(dest_file), "content": "malicious"}),
            ),
            finish_reason="tool_calls",
        )
    ]
    turn2_chunks = [
        ChatChunk(text_delta="Write was denied."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    # Register denying approval handler
    async def deny_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(request_id=req.request_id, approved=False, reason="User rejected file write")

    firewall.approval_manager.register_handler(deny_handler)

    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)
    await worker._process_chat("Write this file")

    # File was not created
    assert not dest_file.exists()
    tool_msg = worker.history[2]
    assert tool_msg.role == Role.TOOL
    assert "denied" in tool_msg.content.lower()


@pytest.mark.asyncio
async def test_streaming_duplicate_execution_prevention(qapp, tmp_path: Path):
    """During streaming, multiple partial chunks for the same tool call must not trigger multiple executions."""
    target_file = tmp_path / "stream_read.txt"
    target_file.write_text("STREAM TEST", encoding="utf-8")

    bridge = UIBridge()
    # 4 partial streaming chunks for one single tool call
    turn1_chunks = [
        ChatChunk(tool_call_delta=ToolCall(id="call_s1", name="read_file", arguments='{"pa')),
        ChatChunk(tool_call_delta=ToolCall(id="call_s1", name="read_file", arguments='{"path": "')),
        ChatChunk(tool_call_delta=ToolCall(id="call_s1", name="read_file", arguments=json.dumps({"path": str(target_file)}))),
        ChatChunk(finish_reason="tool_calls"),
    ]
    turn2_chunks = [
        ChatChunk(text_delta="Read completed."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("Stream read")

    # Audit ledger must have exactly 2 records (intent + completion), not 6 or 8
    ledger_res = firewall.audit_ledger.verify_integrity()
    assert ledger_res.is_valid is True
    assert ledger_res.verified_records == 2


@pytest.mark.asyncio
async def test_prohibited_tool_call_hard_denied(qapp, tmp_path: Path):
    """Prohibited actions like format_disk or reg_delete are hard-denied and audited without execution."""
    bridge = UIBridge()
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_prohibited_1",
                name="format_disk",
                arguments=json.dumps({"drive": "C:"}),
            ),
            finish_reason="tool_calls",
        )
    ]
    turn2_chunks = [
        ChatChunk(text_delta="Prohibited action was blocked."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("Format the disk")

    assert len(worker.history) == 4
    tool_msg = worker.history[2]
    assert tool_msg.role == Role.TOOL
    assert "permanently prohibited" in tool_msg.content.lower()

    # Audit ledger recorded the hard DENY
    ledger_res = firewall.audit_ledger.verify_integrity()
    assert ledger_res.is_valid is True
    assert ledger_res.verified_records == 1


@pytest.mark.asyncio
async def test_tool_execution_failure_safe_delivery(qapp, tmp_path: Path):
    """When a tool raises an unexpected runtime exception, it is caught safely without leaking secrets or crashing."""
    bridge = UIBridge()
    # Read a non-existent file in workspace
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_missing_file",
                name="read_file",
                arguments=json.dumps({"path": str(tmp_path / "nonexistent.txt")}),
            ),
            finish_reason="tool_calls",
        )
    ]
    turn2_chunks = [
        ChatChunk(text_delta="The file could not be found."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("Read missing")

    assert len(worker.history) == 4
    tool_msg = worker.history[2]
    assert tool_msg.role == Role.TOOL
    assert "error" in tool_msg.content.lower()
    # Worker is still in Ready status and stream finished safely
    assert worker.history[3].role == Role.ASSISTANT


@pytest.mark.asyncio
async def test_sec01_a_duplicate_tool_call_in_same_turn_preserves_protocol(qapp, tmp_path: Path):
    """When a model declares two tool calls with the same ID, side effects run once, but protocol requires two corresponding tool responses."""
    target_file = tmp_path / "dup_test.txt"
    target_file.write_text("DUP DATA", encoding="utf-8")

    bridge = UIBridge()
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="dup_1",
                name="read_file",
                arguments=json.dumps({"path": str(target_file)}),
            )
        ),
        # Second distinct call with repeated ID
        ChatChunk(
            tool_call_delta=ToolCall(
                id="dup_1",
                name="read_file",
                arguments=json.dumps({"path": str(target_file)}),
            ),
            finish_reason="tool_calls",
        ),
    ]
    turn2_chunks = [
        ChatChunk(text_delta="Both reads handled."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    execution_count = 0
    orig_execute = firewall.execute_tool

    async def counting_execute(tool_name, arguments, session_id="default", turn_id="turn-0"):
        nonlocal execution_count
        execution_count += 1
        return await orig_execute(tool_name, arguments, session_id, turn_id)

    firewall.execute_tool = counting_execute

    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)
    await worker._process_chat("Read duplicate")

    # Assistant message declared 2 tool calls
    asst_msg = worker.history[1]
    assert asst_msg.role == Role.ASSISTANT
    assert len(asst_msg.tool_calls) == 2

    # Tool side effect executed strictly ONCE
    assert execution_count == 1

    # Every declared tool call MUST have a corresponding tool result message in history
    tool_msgs = [m for m in worker.history if m.role == Role.TOOL]
    assert len(tool_msgs) == 2
    assert tool_msgs[0].tool_call_id == "dup_1"
    assert tool_msgs[1].tool_call_id == "dup_1"
    assert "DUP DATA" in tool_msgs[0].content
    assert "DUP DATA" in tool_msgs[1].content or "cached" in tool_msgs[1].content.lower() or "previously" in tool_msgs[1].content.lower()


@pytest.mark.asyncio
async def test_sec01_b_distinct_calls_without_ids_not_overwritten(qapp, tmp_path: Path):
    """When a provider omits tool IDs, multiple distinct calls to the same tool must not overwrite each other."""
    f1 = tmp_path / "file1.txt"
    f1.write_text("CONTENT 1", encoding="utf-8")
    f2 = tmp_path / "file2.txt"
    f2.write_text("CONTENT 2", encoding="utf-8")

    bridge = UIBridge()
    # Distinct tool calls both having id=""
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="",
                name="read_file",
                arguments=json.dumps({"path": str(f1)}),
            )
        ),
        ChatChunk(
            tool_call_delta=ToolCall(
                id="",
                name="read_file",
                arguments=json.dumps({"path": str(f2)}),
            ),
            finish_reason="tool_calls",
        ),
    ]
    turn2_chunks = [
        ChatChunk(text_delta="Read both files."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("Read both")

    # Both calls must be preserved in assistant message
    asst_msg = worker.history[1]
    assert asst_msg.role == Role.ASSISTANT
    assert len(asst_msg.tool_calls) == 2
    assert "file1.txt" in asst_msg.tool_calls[0].arguments
    assert "file2.txt" in asst_msg.tool_calls[1].arguments

    # Both tool results must be present in history
    tool_msgs = [m for m in worker.history if m.role == Role.TOOL]
    assert len(tool_msgs) == 2
    assert "CONTENT 1" in tool_msgs[0].content
    assert "CONTENT 2" in tool_msgs[1].content


@pytest.mark.asyncio
async def test_sec01_c_tool_turn_limit_triggers_final_response(qapp, tmp_path: Path):
    """When max_tool_turns (5) is reached, worker must request a final assistant response with tools=None, preventing silent truncation."""
    f = tmp_path / "loop.txt"
    f.write_text("LOOP DATA", encoding="utf-8")

    bridge = UIBridge()

    # Generate 5 turns of continuous tool calling
    turns = []
    for i in range(5):
        turns.append([
            ChatChunk(
                tool_call_delta=ToolCall(
                    id=f"call_{i}",
                    name="read_file",
                    arguments=json.dumps({"path": str(f)}),
                ),
                finish_reason="tool_calls",
            )
        ])
    # Final turn 6 when tools=None is passed
    turns.append([
        ChatChunk(text_delta="Final summary after 5 tool turns reached."),
        ChatChunk(finish_reason="stop"),
    ] * 2)

    gateway = MockStreamingGateway(turns)
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("Run loop")

    # Verify that the final call was made with tools=None
    assert len(gateway.received_tools) == 6
    assert gateway.received_tools[-1] is None

    # History must end with an ASSISTANT message, not dangling TOOL messages
    assert worker.history[-1].role == Role.ASSISTANT
    assert "Final summary" in worker.history[-1].content


@pytest.mark.asyncio
async def test_sec01_d_error_message_sanitization(qapp, tmp_path: Path):
    """Exceptions containing API keys and private user paths must be sanitized before being added to history or emitted."""
    bridge = UIBridge()

    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_leak",
                name="custom_leaker",
                arguments="{}",
            ),
            finish_reason="tool_calls",
        )
    ]
    turn2_chunks = [
        ChatChunk(text_delta="Received error."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    # Register custom leaker that throws an exception with secrets and sensitive paths
    secret_key = "sk-or-v1-9876543210abcdef9876543210abcdef98765432"
    sensitive_path = r"C:\Users\SecretAdmin\.ssh\id_rsa"

    async def leaky_tool(args):
        raise RuntimeError(f"Failed to access {sensitive_path} with token {secret_key}")

    firewall.register_custom_executor("custom_leaker", leaky_tool)

    # Also register capability in registry so it passes validation
    from shamu.security.tool_registry import ToolCapability, PermissionTier
    firewall.registry.register(ToolCapability(
        name="custom_leaker",
        tier=PermissionTier.TIER_1_READ_ONLY,
        description="Leaky test tool",
        parameters_schema={"type": "object", "properties": {}},
    ))

    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)
    await worker._process_chat("Leak secrets")

    tool_msg = worker.history[2]
    assert tool_msg.role == Role.TOOL

    # The raw secret key MUST NOT appear in the message content
    assert secret_key not in tool_msg.content
    # The sensitive local username/ssh path MUST NOT appear in the message content
    assert "SecretAdmin" not in tool_msg.content
    assert r".ssh\id_rsa" not in tool_msg.content
    assert "[REDACTED" in tool_msg.content


@pytest.mark.asyncio
async def test_sec01_a_repeated_id_across_iterations(qapp, tmp_path: Path):
    """When a model hallucinates and repeats a tool call ID in a subsequent loop iteration, side-effects are not repeated and protocol is preserved."""
    target_file = tmp_path / "iter_test.txt"
    target_file.write_text("ITERATION DATA", encoding="utf-8")

    bridge = UIBridge()
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_repeat_iter",
                name="read_file",
                arguments=json.dumps({"path": str(target_file)}),
            ),
            finish_reason="tool_calls",
        )
    ]
    # Iteration 2: repeats the exact same tool call ID
    turn2_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_repeat_iter",
                name="read_file",
                arguments=json.dumps({"path": str(target_file)}),
            ),
            finish_reason="tool_calls",
        )
    ]
    # Iteration 3: final text
    turn3_chunks = [
        ChatChunk(text_delta="Completed after repeated iteration."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks, turn3_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    exec_count = 0
    orig_exec = firewall.execute_tool

    async def counting_exec(tool_name, arguments, session_id="default", turn_id="turn-0"):
        nonlocal exec_count
        exec_count += 1
        return await orig_exec(tool_name, arguments, session_id=session_id, turn_id=turn_id)

    firewall.execute_tool = counting_exec

    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)
    await worker._process_chat("Repeat across iterations")

    # Side-effect ran only once
    assert exec_count == 1

    # Both iterations have their tool response messages matching tool_call_id
    tool_msgs = [m for m in worker.history if m.role == Role.TOOL]
    assert len(tool_msgs) == 2
    assert tool_msgs[0].tool_call_id == "call_repeat_iter"
    assert tool_msgs[1].tool_call_id == "call_repeat_iter"

    # Final assistant message received
    assert worker.history[-1].role == Role.ASSISTANT
    assert "Completed after repeated iteration." in worker.history[-1].content


@pytest.mark.asyncio
async def test_sec01_c_normal_completion_before_limit(qapp, tmp_path: Path):
    """When a model finishes naturally on turn 2, the extra tools=None final turn is not triggered."""
    f = tmp_path / "normal.txt"
    f.write_text("NORMAL DATA", encoding="utf-8")

    bridge = UIBridge()
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="c_norm",
                name="read_file",
                arguments=json.dumps({"path": str(f)}),
            ),
            finish_reason="tool_calls",
        )
    ]
    turn2_chunks = [
        ChatChunk(text_delta="Read normal data."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)
    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("Read normal")

    # Only 2 gateway calls occurred
    assert len(gateway.received_tools) == 2
    # Both had the tool definitions enabled
    assert gateway.received_tools[0] is not None
    assert gateway.received_tools[1] is not None
    assert worker.history[-1].content == "Read normal data."


def test_finding_a_forward_slash_and_mixed_path_sanitization():
    """Finding A: sanitize_error_message must redact Windows usernames across backslashes, forward slashes, mixed separators, and case variations."""
    cases = [
        ("C:\\Users\\JohnDoe\\Documents\\secret.txt", "JohnDoe"),
        ("C:/Users/JohnDoe/Documents/secret.txt", "JohnDoe"),
        ("c:\\users\\johndoe/downloads\\secret.txt", "johndoe"),
        ("C:/users\\JohnDoe/secret.txt", "JohnDoe"),
        ("D:/Users/JaneDoe/.ssh/id_rsa", "JaneDoe"),
        ("c:/USERS/AdminUser/AppData/Local/Temp", "AdminUser"),
        ("E:\\users\\mixed_user/file.txt", "mixed_user"),
        ("C:/Users/Spaced Name/secret.txt", "Spaced Name"),
    ]

    for raw_path, username in cases:
        sanitized = sanitize_error_message(f"Error accessing {raw_path}")
        assert username not in sanitized, f"Username '{username}' was retained in: {sanitized}"
        assert "[REDACTED_PATH]" in sanitized

    # Also check sensitive files with forward/mixed slashes
    sensitive_cases = [
        "C:/Users/Alice/.ssh/id_rsa",
        "C:/Users/Bob/.aws/credentials",
        "c:/Users/Charlie/.env.production",
        "d:/Users/Dave/.ssh/id_ed25519",
    ]
    for p in sensitive_cases:
        sanitized = sanitize_error_message(f"Failed opening {p}")
        assert ".ssh" not in sanitized
        assert ".aws" not in sanitized
        assert ".env" not in sanitized
        assert "id_rsa" not in sanitized
        assert "id_ed25519" not in sanitized
        assert "credentials" not in sanitized
        assert "[REDACTED_PATH]" in sanitized


@pytest.mark.asyncio
async def test_finding_a_agent_worker_forward_slash_sanitization(qapp, tmp_path: Path):
    """AgentWorker._process_chat must redact forward-slash user paths when exceptions occur."""
    bridge = UIBridge()
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="call_leak_fwd",
                name="forward_leaker",
                arguments="{}",
            ),
            finish_reason="tool_calls",
        )
    ]
    turn2_chunks = [
        ChatChunk(text_delta="Done"),
        ChatChunk(finish_reason="stop"),
    ]
    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    fwd_path = "C:/Users/ForwardUser/Documents/secret.txt"
    fwd_ssh = "C:/Users/ForwardUser/.ssh/id_rsa"

    async def forward_leak_tool(args):
        raise RuntimeError(f"Permission denied accessing {fwd_path} and {fwd_ssh}")

    firewall.register_custom_executor("forward_leaker", forward_leak_tool)
    from shamu.security.tool_registry import ToolCapability, PermissionTier
    firewall.registry.register(ToolCapability(
        name="forward_leaker",
        tier=PermissionTier.TIER_1_READ_ONLY,
        description="Forward leaker tool",
        parameters_schema={"type": "object", "properties": {}},
    ))

    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)
    await worker._process_chat("Run forward leaker")

    tool_msg = [m for m in worker.history if m.role == Role.TOOL][0]
    assert "ForwardUser" not in tool_msg.content
    assert ".ssh" not in tool_msg.content
    assert "id_rsa" not in tool_msg.content
    assert "[REDACTED_PATH]" in tool_msg.content


@pytest.mark.asyncio
async def test_finding_b_stale_cached_tool_results_across_iterations(qapp, tmp_path: Path):
    """Finding B: Provider omitting tool IDs must not cause stale cached results in later iterations.
    
    1. read_file('config.json') -> returns 'v1'
    2. write_file('config.json', 'v2') -> updates file
    3. read_file('config.json') -> must execute again and return 'v2', not cached 'v1'
    """
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("v1", encoding="utf-8")

    bridge = UIBridge()

    # Turn 1: Model calls read_file with NO tool ID (e.g. Ollama provider)
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="",
                name="read_file",
                arguments=json.dumps({"path": str(cfg_file)}),
            ),
            finish_reason="tool_calls",
        )
    ]
    # Turn 2: Model calls write_file to modify config.json with NO tool ID
    turn2_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="",
                name="write_file",
                arguments=json.dumps({"path": str(cfg_file), "content": "v2"}),
            ),
            finish_reason="tool_calls",
        )
    ]
    # Turn 3: Model calls read_file AGAIN on config.json with NO tool ID
    turn3_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="",
                name="read_file",
                arguments=json.dumps({"path": str(cfg_file)}),
            ),
            finish_reason="tool_calls",
        )
    ]
    # Turn 4: Final response
    turn4_chunks = [
        ChatChunk(text_delta="Updated and verified v2 content successfully."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks, turn3_chunks, turn4_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    async def approve_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(request_id=req.request_id, approved=True, reason="Approved for test")

    firewall.approval_manager.register_handler(approve_handler)

    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)

    await worker._process_chat("Update config and verify")

    tool_msgs = [m for m in worker.history if m.role == Role.TOOL]
    assert len(tool_msgs) == 3

    # Turn 1 tool result was v1
    assert tool_msgs[0].name == "read_file"
    assert "v1" in tool_msgs[0].content

    # Turn 2 tool result was write confirmation
    assert tool_msgs[1].name == "write_file"

    # Turn 3 tool result MUST be v2, NOT stale cached v1!
    assert tool_msgs[2].name == "read_file"
    assert "v2" in tool_msgs[2].content, f"Expected 'v2' in turn 3 result, got: {tool_msgs[2].content}"


@pytest.mark.asyncio
async def test_finding_b_within_response_dedup_without_ids(qapp, tmp_path: Path):
    """Duplicate calls without IDs within the SAME assistant response must not execute twice."""
    target_file = tmp_path / "within_response.txt"
    target_file.write_text("INITIAL", encoding="utf-8")

    bridge = UIBridge()
    # Assistant emits two identical tool calls without IDs in the SAME turn
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(
                id="",
                name="read_file",
                arguments=json.dumps({"path": str(target_file)}),
            ),
            finish_reason=None,
        ),
        ChatChunk(
            tool_call_delta=ToolCall(
                id="",
                name="read_file",
                arguments=json.dumps({"path": str(target_file)}),
            ),
            finish_reason="tool_calls",
        ),
    ]
    turn2_chunks = [
        ChatChunk(text_delta="Done reading."),
        ChatChunk(finish_reason="stop"),
    ]

    gateway = MockStreamingGateway([turn1_chunks, turn2_chunks])
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    exec_count = 0
    orig_exec = firewall.execute_tool

    async def counting_exec(tool_name, arguments, session_id="default", turn_id="turn-0"):
        nonlocal exec_count
        exec_count += 1
        return await orig_exec(tool_name, arguments, session_id=session_id, turn_id=turn_id)

    firewall.execute_tool = counting_exec

    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)
    await worker._process_chat("Read duplicate in same response")

    # Only 1 execution occurred despite 2 identical tool calls in the response
    assert exec_count == 1

    # But both tool responses exist in history
    tool_msgs = [m for m in worker.history if m.role == Role.TOOL]
    assert len(tool_msgs) == 2
    assert "INITIAL" in tool_msgs[0].content
    assert "INITIAL" in tool_msgs[1].content


@pytest.mark.asyncio
async def test_agent_worker_voice_listen_integration(qapp):
    """ARCH-03: AgentWorker.submit_listen triggers voice pipeline and emits user message."""
    from unittest.mock import AsyncMock, MagicMock
    from shamu.speech.pipeline import VoicePipeline

    bridge = UIBridge()
    emitted_user_messages = []
    bridge.user_message_sent.connect(lambda msg: emitted_user_messages.append(msg))

    mock_voice = MagicMock(spec=VoicePipeline)
    mock_voice.listen_and_transcribe = AsyncMock(return_value="What is the weather today?")

    mock_gateway = MockStreamingGateway([[ChatChunk(text_delta="Sunny.", finish_reason="stop")]])
    worker = AgentWorker(gateway=mock_gateway, bridge=bridge, voice_pipeline=mock_voice)

    await worker._process_listen()

    assert mock_voice.listen_and_transcribe.called
    assert len(emitted_user_messages) == 1
    assert emitted_user_messages[0] == "What is the weather today?"


@pytest.mark.asyncio
async def test_agent_worker_memory_context_injection(qapp, tmp_path):
    """ARCH-03: AgentWorker retrieves memories via HybridRetriever and injects into history."""
    from unittest.mock import AsyncMock, MagicMock
    from shamu.memory.hybrid import HybridRetriever
    from shamu.types import MemoryRecord

    bridge = UIBridge()
    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.hybrid_retrieve = AsyncMock(return_value=[
        MemoryRecord(id="rec-1", content="User name is Alice", source="conversation", category="preference")
    ])

    mock_gateway = MockStreamingGateway([[ChatChunk(text_delta="Hello Alice!", finish_reason="stop")]])
    worker = AgentWorker(gateway=mock_gateway, bridge=bridge, retriever=mock_retriever)

    await worker._process_chat("Who am I?")

    assert mock_retriever.hybrid_retrieve.called
    assert any("User name is Alice" in (m.content or "") for m in worker.history)
    # M2.0: Retrieved memory is attached to Role.USER, not Role.SYSTEM
    assert not any(m.role == Role.SYSTEM for m in worker.history)
    assert worker.history[0].role == Role.USER
    assert "User name is Alice" in worker.history[0].content





