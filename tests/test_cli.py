import pytest
from shamu.cli import create_parser, parse_cli_policy
from shamu.gateway.router import RoutingPolicy


def test_cli_parser_defaults():
    parser = create_parser()
    args = parser.parse_args([])
    assert args.model is None
    assert args.offline is False
    assert args.cloud is False
    assert args.voice is False
    assert args.prompt is None


def test_cli_parser_custom_args():
    parser = create_parser()
    args = parser.parse_args(["--model", "llama3.2:3b", "--offline", "--voice", "--prompt", "Hello"])
    assert args.model == "llama3.2:3b"
    assert args.offline is True
    assert args.voice is True
    assert args.prompt == "Hello"


def test_parse_cli_policy():
    parser = create_parser()
    assert parse_cli_policy(parser.parse_args(["--offline"])) == RoutingPolicy.LOCAL_ONLY
    assert parse_cli_policy(parser.parse_args(["--cloud"])) == RoutingPolicy.CLOUD_ONLY
    assert parse_cli_policy(parser.parse_args([])) == RoutingPolicy.AUTO_FALLBACK


@pytest.mark.asyncio
async def test_cli_repl_executes_tool_via_firewall(monkeypatch):
    """CLI-01: Ensure ShamuREPL passes tools, executes via ExecutionFirewall, and updates history."""
    from rich.console import Console
    from shamu.cli import ShamuREPL
    from shamu.types import ChatChunk, ToolCall, Role
    from shamu.security.firewall import ExecutionFirewall
    from unittest.mock import AsyncMock, MagicMock

    mock_gateway = MagicMock()
    mock_firewall = MagicMock(spec=ExecutionFirewall)
    mock_firewall.approval_manager = MagicMock()
    mock_firewall.tool_registry = MagicMock()
    mock_firewall.tool_registry.get_tier.return_value = 0  # Tier 0 safe
    mock_firewall.get_tool_definitions.return_value = [{"name": "read_file"}]
    mock_firewall.execute_tool = AsyncMock(return_value="mock file content")

    # Turn 1: Assistant calls tool
    turn1_chunks = [
        ChatChunk(
            tool_call_delta=ToolCall(id="call_99", name="read_file", arguments='{"path": "data.txt"}')
        )
    ]
    # Turn 2: Assistant gives final answer
    turn2_chunks = [
        ChatChunk(text_delta="The file contains: mock file content", finish_reason="stop")
    ]

    call_count = 0

    async def mock_stream_chat(history, tools=None):
        nonlocal call_count
        call_count += 1
        assert tools is not None
        if call_count == 1:
            for ch in turn1_chunks:
                yield ch
        else:
            for ch in turn2_chunks:
                yield ch

    mock_gateway.stream_chat = mock_stream_chat
    console = Console(record=True)

    repl = ShamuREPL(gateway=mock_gateway, console=console, firewall=mock_firewall)
    await repl.execute_turn("Read data.txt")

    # Verify tool was executed via firewall
    mock_firewall.execute_tool.assert_called_once_with(tool_name="read_file", arguments={"path": "data.txt"})

    # Verify conversation history has: User, Assistant(tool_call), Tool(result), Assistant(final)
    assert len(repl.history) == 4
    assert repl.history[0].role == Role.USER
    assert repl.history[0].content == "Read data.txt"

    assert repl.history[1].role == Role.ASSISTANT
    assert repl.history[1].tool_calls is not None
    assert repl.history[1].tool_calls[0].name == "read_file"

    assert repl.history[2].role == Role.TOOL
    assert repl.history[2].content == "mock file content"
    assert repl.history[2].tool_call_id == "call_99"

    assert repl.history[3].role == Role.ASSISTANT
    assert repl.history[3].content == "The file contains: mock file content"


@pytest.mark.asyncio
async def test_cli_repl_async_input_reading(monkeypatch):
    """Ensure ShamuREPL.run() uses loop.run_in_executor with sys.stdin.readline."""
    import asyncio
    import sys
    from rich.console import Console
    from shamu.cli import ShamuREPL
    from unittest.mock import MagicMock

    mock_gateway = MagicMock()
    mock_firewall = MagicMock()
    mock_firewall.approval_manager = MagicMock()
    console = Console(record=True)
    repl = ShamuREPL(gateway=mock_gateway, console=console, firewall=mock_firewall)

    loop = asyncio.get_running_loop()
    readline_calls = 0

    async def mock_run_in_executor(executor, func, *args):
        nonlocal readline_calls
        assert executor is None
        assert func == sys.stdin.readline
        readline_calls += 1
        return "/exit\n"

    monkeypatch.setattr(loop, "run_in_executor", mock_run_in_executor)

    await repl.run()
    assert readline_calls == 1


@pytest.mark.asyncio
async def test_cli_approval_handler_non_interactive_tty(monkeypatch):
    """Ensure cli_approval_handler rejects immediately when sys.stdin is not a TTY."""
    import sys
    from rich.console import Console
    from shamu.cli import ShamuREPL
    from shamu.security.approval_manager import ApprovalRequest, ApprovalManager
    from shamu.security.tool_registry import PermissionTier
    from unittest.mock import MagicMock

    mock_gateway = MagicMock()
    appr_mgr = ApprovalManager()
    mock_firewall = MagicMock()
    mock_firewall.approval_manager = appr_mgr

    console = Console(record=True)
    _ = ShamuREPL(gateway=mock_gateway, console=console, firewall=mock_firewall)

    # Force non-interactive stdin
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    req = ApprovalRequest(
        request_id="req-999",
        tool_name="delete_database",
        arguments={"path": "/data"},
        canonical_argument_hash="",
        tier=PermissionTier.TIER_3_HIGH_IMPACT,
        reason="Dangerous file deletion",
    )

    assert appr_mgr.handler is not None
    resp = await appr_mgr.handler(req)

    assert resp.approved is False
    assert resp.reason == "Non-interactive terminal: approval unavailable"
    assert resp.request_id == "req-999"


@pytest.mark.asyncio
async def test_cli_approval_handler_timeout(monkeypatch):
    """Ensure cli_approval_handler rejects when user input times out."""
    import asyncio
    import sys
    from rich.console import Console
    from shamu.cli import ShamuREPL
    from shamu.security.approval_manager import ApprovalRequest, ApprovalManager
    from shamu.security.tool_registry import PermissionTier
    from unittest.mock import MagicMock

    mock_gateway = MagicMock()
    appr_mgr = ApprovalManager()
    mock_firewall = MagicMock()
    mock_firewall.approval_manager = appr_mgr

    console = Console(record=True)
    _ = ShamuREPL(gateway=mock_gateway, console=console, firewall=mock_firewall)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    async def hanging_wait_for(fut, timeout):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", hanging_wait_for)

    req = ApprovalRequest(
        request_id="req-timeout",
        tool_name="shell_command",
        arguments={"command": "rm -rf /"},
        canonical_argument_hash="",
        tier=PermissionTier.TIER_3_HIGH_IMPACT,
        reason="Shell execution",
    )

    resp = await appr_mgr.handler(req)
    assert resp.approved is False
    assert "Approval timed out" in resp.reason


@pytest.mark.asyncio
async def test_single_prompt_tool_execution_loop(monkeypatch):
    """Ensure single prompt mode runs multi-turn tool execution loop via firewall."""
    from unittest.mock import MagicMock, AsyncMock
    import shamu.cli
    from shamu.types import ChatChunk, ToolCall

    mock_gateway = MagicMock()
    mock_firewall = MagicMock()
    mock_firewall.get_tool_definitions.return_value = [{"name": "read_file"}]
    mock_firewall.execute_tool = AsyncMock(return_value="secret prompt content")

    call_count = 0
    async def mock_stream_chat(history, tools=None):
        nonlocal call_count
        call_count += 1
        assert tools == [{"name": "read_file"}]
        if call_count == 1:
            yield ChatChunk(tool_call_delta=ToolCall(id="c1", name="read_file", arguments='{"path": "file.txt"}'))
        else:
            yield ChatChunk(text_delta="Read output: secret prompt content", finish_reason="stop")

    mock_gateway.stream_chat = mock_stream_chat

    monkeypatch.setattr(shamu.cli, "ModelGateway", lambda *args, **kwargs: mock_gateway)
    monkeypatch.setattr(shamu.cli, "ExecutionFirewall", lambda *args, **kwargs: mock_firewall)

    parser = shamu.cli.create_parser()
    args = parser.parse_args(["--prompt", "Read file.txt"])

    await shamu.cli.async_main(args)

    mock_firewall.execute_tool.assert_called_once_with(
        tool_name="read_file",
        arguments={"path": "file.txt"},
    )


@pytest.mark.asyncio
async def test_cli_repl_with_compactor_wired():
    """Ensure ShamuREPL invokes compactor during conversation turns."""
    from rich.console import Console
    from shamu.cli import ShamuREPL
    from shamu.types import ChatChunk
    from unittest.mock import MagicMock, AsyncMock

    mock_gateway = MagicMock()
    mock_firewall = MagicMock()
    mock_firewall.approval_manager = MagicMock()
    mock_firewall.get_tool_definitions.return_value = []

    async def mock_stream_chat(history, tools=None):
        yield ChatChunk(text_delta="Response from model")

    mock_gateway.stream_chat = mock_stream_chat
    console = Console(record=True)

    mock_compactor = MagicMock()
    mock_compactor.compact = AsyncMock(side_effect=lambda hist: (hist, 0))

    repl = ShamuREPL(
        gateway=mock_gateway,
        console=console,
        firewall=mock_firewall,
        compactor=mock_compactor,
    )

    await repl.execute_turn("Test input message")

    mock_compactor.compact.assert_called_once()
    compacted_arg = mock_compactor.compact.call_args[0][0]
    assert len(compacted_arg) == 1
    assert compacted_arg[0].content == "Test input message"


@pytest.mark.asyncio
async def test_async_main_wires_compactor(monkeypatch):
    """Ensure async_main initializes SQLiteMemoryStore, ContextCompactor and wires to ShamuREPL."""
    import shamu.cli
    from unittest.mock import MagicMock, AsyncMock

    mock_repl = MagicMock()
    mock_repl.run = AsyncMock()
    mock_repl_cls = MagicMock(return_value=mock_repl)

    monkeypatch.setattr(shamu.cli, "ShamuREPL", mock_repl_cls)
    monkeypatch.setattr(shamu.cli, "ModelGateway", MagicMock())

    parser = shamu.cli.create_parser()
    args = parser.parse_args([])

    await shamu.cli.async_main(args)

    assert mock_repl_cls.called
    kwargs = mock_repl_cls.call_args[1]
    assert "compactor" in kwargs
    assert kwargs["compactor"] is not None


def test_cli_repl_welcome_banner_includes_plan_command():
    """Ensure welcome banner lists /plan <objective> command."""
    from rich.console import Console
    from shamu.cli import ShamuREPL
    from unittest.mock import MagicMock

    console = Console(record=True)
    gateway = MagicMock()
    gateway.ollama.model = "test-ollama"
    gateway.openrouter.model = "test-openrouter"
    gateway.policy.value = "auto_fallback"

    repl = ShamuREPL(gateway=gateway, console=console)
    repl.print_welcome()

    output = console.export_text()
    assert "/plan <objective>" in output


@pytest.mark.asyncio
async def test_cli_async_main_prompt_with_plan_command(monkeypatch, tmp_path):
    """Ensure async_main with --prompt '/plan ...' executes plan workflow directly."""
    import shamu.cli
    from unittest.mock import MagicMock, AsyncMock

    mock_repl = MagicMock()
    mock_repl.execute_plan_command = AsyncMock()
    mock_repl_cls = MagicMock(return_value=mock_repl)

    monkeypatch.setattr(shamu.cli, "ShamuREPL", mock_repl_cls)
    monkeypatch.setattr(shamu.cli, "ModelGateway", MagicMock())
    monkeypatch.setattr(shamu.cli, "ExecutionFirewall", MagicMock())

    parser = shamu.cli.create_parser()
    args = parser.parse_args(["--prompt", "/plan Build test pipeline"])

    await shamu.cli.async_main(args)

    mock_repl.execute_plan_command.assert_called_once_with("/plan Build test pipeline")


@pytest.mark.asyncio
async def test_cli_repl_plan_approval_handler_auto_approve():
    """Ensure cli_approval_handler auto-approves when _auto_approve is set."""
    from rich.console import Console
    from shamu.cli import ShamuREPL
    from shamu.security.approval_manager import ApprovalManager, ApprovalRequest
    from shamu.security.tool_registry import PermissionTier
    from unittest.mock import MagicMock

    gateway = MagicMock()
    appr_mgr = ApprovalManager()
    firewall = MagicMock()
    firewall.approval_manager = appr_mgr
    console = Console(record=True)

    repl = ShamuREPL(gateway=gateway, console=console, firewall=firewall)
    repl._auto_approve = True

    req = ApprovalRequest(
        request_id="req-auto",
        tool_name="write_file",
        arguments={"path": "test.txt"},
        canonical_argument_hash="",
        tier=PermissionTier.TIER_2_MUTATION,
        reason="Writing file in automated mode",
    )

    resp = await appr_mgr.handler(req)
    assert resp.approved is True
    assert "Auto-approved" in resp.reason


