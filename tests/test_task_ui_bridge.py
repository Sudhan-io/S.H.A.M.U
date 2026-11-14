"""Tests for TaskUIBridge, PySide6 GUI signals, and CLI /plan workflow integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from rich.console import Console

from shamu.app import UIBridge
from shamu.cli import ShamuREPL
from shamu.gateway.router import ModelGateway, RoutingPolicy
from shamu.security.approval_manager import ApprovalResponse
from shamu.tasks.models import (
    PlanCompletedEvent,
    PlanStartedEvent,
    StepCompletedEvent,
    StepFailedEvent,
    StepStartedEvent,
    StepVerification,
    StepWaitingApprovalEvent,
    TaskExecutionReport,
    TaskPlan,
    TaskStatus,
    TaskStep,
    VerificationType,
)
from shamu.tasks.storage import TaskStorage
from shamu.tasks.ui_bridge import TaskUIBridge
from shamu.types import ChatChunk


def make_sample_plan() -> TaskPlan:
    """Creates a sample TaskPlan for testing."""
    return TaskPlan(
        id="plan-test-01",
        objective="Deploy landing page",
        steps=[
            TaskStep(
                id="step-1",
                title="Create index.html",
                description="Write HTML template",
                tool_name="write_file",
                arguments={"path": "index.html", "content": "<h1>Hello</h1>"},
                dependencies=[],
                verification=StepVerification(
                    type=VerificationType.FILE_EXISTS,
                    target="index.html",
                ),
            ),
            TaskStep(
                id="step-2",
                title="Start web server",
                description="Launch HTTP server",
                tool_name="run_command",
                arguments={"command": "python -m http.server 8000"},
                dependencies=["step-1"],
            ),
        ],
    )


def test_ui_bridge_signals_exist():
    """Verifies UIBridge defines task_plan_created, task_step_progress, and task_plan_completed signals."""
    bridge = UIBridge()
    assert hasattr(bridge, "task_plan_created")
    assert hasattr(bridge, "task_step_progress")
    assert hasattr(bridge, "task_plan_completed")

    created_events = []
    progress_events = []
    completed_events = []

    bridge.task_plan_created.connect(lambda *args: created_events.append(args))
    bridge.task_step_progress.connect(lambda *args: progress_events.append(args))
    bridge.task_plan_completed.connect(lambda *args: completed_events.append(args))

    bridge.task_plan_created.emit("plan-01", "Build webapp", 2)
    bridge.task_step_progress.emit("step-1", "in_progress", "")
    bridge.task_plan_completed.emit("plan-01", True, "All steps completed")

    assert len(created_events) == 1
    assert created_events[0] == ("plan-01", "Build webapp", 2)

    assert len(progress_events) == 1
    assert progress_events[0] == ("step-1", "in_progress", "")

    assert len(completed_events) == 1
    assert completed_events[0] == ("plan-01", True, "All steps completed")


def test_task_ui_bridge_render_plan():
    """Verifies TaskUIBridge renders a structured Rich table with all columns and metadata."""
    console = Console(record=True, width=120)
    gui_mock = MagicMock()
    bridge = TaskUIBridge(console=console, gui_bridge=gui_mock)

    plan = make_sample_plan()
    table = bridge.render_plan(plan)

    assert table is not None
    output = console.export_text()
    assert "Task Plan: Deploy landing page" in output
    assert "step-1" in output
    assert "Create index.html" in output
    assert "write_file" in output
    assert "file_exists: index.html" in output
    assert "step-2" in output
    assert "Start web server" in output

    # Check gui_mock notified
    assert gui_mock.task_plan_created.emit.called


def test_task_ui_bridge_on_event_streaming():
    """Verifies TaskUIBridge processes all task execution events and emits signals."""
    console = Console(record=True)
    gui_mock = MagicMock()
    bridge = TaskUIBridge(console=console, gui_bridge=gui_mock)

    plan_id = "plan-test-01"

    # 1. Plan started
    bridge.on_event(PlanStartedEvent(plan_id=plan_id, objective="Deploy landing page", total_steps=2))
    # 2. Step started
    bridge.on_event(StepStartedEvent(plan_id=plan_id, step_id="step-1", step_title="Create index.html", tool_name="write_file"))
    # 3. Step waiting approval
    bridge.on_event(StepWaitingApprovalEvent(plan_id=plan_id, step_id="step-1", step_title="Create index.html", tool_name="write_file", arguments={}))
    # 4. Step completed
    bridge.on_event(StepCompletedEvent(plan_id=plan_id, step_id="step-1", step_title="Create index.html", result="File written", verified=True))
    # 5. Step failed
    bridge.on_event(StepFailedEvent(plan_id=plan_id, step_id="step-2", step_title="Start web server", error="Port 8000 in use"))
    # 6. Plan completed
    bridge.on_event(PlanCompletedEvent(plan_id=plan_id, success=False, completed_steps=1, total_steps=2, summary="1/2 steps passed"))

    output = console.export_text()
    assert "Starting Execution of Task Plan" in output
    assert "Deploy landing page" in output
    assert "Step [step-1]" in output
    assert "Approval Required" in output
    assert "Completed" in output
    assert "Verified" in output
    assert "Failed" in output
    assert "Port 8000 in use" in output
    assert "Plan Completed" in output or "Plan Finished" in output or "1/2 steps passed" in output

    assert gui_mock.task_step_progress.emit.called
    assert gui_mock.task_plan_completed.emit.called


@pytest.mark.asyncio
async def test_task_ui_bridge_async_handle_event():
    """Verifies handle_event async interface works identically to on_event."""
    console = Console(record=True)
    bridge = TaskUIBridge(console=console)

    await bridge.handle_event(
        PlanStartedEvent(plan_id="plan-async", objective="Async task", total_steps=1)
    )
    output = console.export_text()
    assert "Async task" in output


def test_task_ui_bridge_event_deduplication():
    """Verifies that duplicate events are ignored to avoid spamming UI and signals."""
    console = Console(record=True)
    gui_mock = MagicMock()
    bridge = TaskUIBridge(console=console, gui_bridge=gui_mock)

    evt = StepStartedEvent(plan_id="p1", step_id="s1", step_title="Do work", tool_name="tool")
    bridge.on_event(evt)
    bridge.on_event(evt)  # duplicate

    assert gui_mock.task_step_progress.emit.call_count == 1


def test_task_ui_bridge_render_report():
    """Verifies TaskUIBridge renders a comprehensive final execution report."""
    console = Console(record=True)
    bridge = TaskUIBridge(console=console)

    report = TaskExecutionReport(
        plan_id="plan-report-test",
        objective="Build and verify artifact",
        status=TaskStatus.COMPLETED,
        total_steps=2,
        completed_steps=2,
        failed_steps=0,
        duration_seconds=1.45,
        step_summaries=[
            {
                "id": "s1",
                "title": "Build code",
                "tool": "compile",
                "status": "completed",
                "result": "Success",
                "error": None,
                "verified": True,
            },
            {
                "id": "s2",
                "title": "Run tests",
                "tool": "pytest",
                "status": "completed",
                "result": "10 passed",
                "error": None,
                "verified": True,
            },
        ],
        final_output="All steps passed successfully.",
    )

    panel = bridge.render_report(report)
    assert panel is not None

    output = console.export_text()
    assert "Task Execution Summary: COMPLETED" in output
    assert "Build and verify artifact" in output
    assert "1.45s" in output
    assert "Build code" in output
    assert "Run tests" in output
    assert "PASSED" in output


@pytest.mark.asyncio
async def test_cli_plan_command_missing_objective():
    """Verifies /plan without objective prints usage instructions."""
    console = Console(record=True)
    mock_gateway = MagicMock()
    repl = ShamuREPL(gateway=mock_gateway, console=console)

    should_exit = repl.handle_slash_command("/plan")
    if asyncio.iscoroutine(should_exit):
        should_exit = await should_exit

    assert not should_exit
    output = console.export_text()
    assert "Usage: /plan <objective>" in output


@pytest.mark.asyncio
async def test_cli_plan_command_user_cancelled():
    """Verifies /plan workflow marks plan CANCELLED and halts if user rejects approval."""
    console = Console(record=True)
    mock_gateway = MagicMock()

    planner_json = (
        '{"objective": "Create testfile", "steps": ['
        '{"id": "step-1", "title": "Create file", "tool_name": null, "dependencies": []}'
        ']}'
    )

    async def mock_stream_chat(history, tools=None):
        yield ChatChunk(text_delta=planner_json)

    mock_gateway.stream_chat = mock_stream_chat
    mock_gateway.ollama = MagicMock(model="test-model", base_url="http://localhost:11434")
    mock_gateway.openrouter = MagicMock(model="fallback-model")
    mock_gateway.policy = RoutingPolicy.AUTO_FALLBACK

    repl = ShamuREPL(gateway=mock_gateway, console=console)

    with patch("builtins.input", return_value="n"):
        res = repl.handle_slash_command("/plan Create testfile")
        if asyncio.iscoroutine(res):
            res = await res

    output = console.export_text()
    assert "Plan execution cancelled by user" in output
    assert repl.last_plan is not None
    assert repl.last_plan.status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_cli_plan_command_user_approved_execution(tmp_path):
    """Verifies full /plan workflow: decompose, display, approve [y], execute, stream, report."""
    console = Console(record=True)
    mock_gateway = MagicMock()

    target_file = tmp_path / "result.txt"
    target_str = str(target_file).replace("\\", "/")

    planner_json = (
        f'{{"objective": "Generate artifact", "steps": ['
        f'{{"id": "s-1", "title": "Write result", "tool_name": "write_file", '
        f'"arguments": {{"path": "{target_str}", "content": "SUCCESS_DATA"}}, "dependencies": [], '
        f'"verification": {{"type": "file_exists", "target": "{target_str}"}}}}'
        f']}}'
    )

    async def mock_stream_chat(history, tools=None):
        yield ChatChunk(text_delta=planner_json)

    mock_gateway.stream_chat = mock_stream_chat
    mock_gateway.ollama = MagicMock(model="test-model", base_url="http://localhost:11434")
    mock_gateway.openrouter = MagicMock(model="fallback-model")
    mock_gateway.policy = RoutingPolicy.AUTO_FALLBACK

    repl = ShamuREPL(gateway=mock_gateway, console=console)
    repl._auto_approve_plan = True
    if repl.firewall and repl.firewall.approval_manager:
        repl.firewall.approval_manager.register_handler(
            AsyncMock(return_value=ApprovalResponse(request_id="test", approved=True, reason="Test approved"))
        )

    # Mock firewall write_file execution to actually create the target file
    async def mock_execute_tool(tool_name, arguments, **kwargs):
        if tool_name == "write_file":
            target_file.write_text(arguments.get("content", ""), encoding="utf-8")
            return f"Wrote {target_file}"
        return "OK"

    repl.firewall.execute_tool = AsyncMock(side_effect=mock_execute_tool)

    res = repl.handle_slash_command(f"/plan Generate artifact at {target_str}")
    if asyncio.iscoroutine(res):
        res = await res

    output = console.export_text()
    assert "Planning objective:" in output
    assert "Task Plan:" in output
    assert "Starting Execution of Task Plan" in output
    assert "Completed Successfully" in output
    assert "Task Execution Summary: COMPLETED" in output
    assert repl.last_plan.status == TaskStatus.COMPLETED
    assert repl.last_execution_report is not None
    assert repl.last_execution_report.completed_steps == 1
    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == "SUCCESS_DATA"
