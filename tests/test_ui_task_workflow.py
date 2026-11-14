"""GUI Acceptance Tests for SHAMU PySide6 Desktop Application Workflow."""

import asyncio
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from shamu.ui.window import MainWindow, UIBridge, TaskPlanCard, TaskStepItem, TaskReportCard
from shamu.tasks.models import (
    TaskPlan,
    TaskStep,
    TaskStatus,
    StepVerification,
    VerificationType,
    TaskExecutionReport,
    PlanStartedEvent,
    StepStartedEvent,
    StepCompletedEvent,
    StepFailedEvent,
    PlanCompletedEvent,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_plan():
    """Generates a representative DAG TaskPlan for GUI verification."""
    step1 = TaskStep(
        id="step-1",
        title="Analyze target directory",
        description="Inspect files and list directory contents",
        tool_name="list_dir",
        arguments={"path": "."},
        status=TaskStatus.PENDING,
    )
    step2 = TaskStep(
        id="step-2",
        title="Generate summary report",
        description="Compile directory listing into markdown report",
        tool_name="write_file",
        arguments={"path": "report.md"},
        dependencies=["step-1"],
        status=TaskStatus.PENDING,
        verification=StepVerification(
            type=VerificationType.FILE_EXISTS,
            target="report.md",
        ),
    )
    return TaskPlan(
        id="plan-test-101",
        objective="Inspect directory and generate summary report",
        steps=[step1, step2],
        status=TaskStatus.PENDING,
        approved=False,
    )


def test_ui_bridge_task_signal_overloads(qapp):
    """Verify UIBridge task planning and execution signal overloads (multi-arg and single-arg)."""
    bridge = UIBridge()

    created_multi = []
    created_single = []
    progress_multi = []
    completed_multi = []
    approved_ids = []
    rejected_ids = []

    bridge.task_plan_created.connect(lambda pid, obj, n: created_multi.append((pid, obj, n)))
    bridge.task_plan_created[object].connect(lambda ev: created_single.append(ev))
    bridge.task_step_progress.connect(lambda sid, stat, det: progress_multi.append((sid, stat, det)))
    bridge.task_plan_completed.connect(lambda pid, succ, summ: completed_multi.append((pid, succ, summ)))
    bridge.task_plan_approved.connect(lambda pid: approved_ids.append(pid))
    bridge.task_plan_rejected.connect(lambda pid: rejected_ids.append(pid))

    # Test multi-arg emit
    bridge.task_plan_created.emit("plan-1", "Objective 1", 3)
    assert created_multi == [("plan-1", "Objective 1", 3)]

    # Test single-arg object emit
    dummy_obj = {"id": "plan-2"}
    bridge.task_plan_created[object].emit(dummy_obj)
    assert created_single == [dummy_obj]

    # Test progress & completion signals
    bridge.task_step_progress.emit("step-1", "completed", "ok")
    assert progress_multi == [("step-1", "completed", "ok")]

    bridge.task_plan_completed.emit("plan-1", True, "All done")
    assert completed_multi == [("plan-1", True, "All done")]

    bridge.task_plan_approved.emit("plan-1")
    assert approved_ids == ["plan-1"]

    bridge.task_plan_rejected.emit("plan-1")
    assert rejected_ids == ["plan-1"]


def test_task_plan_presentation_in_window(qapp, sample_plan):
    """Verify task plan presentation with steps, tools, dependencies, and styling in MainWindow."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)

    # Present plan via signal
    bridge.task_plan_created[object].emit(sample_plan)
    qapp.processEvents()

    assert window.current_task_card is not None
    card = window.current_task_card
    assert card.plan_id == sample_plan.id
    assert card.objective == sample_plan.objective
    assert card.total_steps == 2
    assert card.status_badge.text() == "PENDING"

    # Verify steps rendered inside card
    assert "step-1" in card.step_items
    assert "step-2" in card.step_items

    item1 = card.step_items["step-1"]
    assert "Analyze target directory" in item1.title_label.text()
    assert "[list_dir]" in item1.tool_label.text()
    assert "Pending" in item1.status_pill.text()

    item2 = card.step_items["step-2"]
    assert "Generate summary report" in item2.title_label.text()
    assert "[write_file]" in item2.tool_label.text()


def test_task_plan_gui_approval_flow(qapp, sample_plan):
    """Verify approval flow: buttons appear, approval signal emits, future resolves to True, badge updates."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)

    approved_emitted = []
    bridge.task_plan_approved.connect(lambda pid: approved_emitted.append(pid))

    loop = asyncio.new_event_loop()
    future = loop.create_future()

    # Create plan and request approval
    bridge.task_plan_created[object].emit(sample_plan)
    bridge.task_plan_approval_requested.emit(sample_plan, future)
    qapp.processEvents()

    card = window.current_task_card
    assert card is not None
    assert not card.approval_widget.isHidden()
    assert card.approve_btn.isEnabled()
    assert card.reject_btn.isEnabled()
    assert card.status_badge.text() == "WAITING APPROVAL"

    # Simulate user clicking "Approve Plan"
    card.approve_btn.click()
    qapp.processEvents()

    assert card.status_badge.text() == "APPROVED"
    assert not card.approve_btn.isEnabled()
    assert not card.reject_btn.isEnabled()
    assert approved_emitted == [sample_plan.id]
    assert future.done()
    assert future.result() is True

    loop.close()


def test_task_plan_gui_rejection_flow(qapp, sample_plan):
    """Verify rejection flow: user rejects plan, future resolves to False, card marked CANCELLED."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)

    rejected_emitted = []
    bridge.task_plan_rejected.connect(lambda pid: rejected_emitted.append(pid))

    loop = asyncio.new_event_loop()
    future = loop.create_future()

    # Create plan and request approval
    bridge.task_plan_created[object].emit(sample_plan)
    bridge.task_plan_approval_requested.emit(sample_plan, future)
    qapp.processEvents()

    card = window.current_task_card
    # User clicks "Reject"
    card.reject_btn.click()
    qapp.processEvents()

    assert card.status_badge.text() == "CANCELLED"
    assert not card.approve_btn.isEnabled()
    assert not card.reject_btn.isEnabled()
    assert rejected_emitted == [sample_plan.id]
    assert future.done()
    assert future.result() is False

    loop.close()


def test_task_step_progress_updates(qapp, sample_plan):
    """Verify real-time step progress updates, status pill, and detail preview in GUI."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)

    bridge.task_plan_created[object].emit(sample_plan)
    qapp.processEvents()
    card = window.current_task_card

    # Step 1 starts
    bridge.task_step_progress.emit("step-1", "in_progress", "")
    qapp.processEvents()
    assert "Running" in card.step_items["step-1"].status_pill.text()
    assert "Step step-1: in_progress" in window.status_pill.text()

    # Step 1 completes with result
    bridge.task_step_progress.emit("step-1", "completed", "Found 5 text files in target folder")
    qapp.processEvents()
    assert "Done" in card.step_items["step-1"].status_pill.text()
    assert not card.step_items["step-1"].detail_label.isHidden()
    assert "Found 5 text files" in card.step_items["step-1"].detail_label.text()

    # Step 2 waiting approval
    bridge.task_step_progress.emit("step-2", "waiting_approval", "write_file")
    qapp.processEvents()
    assert "Approval" in card.step_items["step-2"].status_pill.text()


def test_task_step_failure_handling(qapp, sample_plan):
    """Verify failure handling: failed step highlights red with error details, plan marked FAILED."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)

    bridge.task_plan_created[object].emit(sample_plan)
    qapp.processEvents()
    card = window.current_task_card

    # Step 1 fails with error
    bridge.task_step_progress.emit("step-1", "failed", "PermissionDenied: Access denied to C:\\Windows\\System32")
    qapp.processEvents()

    item1 = card.step_items["step-1"]
    assert "Failed" in item1.status_pill.text()
    assert not item1.detail_label.isHidden()
    assert "PermissionDenied" in item1.detail_label.text()

    # Plan completes with failure
    bridge.task_plan_completed.emit(sample_plan.id, False, "Execution aborted due to step-1 failure")
    qapp.processEvents()

    assert card.status_badge.text() == "FAILED"
    assert not card.summary_label.isHidden()
    assert "Execution aborted" in card.summary_label.text()
    assert "Plan failed" in window.status_pill.text()


def test_task_report_display(qapp):
    """Verify TaskReportCard displays duration, completed/failed metrics, and step summaries."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)

    report = TaskExecutionReport(
        plan_id="plan-report-99",
        objective="Automated workspace audit and linting",
        status=TaskStatus.COMPLETED,
        completed_steps=2,
        total_steps=2,
        failed_steps=0,
        duration_seconds=1.45,
        final_output="Audit clean: 2 steps completed with 100% verification.",
        step_summaries=[
            {
                "id": "step-1",
                "title": "Run security check",
                "tool": "check_security",
                "status": "completed",
                "verified": True,
                "result": "Passed all 8 checks",
            },
            {
                "id": "step-2",
                "title": "Write report",
                "tool": "write_file",
                "status": "completed",
                "verified": True,
                "result": "Wrote report.md",
            },
        ],
    )

    report_card = window.display_task_report(report)
    qapp.processEvents()

    assert report_card is not None
    assert report_card.plan_id == "plan-report-99"
    assert report_card.is_success is True
    assert report_card.completed_steps == 2
    assert report_card.total_steps == 2
    assert report_card.duration == 1.45
    assert "Audit clean" in report_card.final_output
    assert len(report_card.step_summaries) == 2


def test_ui_responsiveness_during_rapid_task_events(qapp, sample_plan):
    """Verify UI thread responsiveness: processing events without freezing the Qt main loop."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)
    window.show()
    qapp.processEvents()

    bridge.task_plan_created[object].emit(sample_plan)
    qapp.processEvents()

    # Emit a rapid sequence of events simulating fast background executor yields
    for i in range(1, 10):
        bridge.task_step_progress.emit(f"step-{i}", "in_progress", f"Running pass {i}")
        bridge.task_step_progress.emit(f"step-{i}", "completed", f"Result {i}")
        # Test input responsiveness
        window.input_field.setText(f"Typing while executing {i}")
        qapp.processEvents()
        assert window.input_field.text() == f"Typing while executing {i}"

    bridge.task_plan_completed.emit(sample_plan.id, True, "All fast steps completed successfully")
    qapp.processEvents()

    assert "Plan completed" in window.status_pill.text()
    window.hide()


@pytest.mark.asyncio
async def test_agent_worker_plan_slash_command_usage(qapp):
    """Verify AgentWorker outputs usage information when /plan is missing objective."""
    from unittest.mock import MagicMock
    from shamu.app import AgentWorker

    bridge = UIBridge()
    gateway = MagicMock()
    worker = AgentWorker(gateway=gateway, bridge=bridge)

    chunks = []
    statuses = []
    bridge.chunk_received.connect(lambda c: chunks.append(c))
    bridge.status_updated.connect(lambda s: statuses.append(s))

    await worker._process_chat("/plan   ")

    assert any("Usage: /plan <objective>" in c for c in chunks)
    assert "Ready" in statuses


@pytest.mark.asyncio
async def test_agent_worker_plan_auto_approve_and_execute(qapp, monkeypatch):
    """Verify AgentWorker /plan execution with _auto_approve_plan=True runs end-to-end."""
    from unittest.mock import AsyncMock, MagicMock
    from shamu.app import AgentWorker
    from shamu.tasks.planner import TaskPlanner
    from shamu.tasks.executor import TaskExecutor

    bridge = UIBridge()
    window = MainWindow(bridge=bridge)
    gateway = MagicMock()
    firewall = MagicMock()
    firewall.approval_manager = MagicMock()

    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)
    worker._auto_approve_plan = True

    # Setup mock plan
    step1 = TaskStep(
        id="s1",
        title="Check status",
        status=TaskStatus.PENDING,
    )
    test_plan = TaskPlan(
        id="plan-auto-1",
        objective="Automated health check",
        steps=[step1],
        status=TaskStatus.PENDING,
    )

    async def mock_create_plan(self_planner, obj):
        return test_plan

    monkeypatch.setattr(TaskPlanner, "create_plan", mock_create_plan)

    report_received = []
    bridge.task_plan_report.connect(lambda r: report_received.append(r))

    await worker._process_chat("/plan Automated health check")
    qapp.processEvents()

    assert window.current_task_card is not None
    assert window.current_task_card.plan_id == "plan-auto-1"
    assert len(report_received) == 1
    assert report_received[0].plan_id == "plan-auto-1"


@pytest.mark.asyncio
async def test_agent_worker_plan_rejection(qapp, monkeypatch):
    """Verify AgentWorker /plan interactive rejection sets status to CANCELLED."""
    from unittest.mock import MagicMock
    from shamu.app import AgentWorker
    from shamu.tasks.planner import TaskPlanner

    bridge = UIBridge()
    window = MainWindow(bridge=bridge)
    gateway = MagicMock()
    firewall = MagicMock()

    worker = AgentWorker(gateway=gateway, bridge=bridge, firewall=firewall)
    worker._auto_approve_plan = False

    step1 = TaskStep(id="s1", title="Risky step", status=TaskStatus.PENDING)
    test_plan = TaskPlan(
        id="plan-reject-1",
        objective="Risky deployment",
        steps=[step1],
        status=TaskStatus.PENDING,
    )

    async def mock_create_plan(self_planner, obj):
        return test_plan

    monkeypatch.setattr(TaskPlanner, "create_plan", mock_create_plan)

    def handle_approval_request(plan, future):
        # User rejects via GUI
        qapp.processEvents()
        window.reject_current_task_plan()

    bridge.task_plan_approval_requested.connect(handle_approval_request)

    await worker._process_chat("/plan Risky deployment")
    qapp.processEvents()

    assert test_plan.status == TaskStatus.CANCELLED
    assert window.current_task_card.status_badge.text() == "CANCELLED"

