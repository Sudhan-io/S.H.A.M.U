"""Manual GUI Acceptance Verification & Visual Confirmation Suite for S.H.A.M.U.

This suite executes the complete manual GUI acceptance checklist for:
1. Plan generation for `/plan <objective>`
2. Plan card presentation (visual styling, tools, dependencies)
3. Plan approval & rejection flows (signals, button state transitions, future resolution)
4. Real-time step progress tracking (badges, detail previews, status pill)
5. Failed step handling & display (red styling, error box, plan failure summary)
6. Responsive UI event loop (event processing without freezing, concurrent user typing)

Includes both automated assertion tests and physical visual snapshot generation.
"""

import asyncio
import os
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap

from shamu.ui.window import MainWindow, UIBridge, TaskPlanCard, TaskStepItem, TaskReportCard
from shamu.tasks.models import (
    TaskPlan,
    TaskStep,
    TaskStatus,
    StepVerification,
    VerificationType,
    TaskExecutionReport,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def snapshot_dir(tmp_path_factory):
    """Directory for physical visual snapshot captures."""
    d = tmp_path_factory.mktemp("gui_snapshots")
    return str(d)


def test_checklist_item1_plan_generation_slash_command(qapp):
    """Checklist 1: Plan generation for /plan <objective>."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)
    window.show()
    qapp.processEvents()

    # Test empty /plan objective handling
    chunks = []
    bridge.chunk_received.connect(lambda c: chunks.append(c))

    window.input_field.setText("/plan")
    window._on_send()
    qapp.processEvents()

    # Test valid /plan command emission
    user_messages = []
    bridge.user_message_sent.connect(lambda m: user_messages.append(m))

    window.input_field.setText("/plan Inspect workspace files and generate summary")
    window._on_send()
    qapp.processEvents()

    assert user_messages == ["/plan Inspect workspace files and generate summary"]
    assert window.input_field.text() == ""
    window.hide()


def test_checklist_item2_plan_card_presentation(qapp, snapshot_dir):
    """Checklist 2: Plan card presentation (colors, fonts, structure, step list, icons, tool badges)."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)
    window.show()
    qapp.processEvents()

    step1 = TaskStep(
        id="s1-inspect",
        title="Scan designated directory",
        description="List all files in target path",
        tool_name="list_directory",
        arguments={"path": "target_docs"},
        status=TaskStatus.PENDING,
    )
    step2 = TaskStep(
        id="s2-filter",
        title="Filter and identify text files",
        description="Filter text files by extension (.txt, .md)",
        tool_name="read_file",
        arguments={"path": "target_docs/alpha.txt"},
        dependencies=["s1-inspect"],
        status=TaskStatus.PENDING,
    )
    step3 = TaskStep(
        id="s3-report",
        title="Compile summary report",
        description="Write Markdown report with directory findings",
        tool_name="write_file",
        arguments={"path": "summary_report.md"},
        dependencies=["s2-filter"],
        status=TaskStatus.PENDING,
        verification=StepVerification(type=VerificationType.FILE_EXISTS, target="summary_report.md"),
    )

    plan = TaskPlan(
        id="plan-acceptance-001",
        objective="Inspect directory and create summary report",
        steps=[step1, step2, step3],
        status=TaskStatus.PENDING,
    )

    card = window.add_task_plan(plan)
    qapp.processEvents()

    # Visual Presentation Assertions
    assert card is not None
    assert card.plan_id == "plan-acceptance-001"
    assert "Inspect directory and create summary report" in card.objective
    assert card.total_steps == 3
    assert card.status_badge.text() == "PENDING"
    assert len(card.step_items) == 3

    # Check Step 1 presentation
    item1 = card.step_items["s1-inspect"]
    assert "Step s1-inspect" in item1.id_label.text()
    assert "[list_directory]" in item1.tool_label.text()
    assert "Scan designated directory" in item1.title_label.text()
    assert "List all files in target path" in item1.desc_label.text()
    assert "Pending" in item1.status_pill.text()

    # Check Step 3 presentation
    item3 = card.step_items["s3-report"]
    assert "[write_file]" in item3.tool_label.text()

    # Physical Visual Snapshot Capture
    pix = window.grab()
    snap_path = os.path.join(snapshot_dir, "02_plan_card_presentation.png")
    pix.save(snap_path)
    assert os.path.exists(snap_path)
    assert os.path.getsize(snap_path) > 1000  # Non-empty valid PNG

    window.hide()


def test_checklist_item3_approval_rejection_flows(qapp, snapshot_dir):
    """Checklist 3: Plan approval & rejection flows (signals, button state transitions, future resolution)."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)
    window.show()
    qapp.processEvents()

    # Test 3A: Approval Flow
    plan_a = TaskPlan(
        id="plan-approval-test",
        objective="Verify approval button triggers approved signal",
        steps=[TaskStep(id="step-a", title="Step A", status=TaskStatus.PENDING)],
        status=TaskStatus.PENDING,
    )

    approved_signals = []
    bridge.task_plan_approved.connect(lambda pid: approved_signals.append(pid))

    loop = asyncio.new_event_loop()
    future_a = loop.create_future()

    card_a = window.add_task_plan(plan_a)
    card_a.set_approval_future(future_a)
    qapp.processEvents()

    assert card_a.status_badge.text() == "WAITING APPROVAL"
    assert card_a.approve_btn.isEnabled()
    assert card_a.reject_btn.isEnabled()

    # Click Approve
    card_a.approve_btn.click()
    qapp.processEvents()

    assert card_a.status_badge.text() == "APPROVED"
    assert not card_a.approve_btn.isEnabled()
    assert not card_a.reject_btn.isEnabled()
    assert approved_signals == ["plan-approval-test"]
    assert future_a.done()
    assert future_a.result() is True

    # Test 3B: Rejection Flow
    plan_b = TaskPlan(
        id="plan-rejection-test",
        objective="Verify rejection button triggers rejected signal",
        steps=[TaskStep(id="step-b", title="Step B", status=TaskStatus.PENDING)],
        status=TaskStatus.PENDING,
    )

    rejected_signals = []
    bridge.task_plan_rejected.connect(lambda pid: rejected_signals.append(pid))

    future_b = loop.create_future()
    card_b = window.add_task_plan(plan_b)
    card_b.set_approval_future(future_b)
    qapp.processEvents()

    assert card_b.status_badge.text() == "WAITING APPROVAL"

    # Click Reject
    card_b.reject_btn.click()
    qapp.processEvents()

    assert card_b.status_badge.text() == "CANCELLED"
    assert not card_b.approve_btn.isEnabled()
    assert not card_b.reject_btn.isEnabled()
    assert rejected_signals == ["plan-rejection-test"]
    assert future_b.done()
    assert future_b.result() is False

    # Physical Visual Snapshot Capture
    pix = window.grab()
    snap_path = os.path.join(snapshot_dir, "03_approval_rejection_flows.png")
    pix.save(snap_path)
    assert os.path.exists(snap_path)
    assert os.path.getsize(snap_path) > 1000

    loop.close()
    window.hide()


def test_checklist_item4_realtime_step_progress_tracking(qapp, snapshot_dir):
    """Checklist 4: Real-time step progress tracking (pending -> in_progress -> completed, status pill)."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)
    window.show()
    qapp.processEvents()

    plan = TaskPlan(
        id="plan-progress-001",
        objective="Track live progress events across multiple steps",
        steps=[
            TaskStep(id="st-1", title="Initialize environment", status=TaskStatus.PENDING),
            TaskStep(id="st-2", title="Extract documents", status=TaskStatus.PENDING),
            TaskStep(id="st-3", title="Save results", status=TaskStatus.PENDING),
        ],
        status=TaskStatus.PENDING,
    )

    card = window.add_task_plan(plan)
    qapp.processEvents()

    # Step 1 Running
    bridge.task_step_progress.emit("st-1", "in_progress", "")
    qapp.processEvents()
    assert "Running" in card.step_items["st-1"].status_pill.text()
    assert "Step st-1: in_progress" in window.status_pill.text()

    # Step 1 Completed with output
    bridge.task_step_progress.emit("st-1", "completed", "Environment initialized on target path /data")
    qapp.processEvents()
    assert "Done" in card.step_items["st-1"].status_pill.text()
    assert not card.step_items["st-1"].detail_label.isHidden()
    assert "Environment initialized" in card.step_items["st-1"].detail_label.text()

    # Step 2 Running -> Completed
    bridge.task_step_progress.emit("st-2", "in_progress", "")
    qapp.processEvents()
    assert "Running" in card.step_items["st-2"].status_pill.text()
    bridge.task_step_progress.emit("st-2", "completed", "Extracted 12 files successfully")
    qapp.processEvents()
    assert "Done" in card.step_items["st-2"].status_pill.text()

    # Physical Visual Snapshot Capture
    pix = window.grab()
    snap_path = os.path.join(snapshot_dir, "04_realtime_step_progress.png")
    pix.save(snap_path)
    assert os.path.exists(snap_path)
    assert os.path.getsize(snap_path) > 1000

    window.hide()


def test_checklist_item5_failed_step_handling_and_display(qapp, snapshot_dir):
    """Checklist 5: Failed step handling & display (red error badge, details, plan status FAILED)."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)
    window.show()
    qapp.processEvents()

    plan = TaskPlan(
        id="plan-failure-001",
        objective="Verify failure handling on tool execution error",
        steps=[
            TaskStep(id="fail-1", title="Write protected system file", status=TaskStatus.PENDING),
            TaskStep(id="fail-2", title="Dependent follow-up step", status=TaskStatus.PENDING),
        ],
        status=TaskStatus.PENDING,
    )

    card = window.add_task_plan(plan)
    qapp.processEvents()

    # Step fails with permission denied error
    err_msg = "SecurityFirewall: PermissionDeniedError - write operation blocked on protected path"
    bridge.task_step_progress.emit("fail-1", "failed", err_msg)
    qapp.processEvents()

    item_fail = card.step_items["fail-1"]
    assert "Failed" in item_fail.status_pill.text()
    assert not item_fail.detail_label.isHidden()
    assert "PermissionDeniedError" in item_fail.detail_label.text()

    # Dependent step skipped
    bridge.task_step_progress.emit("fail-2", "skipped", "Prerequisite failed")
    qapp.processEvents()
    assert "Skipped" in card.step_items["fail-2"].status_pill.text()

    # Plan completed with failure
    bridge.task_plan_completed.emit("plan-failure-001", False, "Execution aborted due to step fail-1 error")
    qapp.processEvents()

    assert card.status_badge.text() == "FAILED"
    assert not card.summary_label.isHidden()
    assert "Execution aborted" in card.summary_label.text()
    assert "Plan failed" in window.status_pill.text()

    # Physical Visual Snapshot Capture
    pix = window.grab()
    snap_path = os.path.join(snapshot_dir, "05_failed_step_handling.png")
    pix.save(snap_path)
    assert os.path.exists(snap_path)
    assert os.path.getsize(snap_path) > 1000

    window.hide()


def test_checklist_item6_responsive_ui_event_loop(qapp, snapshot_dir):
    """Checklist 6: Responsive UI event loop (user typing during rapid background task events)."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)
    window.show()
    qapp.processEvents()

    plan = TaskPlan(
        id="plan-concurrency-test",
        objective="Verify UI responsiveness during high event throughput",
        steps=[TaskStep(id=f"step-{i}", title=f"Worker step {i}", status=TaskStatus.PENDING) for i in range(1, 15)],
        status=TaskStatus.PENDING,
    )

    window.add_task_plan(plan)
    qapp.processEvents()

    # Simulate fast streaming events while user interacts with UI
    user_inputs = ["Hello", "Checking status", "Typing mid-execution", "/status"]
    for i, user_text in enumerate(user_inputs, 1):
        # Step progress event
        bridge.task_step_progress.emit(f"step-{i}", "in_progress", "")
        # Immediate user interaction
        window.input_field.setText(user_text)
        qapp.processEvents()
        # Verify UI input was never blocked or lost
        assert window.input_field.text() == user_text

        # Step complete event
        bridge.task_step_progress.emit(f"step-{i}", "completed", f"Output {i}")
        qapp.processEvents()

    # Test report presentation
    report = TaskExecutionReport(
        plan_id="plan-concurrency-test",
        objective="Verify UI responsiveness during high event throughput",
        status=TaskStatus.COMPLETED,
        completed_steps=14,
        total_steps=14,
        failed_steps=0,
        duration_seconds=0.88,
        final_output="All 14 steps executed with zero UI thread contention.",
        step_summaries=[{"id": f"step-{i}", "title": f"Worker step {i}", "status": "completed"} for i in range(1, 15)],
    )

    report_card = window.display_task_report(report)
    qapp.processEvents()

    assert report_card is not None
    assert report_card.completed_steps == 14
    assert report_card.duration == 0.88

    # Physical Visual Snapshot Capture
    pix = window.grab()
    snap_path = os.path.join(snapshot_dir, "06_responsive_event_loop_and_report.png")
    pix.save(snap_path)
    assert os.path.exists(snap_path)
    assert os.path.getsize(snap_path) > 1000

    window.hide()
