"""Unit and integration tests for TaskExecutor, firewall gating, and post-condition verification."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from shamu.protocols import PermissionDeniedError
from shamu.tasks.executor import TaskExecutor, verify_postcondition
from shamu.tasks.models import (
    PlanCompletedEvent,
    PlanStartedEvent,
    StepCompletedEvent,
    StepFailedEvent,
    StepStartedEvent,
    StepVerification,
    StepWaitingApprovalEvent,
    TaskPlan,
    TaskStatus,
    TaskStep,
    VerificationType,
)


def test_verify_postcondition_file_exists():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        test_file = root / "sample.txt"
        v = StepVerification(type=VerificationType.FILE_EXISTS, target="sample.txt")

        # Initially does not exist
        assert verify_postcondition(v, workspace_root=root) is False
        assert v.passed is False

        # Create file
        test_file.write_text("hello", encoding="utf-8")
        assert verify_postcondition(v, workspace_root=root) is True
        assert v.passed is True


def test_verify_postcondition_file_contains():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        test_file = root / "output.log"
        test_file.write_text("Build SUCCESSFUL in 12s", encoding="utf-8")

        v_pass = StepVerification(
            type=VerificationType.FILE_CONTAINS,
            target="output.log",
            expected="SUCCESSFUL",
        )
        assert verify_postcondition(v_pass, workspace_root=root) is True

        v_fail = StepVerification(
            type=VerificationType.FILE_CONTAINS,
            target="output.log",
            expected="ERROR",
        )
        assert verify_postcondition(v_fail, workspace_root=root) is False


def test_verify_postcondition_file_not_exists():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        v = StepVerification(type=VerificationType.FILE_NOT_EXISTS, target="cleaned.tmp")
        assert verify_postcondition(v, workspace_root=root) is True

        (root / "cleaned.tmp").write_text("temp", encoding="utf-8")
        assert verify_postcondition(v, workspace_root=root) is False


@pytest.mark.asyncio
async def test_task_executor_unapproved_plan_raises():
    """Unapproved plans must be refused by the executor before executing any step."""
    mock_firewall = MagicMock()
    executor = TaskExecutor(firewall=mock_firewall)

    plan = TaskPlan(
        id="plan-unapproved",
        objective="Run unapproved action",
        steps=[TaskStep(id="s1", title="Step 1")],
        approved=False,
    )

    with pytest.raises(PermissionDeniedError, match="not been approved"):
        async for _ in executor.run_plan(plan):
            pass


@pytest.mark.asyncio
async def test_task_executor_sequential_execution_and_events():
    """Verifies that an approved plan runs sequentially, calls the firewall, verifies post-conditions, and emits events."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)

        # Mock firewall
        mock_firewall = MagicMock()
        mock_firewall.workspace_root = root

        async def fake_execute_tool(tool_name, arguments, **kwargs):
            if tool_name == "write_file":
                p = root / arguments["path"]
                p.write_text(arguments["content"], encoding="utf-8")
                return f"Wrote {arguments['path']}"
            return f"Executed {tool_name}"

        mock_firewall.execute_tool = AsyncMock(side_effect=fake_execute_tool)

        executor = TaskExecutor(firewall=mock_firewall, workspace_root=root)

        # Build 2-step plan with dependency
        step1 = TaskStep(
            id="s1",
            title="Create file",
            tool_name="write_file",
            arguments={"path": "result.txt", "content": "Milestone 3 Done"},
            dependencies=[],
            verification=StepVerification(
                type=VerificationType.FILE_CONTAINS,
                target="result.txt",
                expected="Milestone 3",
            ),
        )
        step2 = TaskStep(
            id="s2",
            title="Verify completion",
            tool_name="read_file",
            arguments={"path": "result.txt"},
            dependencies=["s1"],
            verification=StepVerification(
                type=VerificationType.FILE_EXISTS,
                target="result.txt",
            ),
        )

        plan = TaskPlan(
            id="plan-run",
            objective="Write and verify output",
            steps=[step1, step2],
            approved=True,
        )

        events = []
        async for evt in executor.run_plan(plan):
            events.append(evt)

        # Check events
        assert any(isinstance(e, PlanStartedEvent) for e in events)
        assert any(isinstance(e, StepCompletedEvent) and e.step_id == "s1" for e in events)
        assert any(isinstance(e, StepCompletedEvent) and e.step_id == "s2" for e in events)
        assert any(isinstance(e, PlanCompletedEvent) and e.success is True for e in events)

        # Check final plan state
        assert plan.status == TaskStatus.COMPLETED
        assert step1.status == TaskStatus.COMPLETED
        assert step1.verification.passed is True
        assert step2.status == TaskStatus.COMPLETED
        assert step2.verification.passed is True

        # Generate report
        report = executor.generate_report(plan)
        assert report.plan_id == "plan-run"
        assert report.total_steps == 2
        assert report.completed_steps == 2
        assert report.failed_steps == 0
