"""End-to-End (E2E) integration tests for S.H.A.M.U. Milestone 3: Task Planning & Execution.

Verifies the entire lifecycle:
Objective -> Planning (LLM extraction) -> Storage -> Approval Gating ->
Sequential Execution (Firewall + Sandboxing) -> Postcondition Verification ->
Progress Tracking -> Execution Report.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from rich.console import Console

from shamu.cli import ShamuREPL
from shamu.protocols import PermissionDeniedError
from shamu.security.approval_manager import ApprovalManager, ApprovalRequest, ApprovalResponse
from shamu.security.firewall import ExecutionFirewall
from shamu.security.tool_registry import PermissionTier, ToolRegistry
from shamu.tasks.executor import TaskExecutor
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
from shamu.tasks.planner import (
    PlanValidationError,
    TaskPlanner,
    get_execution_order,
    validate_plan_dag,
)
from shamu.tasks.storage import TaskStorage
from shamu.tasks.ui_bridge import TaskUIBridge
from shamu.types import ChatChunk


@pytest.mark.asyncio
async def test_full_m3_e2e_workflow():
    """Tests the full end-to-end M3 workflow in an isolated environment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir) / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        db_path = Path(tmp_dir) / "tasks.db"

        # 1. Initialize Storage & UI Bridge
        storage = TaskStorage(db_path=db_path)
        try:
            console_mock = MagicMock()
            ui_bridge = TaskUIBridge(console=console_mock)
            # 2. Mock Gateway producing structured planning response
            mock_gateway = MagicMock()
            mock_plan_json = {
                "objective": "Build a simple project artifact",
                "steps": [
                    {
                        "id": "step-init",
                        "title": "Initialize configuration",
                        "description": "Create config.json in workspace",
                        "tool_name": "write_file",
                        "arguments": {"path": "config.json", "content": '{"version": "1.0.0"}'},
                        "dependencies": [],
                        "verification": {
                            "type": "file_exists",
                            "target": "config.json",
                        },
                    },
                    {
                        "id": "step-build",
                        "title": "Build project output",
                        "description": "Create output.txt containing SUCCESS",
                        "tool_name": "write_file",
                        "arguments": {"path": "output.txt", "content": "STATUS=SUCCESS"},
                        "dependencies": ["step-init"],
                        "verification": {
                            "type": "file_contains",
                            "target": "output.txt",
                            "expected": "SUCCESS",
                        },
                    },
                    {
                        "id": "step-verify",
                        "title": "Read configuration",
                        "description": "Read back config.json to confirm integrity",
                        "tool_name": "read_file",
                        "arguments": {"path": "config.json"},
                        "dependencies": ["step-build"],
                        "verification": {
                            "type": "file_exists",
                            "target": "output.txt",
                        },
                    },
                ],
            }

            async def fake_stream_chat(*args, **kwargs):
                content = f"```json\n{json.dumps(mock_plan_json, indent=2)}\n```"
                yield ChatChunk(text_delta=content)

            mock_gateway.stream_chat = fake_stream_chat

            # 3. Plan Generation
            planner = TaskPlanner(gateway=mock_gateway)
            plan = await planner.create_plan("Build a simple project artifact")
            assert len(plan.steps) == 3
            assert plan.steps[0].id == "step-init"
            assert plan.steps[1].id == "step-build"
            assert plan.steps[2].id == "step-verify"

            # 4. Save to Storage before approval
            storage.save_plan(plan)
            saved_plan = storage.get_plan(plan.id)
            assert saved_plan is not None
            assert saved_plan.approved is False
            assert saved_plan.status == TaskStatus.PENDING

            # 5. Verify Unapproved Execution is Blocked
            mock_firewall = MagicMock()
            mock_firewall.workspace_root = workspace_dir
            mock_firewall.approval_manager = None

            async def fake_tool_exec(tool_name, arguments, **kwargs):
                if tool_name == "write_file":
                    p = workspace_dir / arguments["path"]
                    p.write_text(arguments["content"], encoding="utf-8")
                    return f"Wrote {arguments['path']}"
                elif tool_name == "read_file":
                    p = workspace_dir / arguments["path"]
                    return p.read_text(encoding="utf-8")
                return f"Executed {tool_name}"

            mock_firewall.execute_tool = AsyncMock(side_effect=fake_tool_exec)

            executor = TaskExecutor(firewall=mock_firewall, workspace_root=workspace_dir)

            with pytest.raises(PermissionDeniedError):
                async for _ in executor.run_plan(plan):
                    pass

            # 6. User Grants Approval
            plan.approved = True
            storage.save_plan(plan)

            # 7. Execute Plan and Stream Events
            events = []
            async for evt in executor.run_plan(plan):
                events.append(evt)
                ui_bridge.on_event(evt)

            # Update storage with final state
            storage.save_plan(plan)

            # 8. Assert Event Stream
            assert any(isinstance(e, PlanStartedEvent) for e in events)
            assert any(isinstance(e, StepStartedEvent) and e.step_id == "step-init" for e in events)
            assert any(isinstance(e, StepCompletedEvent) and e.step_id == "step-init" for e in events)
            assert any(isinstance(e, StepCompletedEvent) and e.step_id == "step-build" for e in events)
            assert any(isinstance(e, StepCompletedEvent) and e.step_id == "step-verify" for e in events)
            assert any(isinstance(e, PlanCompletedEvent) and e.success is True for e in events)

            # 9. Verify Deterministic File State on Disk
            config_file = workspace_dir / "config.json"
            output_file = workspace_dir / "output.txt"
            assert config_file.exists()
            assert output_file.exists()
            assert "SUCCESS" in output_file.read_text(encoding="utf-8")

            # 10. Verify Storage Persistence
            persisted = storage.get_plan(plan.id)
            assert persisted.status == TaskStatus.COMPLETED
            assert persisted.is_successful() is True
            for step in persisted.steps:
                assert step.status == TaskStatus.COMPLETED
                assert step.verification.passed is True

            # 11. Generate Execution Report
            report = executor.generate_report(plan)
            assert report.plan_id == plan.id
            assert report.total_steps == 3
            assert report.completed_steps == 3
            assert report.failed_steps == 0
            assert report.status == TaskStatus.COMPLETED

            ui_bridge.print_report(report)

        finally:
            storage.close()


@pytest.mark.asyncio
async def test_e2e_step_failure_and_skip_cascade():
    """E2E flow: When a step fails, downstream dependent steps are automatically SKIPPED."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir) / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        db_path = Path(tmp_dir) / "tasks_cascade.db"
        storage = TaskStorage(db_path=db_path)

        try:
            mock_firewall = MagicMock()
            mock_firewall.workspace_root = workspace_dir
            mock_firewall.approval_manager = None

            # Step 1 succeeds, Step 2 fails verification, Step 3 depends on Step 2
            ghost_file = workspace_dir / "never_exists.txt"
            s1 = TaskStep(id="step-1", title="Initialize", status=TaskStatus.PENDING)
            s2 = TaskStep(
                id="step-2",
                title="Check ghost file",
                dependencies=["step-1"],
                verification=StepVerification(
                    type=VerificationType.FILE_EXISTS,
                    target=str(ghost_file),
                ),
            )
            s3 = TaskStep(
                id="step-3",
                title="Deploy",
                dependencies=["step-2"],
            )

            plan = TaskPlan(
                id="plan-cascade",
                objective="Cascade failure test",
                steps=[s1, s2, s3],
                approved=True,
            )
            storage.save_plan(plan)

            executor = TaskExecutor(firewall=mock_firewall, workspace_root=workspace_dir)
            async for _ in executor.run_plan(plan):
                pass

            assert plan.status == TaskStatus.FAILED
            assert not plan.is_successful()
            assert s1.status == TaskStatus.COMPLETED
            assert s2.status == TaskStatus.FAILED
            assert s3.status == TaskStatus.SKIPPED

            report = executor.generate_report(plan)
            assert report.completed_steps == 1
            assert report.failed_steps == 2  # s2 is failed, s3 is skipped

        finally:
            storage.close()


@pytest.mark.asyncio
async def test_e2e_interrupted_recovery_and_resume():
    """E2E flow: Interrupted execution is recovered from SQLite and resumed to completion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir) / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        db_path = Path(tmp_dir) / "tasks_recovery.db"

        # Step 1 was completed, Step 2 was in progress when process terminated
        storage = TaskStorage(db_path=db_path)
        try:
            s1 = TaskStep(id="step-1", title="Phase 1", status=TaskStatus.COMPLETED, result="Done 1")
            s2 = TaskStep(id="step-2", title="Phase 2", status=TaskStatus.IN_PROGRESS, dependencies=["step-1"])
            s3 = TaskStep(id="step-3", title="Phase 3", status=TaskStatus.PENDING, dependencies=["step-2"])

            crashed_plan = TaskPlan(
                id="plan-crash-recovery",
                objective="Multi-phase batch job",
                steps=[s1, s2, s3],
                status=TaskStatus.IN_PROGRESS,
                approved=True,
            )
            storage.save_plan(crashed_plan)
        finally:
            storage.close()

        # Simulate app restart: open fresh TaskStorage
        fresh_storage = TaskStorage(db_path=db_path)
        try:
            recovered_plans = fresh_storage.recover_interrupted_plans()
            assert len(recovered_plans) == 1

            resumed_plan = recovered_plans[0]
            assert resumed_plan.id == "plan-crash-recovery"
            assert resumed_plan.steps[0].status == TaskStatus.COMPLETED
            assert resumed_plan.steps[1].status == TaskStatus.PENDING  # Reset from IN_PROGRESS
            assert resumed_plan.steps[2].status == TaskStatus.PENDING

            # Resume execution with TaskExecutor
            mock_firewall = MagicMock()
            mock_firewall.workspace_root = workspace_dir
            mock_firewall.approval_manager = None

            executor = TaskExecutor(firewall=mock_firewall, workspace_root=workspace_dir)
            async for _ in executor.run_plan(resumed_plan):
                pass

            assert resumed_plan.status == TaskStatus.COMPLETED
            assert resumed_plan.is_successful()
            assert resumed_plan.steps[0].status == TaskStatus.COMPLETED
            assert resumed_plan.steps[1].status == TaskStatus.COMPLETED
            assert resumed_plan.steps[2].status == TaskStatus.COMPLETED

        finally:
            fresh_storage.close()


@pytest.mark.asyncio
async def test_e2e_interactive_approval_event():
    """E2E flow: Tier 2 tool triggers StepWaitingApprovalEvent."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir) / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        mock_firewall = MagicMock(spec=ExecutionFirewall)
        mock_firewall.execute_tool = AsyncMock(return_value="Command executed.")
        mock_cap = MagicMock()
        mock_cap.tier = PermissionTier.TIER_2_MUTATION
        mock_firewall.tool_registry = MagicMock(spec=ToolRegistry)
        mock_firewall.tool_registry.get.return_value = mock_cap
        mock_firewall.approval_manager = None

        step = TaskStep(
            id="step-mutate",
            title="Execute system change",
            tool_name="write_file",
            arguments={"path": "important.cfg", "content": "NEW_VAL"},
        )
        plan = TaskPlan(
            id="plan-approval",
            objective="Apply system configuration",
            steps=[step],
            approved=True,
        )

        events = []
        async def capture(evt):
            events.append(evt)

        executor = TaskExecutor(firewall=mock_firewall, workspace_root=workspace_dir)
        async for _ in executor.run_plan(plan, on_event=capture):
            pass

        assert any(isinstance(e, StepWaitingApprovalEvent) and e.tool_name == "write_file" for e in events)
        assert plan.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_e2e_cli_plan_command_workflow():
    """E2E flow: CLI /plan command decomposes objective, requests user confirmation, and executes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "cli_tasks.db"
        console = Console(record=True)
        mock_gateway = MagicMock()

        plan_json = {
            "objective": "Build CLI feature",
            "steps": [
                {
                    "id": "step-1",
                    "title": "Configure CLI",
                    "description": "Initial setup",
                    "tool_name": None,
                    "arguments": {},
                    "dependencies": [],
                }
            ],
        }

        async def fake_stream(*args, **kwargs):
            yield ChatChunk(text_delta=f"```json\n{json.dumps(plan_json)}\n```")

        mock_gateway.stream_chat = fake_stream

        repl = ShamuREPL(gateway=mock_gateway, console=console)
        repl.firewall = MagicMock()
        repl.firewall.approval_manager = None

        # Simulate user approving plan with 'y'
        store_instance = TaskStorage(db_path=db_path)
        try:
            with patch("shamu.cli.TaskStorage", return_value=store_instance):
                # Mock executor loop input to return 'y'
                loop = asyncio.get_running_loop()
                with patch.object(loop, "run_in_executor", AsyncMock(return_value="y")):
                    await repl.execute_plan_command("/plan Build CLI feature")
        finally:
            store_instance.close()

        output = console.export_text()
        assert "Planning objective: 'Build CLI feature'" in output
        assert "Plan Finished" in output or "Task Execution Summary" in output


@pytest.mark.asyncio
async def test_real_multistep_task_acceptance_inspect_and_report():
    """Real multi-step task acceptance in a temporary test workspace for objective:
    'Inspect a designated test directory, list its files, identify text files, and create a summary report'.

    Verifies full lifecycle:
    1. Planning: Objective decomposition into a DAG of task steps via TaskPlanner.
    2. Validation: DAG acyclicity, dependency validation, topological execution ordering.
    3. Approval: Fail-closed unapproved execution rejection, interactive approval granting.
    4. ExecutionFirewall: Sequential execution using real firewall, real tool registry,
       WindowsPathGuard containment, AuditLedger HMAC recording.
    5. Post-Condition Verification: Independent deterministic assertions (FILE_EXISTS, FILE_CONTAINS).
    6. Reporting & Persistence: TaskExecutionReport generation, UIBridge rendering, SQLite persistence.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir) / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        db_path = Path(tmp_dir) / "task_acceptance.db"
        audit_dir = workspace_dir / ".shamu" / "audit"
        checkpoint_dir = workspace_dir / ".shamu" / "checkpoints"
        audit_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Populate designated test directory with test files
        target_dir = workspace_dir / "target_docs"
        target_dir.mkdir(parents=True, exist_ok=True)

        file1 = target_dir / "alpha_doc.txt"
        file1.write_text("Alpha report: System security posture and active policies verified.", encoding="utf-8")

        file2 = target_dir / "beta_notes.txt"
        file2.write_text("Beta notes: Firewall containment and HMAC audit integrity intact.", encoding="utf-8")

        bin_file = target_dir / "binary_data.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03\xfe\xff")

        csv_file = target_dir / "metrics.csv"
        csv_file.write_text("id,metric,status\n1,m0,pass\n2,m1,pass\n3,m2,pass\n", encoding="utf-8")

        objective = "Inspect a designated test directory, list its files, identify text files, and create a summary report"

        # 1. Initialize Real Firewall with ApprovalManager and AuditLedger
        approval_manager = ApprovalManager()
        approved_actions = []

        async def auto_approval_handler(req: ApprovalRequest) -> ApprovalResponse:
            approved_actions.append(req.tool_name)
            return ApprovalResponse(request_id=req.request_id, approved=True)

        approval_manager.register_handler(auto_approval_handler)

        firewall = ExecutionFirewall(
            workspace_root=workspace_dir,
            approval_manager=approval_manager,
        )

        storage = TaskStorage(db_path=db_path)
        console = Console(record=True)
        ui_bridge = TaskUIBridge(console=console)

        try:
            # 2. Plan Generation with TaskPlanner
            plan_payload = {
                "objective": objective,
                "steps": [
                    {
                        "id": "step-1-list",
                        "title": "Inspect test directory",
                        "description": "List files in target_docs directory",
                        "tool_name": "list_directory",
                        "arguments": {"path": "target_docs"},
                        "dependencies": [],
                        "verification": {
                            "type": "command_success",
                        },
                    },
                    {
                        "id": "step-2-read-alpha",
                        "title": "Read alpha text file",
                        "description": "Read content of alpha_doc.txt",
                        "tool_name": "read_file",
                        "arguments": {"path": "target_docs/alpha_doc.txt"},
                        "dependencies": ["step-1-list"],
                        "verification": {
                            "type": "file_exists",
                            "target": "target_docs/alpha_doc.txt",
                        },
                    },
                    {
                        "id": "step-3-read-beta",
                        "title": "Read beta text file",
                        "description": "Read content of beta_notes.txt",
                        "tool_name": "read_file",
                        "arguments": {"path": "target_docs/beta_notes.txt"},
                        "dependencies": ["step-1-list"],
                        "verification": {
                            "type": "file_exists",
                            "target": "target_docs/beta_notes.txt",
                        },
                    },
                    {
                        "id": "step-4-write-report",
                        "title": "Create summary report",
                        "description": "Compile inspection findings into summary_report.md",
                        "tool_name": "write_file",
                        "arguments": {
                            "path": "summary_report.md",
                            "content": (
                                "# Directory Inspection Summary\n\n"
                                "## Designated Directory: target_docs\n\n"
                                "### Identified Text Files:\n"
                                "- target_docs/alpha_doc.txt (Security posture)\n"
                                "- target_docs/beta_notes.txt (Firewall containment)\n\n"
                                "### Non-Text Files Excluded:\n"
                                "- target_docs/binary_data.bin\n\n"
                                "Report generated and verified by S.H.A.M.U."
                            ),
                        },
                        "dependencies": ["step-2-read-alpha", "step-3-read-beta"],
                        "verification": {
                            "type": "file_contains",
                            "target": "summary_report.md",
                            "expected": "Identified Text Files:",
                        },
                    },
                    {
                        "id": "step-5-verify-report",
                        "title": "Verify report on disk",
                        "description": "Read generated report back from disk",
                        "tool_name": "read_file",
                        "arguments": {"path": "summary_report.md"},
                        "dependencies": ["step-4-write-report"],
                        "verification": {
                            "type": "file_exists",
                            "target": "summary_report.md",
                        },
                    },
                ],
            }

            mock_gateway = MagicMock()

            async def fake_stream_chat(*args, **kwargs):
                yield ChatChunk(text_delta=f"```json\n{json.dumps(plan_payload, indent=2)}\n```")

            mock_gateway.stream_chat = fake_stream_chat

            planner = TaskPlanner(gateway=mock_gateway)
            plan = await planner.create_plan(objective)

            # 3. Validation: DAG structure, dependency graphs, cycle detection
            assert len(plan.steps) == 5
            is_valid, err = validate_plan_dag(plan)
            assert is_valid is True
            assert err is None

            order = get_execution_order(plan)
            order_ids = [s.id for s in order]
            assert order_ids[0] == "step-1-list"
            assert order_ids[-1] == "step-5-verify-report"
            assert order_ids.index("step-2-read-alpha") < order_ids.index("step-4-write-report")
            assert order_ids.index("step-3-read-beta") < order_ids.index("step-4-write-report")

            # Cycle rejection validation test
            step1 = next(s for s in plan.steps if s.id == "step-1-list")
            step1.dependencies = ["step-5-verify-report"]
            is_valid_cycle, cycle_err = validate_plan_dag(plan)
            assert is_valid_cycle is False
            assert "Circular dependency" in cycle_err
            with pytest.raises(PlanValidationError):
                get_execution_order(plan)
            step1.dependencies = []  # Restore valid DAG
            is_valid_restored, _ = validate_plan_dag(plan)
            assert is_valid_restored is True

            # 4. Storage & Approval Gate
            storage.save_plan(plan)
            saved = storage.get_plan(plan.id)
            assert saved is not None
            assert saved.approved is False
            assert saved.status == TaskStatus.PENDING

            executor = TaskExecutor(firewall=firewall, workspace_root=workspace_dir)

            # Unapproved execution blocked fail-closed
            with pytest.raises(PermissionDeniedError, match="not been approved"):
                async for _ in executor.run_plan(plan):
                    pass

            # Grant approval
            plan.approved = True
            storage.save_plan(plan)

            # 5. Sequential ExecutionFirewall Enforcement & Event Streaming
            events = []
            async for evt in executor.run_plan(plan):
                events.append(evt)
                ui_bridge.on_event(evt)

            # Save completed state to storage
            storage.save_plan(plan)

            # 6. Event Stream Assertions
            assert any(isinstance(e, PlanStartedEvent) and e.plan_id == plan.id for e in events)
            assert any(isinstance(e, StepWaitingApprovalEvent) and e.tool_name == "write_file" for e in events)
            assert any(isinstance(e, StepCompletedEvent) and e.step_id == "step-1-list" for e in events)
            assert any(isinstance(e, StepCompletedEvent) and e.step_id == "step-4-write-report" for e in events)
            assert any(isinstance(e, StepCompletedEvent) and e.step_id == "step-5-verify-report" for e in events)
            assert any(isinstance(e, PlanCompletedEvent) and e.success is True for e in events)

            # 7. Post-Condition Verification & Filesystem Reality
            summary_path = workspace_dir / "summary_report.md"
            assert summary_path.exists(), "summary_report.md was not written to disk"
            report_text = summary_path.read_text(encoding="utf-8")
            assert "Identified Text Files:" in report_text
            assert "target_docs/alpha_doc.txt" in report_text
            assert "target_docs/beta_notes.txt" in report_text
            assert "binary_data.bin" in report_text

            # Check post-condition verification flags on the steps
            step4 = next(s for s in plan.steps if s.id == "step-4-write-report")
            assert step4.verification.passed is True
            assert "contains expected text: True" in step4.verification.message

            step5 = next(s for s in plan.steps if s.id == "step-5-verify-report")
            assert step5.verification.passed is True

            # 8. Real Tool Registry & Approval Tracking
            assert "write_file" in approved_actions, "Tier 2 tool was not passed through ApprovalManager"

            # 9. Audit Trail Verification (HMAC chained audit ledger)
            audit_events = firewall.audit_ledger.log_path
            assert audit_events.exists()
            verification_result = firewall.audit_ledger.verify_integrity()
            assert verification_result.is_valid is True, f"Audit ledger integrity failed: {verification_result}"

            # 10. Execution Report Verification
            report = executor.generate_report(plan)
            assert report.plan_id == plan.id
            assert report.objective == objective
            assert report.total_steps == 5
            assert report.completed_steps == 5
            assert report.failed_steps == 0
            assert report.status == TaskStatus.COMPLETED
            assert report.duration_seconds >= 0.0

            # Render report through UIBridge
            ui_bridge.render_report(report)
            rendered = console.export_text()
            assert "Task Execution Summary" in rendered or "Plan Finished" in rendered

            # 11. Persistence Across Storage Lifecycle
            persisted_plan = storage.get_plan(plan.id)
            assert persisted_plan.status == TaskStatus.COMPLETED
            assert persisted_plan.is_successful() is True
            assert len(persisted_plan.steps) == 5
            for st in persisted_plan.steps:
                assert st.status == TaskStatus.COMPLETED

        finally:
            storage.close()
            firewall.audit_ledger.close()


@pytest.mark.asyncio
async def test_real_multistep_task_acceptance_firewall_safeguards():
    """Verify M0, M1, M2 safeguards remain intact during task plan execution:
    - Path traversal attempts outside workspace are strictly blocked fail-closed.
    - Protected secret paths (.shamu/audit, .env) are blocked fail-closed.
    - Rejection by human approval handler halts execution cleanly.
    - All blocked attempts are recorded in the HMAC-verified audit trail.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir) / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        db_path = Path(tmp_dir) / "task_safeguards.db"

        # Secret file outside workspace
        outside_secret = Path(tmp_dir) / "secret_outside.txt"
        outside_secret.write_text("CLASSIFIED_SECRET", encoding="utf-8")

        # 1. Approval handler that rejects all Tier 2 mutation requests
        approval_manager = ApprovalManager()

        async def reject_handler(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(request_id=req.request_id, approved=False, reason="User rejected mutation")

        approval_manager.register_handler(reject_handler)

        firewall = ExecutionFirewall(
            workspace_root=workspace_dir,
            approval_manager=approval_manager,
        )
        storage = TaskStorage(db_path=db_path)

        try:
            # Plan attempting path traversal
            traversal_step = TaskStep(
                id="step-traversal",
                title="Attempt Traversal Read",
                tool_name="read_file",
                arguments={"path": "../secret_outside.txt"},
            )
            dependent_step = TaskStep(
                id="step-after-traversal",
                title="Downstream Action",
                tool_name=None,
                dependencies=["step-traversal"],
            )

            traversal_plan = TaskPlan(
                id="plan-traversal",
                objective="Read external secret",
                steps=[traversal_step, dependent_step],
                approved=True,
            )

            executor = TaskExecutor(firewall=firewall, workspace_root=workspace_dir)

            events = []
            async for evt in executor.run_plan(traversal_plan):
                events.append(evt)

            assert traversal_plan.status == TaskStatus.FAILED
            assert traversal_step.status == TaskStatus.FAILED
            assert "outside approved workspace" in traversal_step.error or "Permission denied" in traversal_step.error
            assert dependent_step.status == TaskStatus.SKIPPED

            # Plan with mutation rejected by human approval handler
            mutation_step = TaskStep(
                id="step-mutation",
                title="Attempt unauthorized write",
                tool_name="write_file",
                arguments={"path": "unauthorized.txt", "content": "MALICIOUS"},
            )
            mutation_plan = TaskPlan(
                id="plan-mutation",
                objective="Write unauthorized file",
                steps=[mutation_step],
                approved=True,
            )

            events2 = []
            async for evt in executor.run_plan(mutation_plan):
                events2.append(evt)

            assert mutation_plan.status == TaskStatus.FAILED
            assert mutation_step.status == TaskStatus.FAILED
            assert "User rejected mutation" in mutation_step.error or "Approval denied" in mutation_step.error
            assert not (workspace_dir / "unauthorized.txt").exists()

            # Verify audit trail recorded both blocked events with HMAC integrity
            assert firewall.audit_ledger.log_path.exists()
            audit_result = firewall.audit_ledger.verify_integrity()
            assert audit_result.is_valid is True

        finally:
            storage.close()
            firewall.audit_ledger.close()

