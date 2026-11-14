import asyncio
from pathlib import Path
import pytest
from shamu.protocols import PermissionDeniedError
from shamu.security.firewall import ExecutionFirewall
from shamu.security.approval_manager import ApprovalRequest, ApprovalResponse
from shamu.security.audit_ledger import AuditIntegrityError


@pytest.mark.asyncio
async def test_tier_1_read_file_autonomous_execution(tmp_path: Path):
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Autonomous read content", encoding="utf-8")

    firewall = ExecutionFirewall(workspace_root=tmp_path)

    result = await firewall.execute_tool(
        tool_name="read_file",
        arguments={"path": str(doc_path)},
        session_id="test-session",
        turn_id="turn-1",
    )

    assert result == "Autonomous read content"

    # Verify audit ledger (two-phase: intent + completion)
    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records == 2


@pytest.mark.asyncio
async def test_tier_2_write_file_interactive_approval(tmp_path: Path):
    out_file = tmp_path / "output.txt"
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    approval_called = False

    async def mock_handler(req: ApprovalRequest) -> ApprovalResponse:
        nonlocal approval_called
        approval_called = True
        assert req.tool_name == "write_file"
        return ApprovalResponse(request_id=req.request_id, approved=True)

    firewall.approval_manager.register_handler(mock_handler)

    result = await firewall.execute_tool(
        tool_name="write_file",
        arguments={"path": str(out_file), "content": "Approved content"},
        session_id="test-session",
        turn_id="turn-2",
    )

    assert approval_called is True
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == "Approved content"

    # Verify audit ledger (two-phase: intent + completion)
    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records == 2


@pytest.mark.asyncio
async def test_tier_3_powershell_double_confirmation(tmp_path: Path):
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    async def double_confirm_handler(req: ApprovalRequest) -> ApprovalResponse:
        assert req.requires_double_confirmation is True
        return ApprovalResponse(
            request_id=req.request_id,
            approved=True,
            double_confirmed=True,
        )

    firewall.approval_manager.register_handler(double_confirm_handler)

    res = await firewall.execute_tool(
        tool_name="powershell_exec",
        arguments={"command": "Write-Output 'P3 Verified'"},
        session_id="test-session",
        turn_id="turn-3",
    )

    assert "P3 Verified" in res.stdout
    assert res.exit_code == 0

    # Verify audit ledger recorded the execution (two-phase: intent + completion)
    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records == 2


@pytest.mark.asyncio
async def test_denied_approval_aborts_execution(tmp_path: Path):
    target_file = tmp_path / "never_created.txt"
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    async def reject_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            request_id=req.request_id,
            approved=False,
            reason="User denied action",
        )

    firewall.approval_manager.register_handler(reject_handler)

    with pytest.raises(PermissionDeniedError, match="User denied action"):
        await firewall.execute_tool(
            tool_name="write_file",
            arguments={"path": str(target_file), "content": "Should not exist"},
        )

    assert not target_file.exists()

    # Denied event was still committed to audit trail
    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records == 1


@pytest.mark.asyncio
async def test_path_violation_outside_workspace_rejected(tmp_path: Path):
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    with pytest.raises(PermissionDeniedError, match="outside approved workspace"):
        await firewall.execute_tool(
            tool_name="read_file",
            arguments={"path": "C:\\Windows\\System32\\drivers\\etc\\hosts"},
        )

    # Denied attempt committed to audit trail
    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records == 1


@pytest.mark.asyncio
async def test_prohibited_tool_rejected(tmp_path: Path):
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    with pytest.raises(PermissionDeniedError, match="permanently prohibited"):
        await firewall.execute_tool(
            tool_name="format_disk",
            arguments={"drive": "C:"},
        )

    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records == 1


@pytest.mark.asyncio
async def test_fail_closed_on_audit_ledger_corruption(tmp_path: Path):
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("test", encoding="utf-8")

    firewall = ExecutionFirewall(workspace_root=tmp_path)

    # Run valid tool
    await firewall.execute_tool("read_file", {"path": str(doc_path)})

    # Corrupt the audit ledger line
    lines = firewall.audit_ledger.log_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("read_file", "forged_tool")
    firewall.audit_ledger.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Run integrity check - marks ledger locked
    check = firewall.audit_ledger.verify_integrity()
    assert check.is_valid is False
    assert firewall.audit_ledger.is_locked is True

    # Any subsequent tool execution must fail closed
    with pytest.raises(AuditIntegrityError, match="locked"):
        await firewall.execute_tool("read_file", {"path": str(doc_path)})


@pytest.mark.asyncio
async def test_tier_3_powershell_prohibited_command_rejected_by_firewall(tmp_path: Path):
    """SEC-02: Prohibited PowerShell execution through ExecutionFirewall must be rejected and audited."""
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    async def double_confirm_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            request_id=req.request_id,
            approved=True,
            double_confirmed=True,
        )

    firewall.approval_manager.register_handler(double_confirm_handler)

    # Attempt an ExecutionPolicy bypass
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await firewall.execute_tool(
            tool_name="powershell_exec",
            arguments={"command": "powershell.exe -ExecutionPolicy Bypass -Command 'Get-Process'"},
            session_id="test-session",
            turn_id="turn-ps-bypass",
        )

    # Attempt a filesystem traversal outside workspace
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await firewall.execute_tool(
            tool_name="powershell_exec",
            arguments={"command": r"Get-Content C:\Windows\win.ini"},
            session_id="test-session",
            turn_id="turn-ps-path",
        )

    # Audit ledger records both execution attempts (each has intent + failure = 4 records)
    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records == 4


@pytest.mark.asyncio
async def test_cmd_exec_rejected_by_firewall(tmp_path: Path):
    """SEC-02: cmd_exec is disabled and must be rejected as an unapproved capability."""
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    with pytest.raises(PermissionDeniedError, match="not registered"):
        await firewall.execute_tool(
            tool_name="cmd_exec",
            arguments={"command": "dir"},
            session_id="test-session",
            turn_id="turn-cmd",
        )

    # Rejection was audited
    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records == 1


@pytest.mark.asyncio
async def test_powershell_dynamic_attacks_rejected_by_firewall(tmp_path: Path):
    """SEC-02: Dynamic PowerShell evasion attempts must be rejected by ExecutionFirewall."""
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    async def double_confirm_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            request_id=req.request_id,
            approved=True,
            double_confirmed=True,
        )

    firewall.approval_manager.register_handler(double_confirm_handler)

    # 1. Environment variable path
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await firewall.execute_tool(
            tool_name="powershell_exec",
            arguments={"command": r"Get-Content $env:WINDIR\win.ini"},
        )

    # 2. Dynamic invocation via &
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await firewall.execute_tool(
            tool_name="powershell_exec",
            arguments={"command": "$c = 'calc'; & $c"},
        )

    # 3. String concatenation
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await firewall.execute_tool(
            tool_name="powershell_exec",
            arguments={"command": r'"C:\Win" + "dows\win.ini"'},
        )


@pytest.mark.asyncio
async def test_powershell_foreach_object_rejected_by_firewall(tmp_path: Path):
    """SEC-02-R2-01: ForEach-Object / % method execution must be rejected fail-closed by ExecutionFirewall and audited."""
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    async def double_confirm_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            request_id=req.request_id,
            approved=True,
            double_confirmed=True,
        )

    firewall.approval_manager.register_handler(double_confirm_handler)

    # 1. Full cmdlet name with method name
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await firewall.execute_tool(
            tool_name="powershell_exec",
            arguments={"command": "Get-Process -Id 123 | ForEach-Object -MemberName Kill"},
            session_id="sec02-sess",
            turn_id="turn-foreach-1",
        )

    # 2. Alias % with positional method name
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await firewall.execute_tool(
            tool_name="powershell_exec",
            arguments={"command": "Get-Process | % Kill"},
            session_id="sec02-sess",
            turn_id="turn-foreach-2",
        )

    # Audit ledger records both execution attempts (each has intent + failure = 4 records)
    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records == 4


@pytest.mark.asyncio
async def test_two_phase_write_ahead_audit_logging(tmp_path: Path):
    """SEC-07: ExecutionFirewall must write intent before side effects and complete in finally."""
    import json
    firewall = ExecutionFirewall(workspace_root=tmp_path)

    intent_recorded_during_execution = False

    def monitored_executor(args):
        nonlocal intent_recorded_during_execution
        # Read audit log while the tool is inside execution
        lines = firewall.audit_ledger.log_path.read_text(encoding="utf-8").splitlines()
        if len(lines) == 1:
            rec = json.loads(lines[0])
            if rec["action_type"] == "EXECUTE_INTENT" and rec["outcome"] == "PENDING":
                intent_recorded_during_execution = True
        return "custom execution output"

    firewall.register_custom_executor("read_file", monitored_executor)

    res = await firewall.execute_tool(
        tool_name="read_file",
        arguments={"path": str(tmp_path / "dummy.txt")},
        session_id="sess-sec07",
        turn_id="turn-1",
    )

    assert res == "custom execution output"
    assert intent_recorded_during_execution is True, "Intent must be recorded before tool execution starts"

    # Integrity verification must succeed with both records present
    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records == 2

    lines = firewall.audit_ledger.log_path.read_text(encoding="utf-8").splitlines()
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["action_type"] == "EXECUTE_INTENT"
    assert rec1["outcome"] == "PENDING"
    assert rec2["action_type"] == "EXECUTE_COMPLETE"
    assert rec2["outcome"] == "SUCCESS"





