"""Tests for ExecutionFirewall process protection and Windows critical system binary guard (CANON-P1-01)."""

import asyncio
import os
import sys
import time
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import psutil

from shamu.protocols import PermissionDeniedError, ToolExecutionError
from shamu.security.approval_manager import ApprovalManager, ApprovalResponse
from shamu.security.audit_ledger import AuditLedger
from shamu.security.firewall import (
    CRITICAL_SYSTEM_PROCESSES,
    ExecutionFirewall,
    PermissionTier,
    ProcessProtectionError,
    SecurityFirewall,
    is_critical_process_name,
)


def _create_test_firewall(ws: Path, memory_store=None) -> ExecutionFirewall:
    audit_dir = ws / ".shamu" / "audit"
    cp_dir = ws / ".shamu" / "checkpoints"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cp_dir.mkdir(parents=True, exist_ok=True)
    ledger = AuditLedger(
        log_path=audit_dir / "audit.jsonl",
        checkpoint_path=cp_dir / "checkpoints.db",
        master_key=b"F" * 32,
    )
    appr_mgr = ApprovalManager()

    async def auto_approve(req):
        return ApprovalResponse(request_id=req.request_id, approved=True, double_confirmed=True)

    appr_mgr.register_handler(auto_approve)
    return ExecutionFirewall(
        workspace_root=ws,
        approval_manager=appr_mgr,
        audit_ledger=ledger,
        memory_store=memory_store,
    )


def test_critical_process_name_helper():
    """Verify is_critical_process_name identifies critical binaries with/without .exe and case-insensitively."""
    assert is_critical_process_name("csrss.exe")
    assert is_critical_process_name("CSRSS.EXE")
    assert is_critical_process_name("csrss")
    assert is_critical_process_name("lsass.exe")
    assert is_critical_process_name("smss.exe")
    assert is_critical_process_name("services.exe")
    assert is_critical_process_name("wininit.exe")
    assert is_critical_process_name("dwm.exe")
    assert is_critical_process_name("svchost.exe")
    assert is_critical_process_name("System")
    assert is_critical_process_name("system idle process")

    # Legitimate non-critical binaries must not match
    assert not is_critical_process_name("python.exe")
    assert not is_critical_process_name("notepad.exe")
    assert not is_critical_process_name("chrome.exe")
    assert not is_critical_process_name("")
    assert not is_critical_process_name(None)


@pytest.mark.asyncio
async def test_system_and_self_pid_protection(tmp_path: Path):
    """Core system PIDs (<= 4) and current process PID must always be rejected."""
    fw = _create_test_firewall(tmp_path)

    # PID 0 (System Idle)
    with pytest.raises(ProcessProtectionError):
        await fw.execute_tool("kill_process", {"pid": 0})

    # PID 4 (System)
    with pytest.raises(ProcessProtectionError):
        await fw.execute_tool("kill_process", {"pid": 4})

    # Negative PID
    with pytest.raises(ProcessProtectionError):
        await fw.execute_tool("kill_process", {"pid": -1})

    # Current process PID
    with pytest.raises(ProcessProtectionError):
        await fw.execute_tool("kill_process", {"pid": os.getpid()})

    # Parent terminal PID
    if hasattr(os, "getppid"):
        parent_pid = os.getppid()
        with pytest.raises(ProcessProtectionError):
            await fw.execute_tool("kill_process", {"pid": parent_pid})
        dec_parent = fw.evaluate_tool_call("kill_process", {"pid": parent_pid})
        assert dec_parent.tier == PermissionTier.PROHIBITED
        assert not dec_parent.allowed

    # Legacy evaluate_tool_call also marks these PROHIBITED
    dec_4 = fw.evaluate_tool_call("kill_process", {"pid": 4})
    assert dec_4.tier == PermissionTier.PROHIBITED
    assert not dec_4.allowed

    dec_self = fw.evaluate_tool_call("kill_process", {"pid": os.getpid()})
    assert dec_self.tier == PermissionTier.PROHIBITED
    assert not dec_self.allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("binary_name", [
    "csrss.exe",
    "lsass.exe",
    "smss.exe",
    "services.exe",
    "wininit.exe",
    "winlogon.exe",
    "dwm.exe",
    "svchost.exe",
    "CSRSS.EXE",
    "lsass",
    "explorer.exe",
    "explorer",
])
async def test_critical_windows_processes_blocked(tmp_path: Path, binary_name: str):
    """Mocks critical Windows system processes to verify kill_process strictly blocks them."""
    fw = _create_test_firewall(tmp_path)
    fake_pid = 7788

    mock_proc = MagicMock()
    mock_proc.is_running.return_value = True
    mock_proc.create_time.return_value = 1000.0
    mock_proc.name.return_value = binary_name
    mock_proc.exe.return_value = f"C:\\Windows\\System32\\{binary_name}"

    with patch("psutil.Process", return_value=mock_proc):
        with pytest.raises(ProcessProtectionError) as exc_info:
            await fw.execute_tool("kill_process", {"pid": fake_pid})
        assert "protected Windows critical system process" in str(exc_info.value)
        # Verify proc.terminate() was NEVER called
        assert not mock_proc.terminate.called

        # Also verify evaluate_tool_call marks it PROHIBITED
        dec = fw.evaluate_tool_call("kill_process", {"pid": fake_pid})
        assert dec.tier == PermissionTier.PROHIBITED
        assert not dec.allowed


@pytest.mark.asyncio
async def test_fail_closed_access_denied_on_process_open(tmp_path: Path):
    """Fail-closed: If psutil.Process(pid) raises AccessDenied, termination is rejected."""
    fw = _create_test_firewall(tmp_path)
    fake_pid = 8899

    with patch("psutil.Process", side_effect=psutil.AccessDenied(pid=fake_pid)):
        with pytest.raises(ProcessProtectionError) as exc_info:
            await fw.execute_tool("kill_process", {"pid": fake_pid})
        assert "fail-closed" in str(exc_info.value).lower()

        # evaluate_tool_call also fails closed
        dec = fw.evaluate_tool_call("kill_process", {"pid": fake_pid})
        assert dec.tier == PermissionTier.PROHIBITED
        assert not dec.allowed


@pytest.mark.asyncio
async def test_fail_closed_access_denied_on_name_query(tmp_path: Path):
    """Fail-closed: If proc.name() raises AccessDenied, termination is rejected."""
    fw = _create_test_firewall(tmp_path)
    fake_pid = 8899

    mock_proc = MagicMock()
    mock_proc.is_running.return_value = True
    mock_proc.create_time.return_value = 1000.0
    mock_proc.name.side_effect = psutil.AccessDenied(pid=fake_pid)

    with patch("psutil.Process", return_value=mock_proc):
        with pytest.raises(ProcessProtectionError) as exc_info:
            await fw.execute_tool("kill_process", {"pid": fake_pid})
        assert "fail-closed" in str(exc_info.value).lower()
        assert not mock_proc.terminate.called


@pytest.mark.asyncio
async def test_fail_closed_empty_process_name(tmp_path: Path):
    """Fail-closed: If process name is empty or None, termination is rejected."""
    fw = _create_test_firewall(tmp_path)
    fake_pid = 8899

    mock_proc = MagicMock()
    mock_proc.is_running.return_value = True
    mock_proc.create_time.return_value = 1000.0
    mock_proc.name.return_value = ""

    with patch("psutil.Process", return_value=mock_proc):
        with pytest.raises(ProcessProtectionError) as exc_info:
            await fw.execute_tool("kill_process", {"pid": fake_pid})
        assert "fail-closed" in str(exc_info.value).lower()
        assert not mock_proc.terminate.called


@pytest.mark.asyncio
async def test_toctou_pid_reuse_prevention(tmp_path: Path):
    """PID reuse / TOCTOU protection: If create_time changes between validation and kill, abort."""
    fw = _create_test_firewall(tmp_path)
    fake_pid = 6677

    mock_proc = MagicMock()
    mock_proc.is_running.return_value = True
    # Initial inspection returns create_time 1000.0, post-validation returns 2000.0 (different process reused PID)
    mock_proc.create_time.side_effect = [1000.0, 2000.0]
    mock_proc.name.return_value = "worker_child.exe"
    mock_proc.exe.return_value = "C:\\app\\worker_child.exe"

    with patch("psutil.Process", return_value=mock_proc):
        res = await fw.execute_tool("kill_process", {"pid": fake_pid})
        assert "not found" in res
        assert not mock_proc.terminate.called


@pytest.mark.asyncio
async def test_nonexistent_pid_graceful_handling(tmp_path: Path):
    """Non-existent process PID returns informative message without raising."""
    fw = _create_test_firewall(tmp_path)
    res = await fw.execute_tool("kill_process", {"pid": 9999999})
    assert "not found" in res


@pytest.mark.asyncio
async def test_legitimate_user_process_termination(tmp_path: Path):
    """Legitimate disposable child process is successfully validated and terminated with safe cleanup."""
    fw = _create_test_firewall(tmp_path)

    sub = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        pid = sub.pid
        result = await asyncio.wait_for(
            fw.execute_tool("kill_process", {"pid": pid}),
            timeout=5.0,
        )
        assert "terminated successfully" in result
        sub.wait(timeout=3.0)
        assert sub.poll() is not None
    finally:
        if sub.poll() is None:
            try:
                sub.kill()
                sub.wait(timeout=2.0)
            except Exception:
                pass


@pytest.mark.asyncio
async def test_audit_logging_on_critical_process_block(tmp_path: Path):
    """Blocked attempts on critical processes must be recorded in the audit ledger as DENIED."""
    fw = _create_test_firewall(tmp_path)
    fake_pid = 7711

    mock_proc = MagicMock()
    mock_proc.is_running.return_value = True
    mock_proc.create_time.return_value = 1000.0
    mock_proc.name.return_value = "lsass.exe"

    with patch("psutil.Process", return_value=mock_proc):
        with pytest.raises(ProcessProtectionError):
            await fw.execute_tool("kill_process", {"pid": fake_pid})

    # Read audit log
    audit_file = tmp_path / ".shamu" / "audit" / "audit.jsonl"
    assert audit_file.exists()
    content = audit_file.read_text(encoding="utf-8")
    assert "DENIED" in content
    assert "kill_process" in content
    assert "protected Windows critical system process" in content


@pytest.mark.asyncio
async def test_policy_authorizer_separation_from_execution(tmp_path: Path):
    """Verify PolicyAuthorizer validates policy and returns AuthorizedToolCall without executing concrete action."""
    from shamu.security.firewall import PolicyAuthorizer, AuthorizedToolCall
    from shamu.security.tool_registry import ToolRegistry
    from shamu.security.path_guard import WindowsPathGuard
    from shamu.security.process_guard import WindowsProcessGuard

    reg = ToolRegistry.create_default()
    pg = WindowsPathGuard(workspace_root=tmp_path)
    proc_g = WindowsProcessGuard(workspace_root=tmp_path)
    appr_mgr = ApprovalManager()
    audit_dir = tmp_path / ".shamu" / "audit"
    cp_dir = tmp_path / ".shamu" / "checkpoints"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cp_dir.mkdir(parents=True, exist_ok=True)
    ledger = AuditLedger(
        log_path=audit_dir / "audit.jsonl",
        checkpoint_path=cp_dir / "checkpoints.db",
        master_key=b"F" * 32,
    )

    authorizer = PolicyAuthorizer(
        registry=reg,
        path_guard=pg,
        process_guard=proc_g,
        approval_manager=appr_mgr,
        audit_ledger=ledger,
        workspace_root=tmp_path,
    )

    test_file = tmp_path / "hello.txt"
    test_file.write_text("world", encoding="utf-8")

    # Authorize read_file
    auth_call = await authorizer.authorize_tool_call(
        tool_name="read_file",
        arguments={"path": str(test_file)},
    )
    assert isinstance(auth_call, AuthorizedToolCall)
    assert auth_call.tool_name == "read_file"
    assert auth_call.tier == PermissionTier.TIER_1_READ_ONLY
    assert auth_call.decision == "ALLOW"
    assert "path" in auth_call.arguments


@pytest.mark.asyncio
async def test_security_boundary_no_bypass_unregistered_and_prohibited(tmp_path: Path):
    """Ensure unregistered and prohibited tools strictly fail closed and are audited."""
    fw = _create_test_firewall(tmp_path)

    # 1. Unregistered tool
    with pytest.raises(PermissionDeniedError) as exc_info:
        await fw.execute_tool("malicious_unregistered_tool", {"foo": "bar"})
    assert "not registered" in str(exc_info.value)

    # 2. Prohibited tool
    with pytest.raises(PermissionDeniedError) as exc_info2:
        await fw.execute_tool("format_disk", {})
    assert "permanently prohibited" in str(exc_info2.value)

    # Verify both were faithfully recorded as DENIED in audit ledger
    audit_file = tmp_path / ".shamu" / "audit" / "audit.jsonl"
    content = audit_file.read_text(encoding="utf-8")
    assert "malicious_unregistered_tool" in content
    assert "format_disk" in content
    assert content.count("DENIED") >= 2


@pytest.mark.asyncio
async def test_security_boundary_custom_executor_cannot_bypass_policy(tmp_path: Path):
    """Ensure custom executors cannot execute if unregistered or if path/permission policies are violated."""
    fw = _create_test_firewall(tmp_path)
    custom_executed = False

    def evil_hook(args):
        nonlocal custom_executed
        custom_executed = True
        return "hacked"

    # Register custom executor for an unregistered tool name
    fw.register_custom_executor("unregistered_custom", evil_hook)

    # Attempt execution -> must be blocked at policy authorization stage
    with pytest.raises(PermissionDeniedError):
        await fw.execute_tool("unregistered_custom", {})

    assert not custom_executed

    # Register capability but with prohibited tier
    from shamu.security.tool_registry import ToolCapability
    fw.registry.register(ToolCapability(
        name="prohibited_custom",
        tier=PermissionTier.PROHIBITED,
        description="Banned custom tool",
        parameters_schema={"type": "object", "properties": {}},
        is_prohibited=True,
    ))
    fw.register_custom_executor("prohibited_custom", evil_hook)

    with pytest.raises(PermissionDeniedError):
        await fw.execute_tool("prohibited_custom", {})

    assert not custom_executed


@pytest.mark.asyncio
async def test_two_phase_faithful_audit_trail(tmp_path: Path):
    """Ensure execute_tool faithfully logs EXECUTE_INTENT (PENDING) followed by EXECUTE_COMPLETE (SUCCESS)."""
    fw = _create_test_firewall(tmp_path)

    res = await fw.execute_tool("get_system_metrics", {})
    assert "cpu_percent" in res

    audit_file = tmp_path / ".shamu" / "audit" / "audit.jsonl"
    lines = [line for line in audit_file.read_text(encoding="utf-8").strip().splitlines() if line]
    assert len(lines) == 2

    import json
    intent_record = json.loads(lines[0])
    complete_record = json.loads(lines[1])

    assert intent_record["action_type"] == "EXECUTE_INTENT"
    assert intent_record["outcome"] == "PENDING"
    assert intent_record["tool_name"] == "get_system_metrics"

    assert complete_record["action_type"] == "EXECUTE_COMPLETE"
    assert complete_record["outcome"] == "SUCCESS"
    assert complete_record["tool_name"] == "get_system_metrics"
    assert complete_record["duration_ms"] is not None
    assert complete_record["duration_ms"] >= 0.0


@pytest.mark.asyncio
async def test_tool_executor_isolation(tmp_path: Path):
    """Verify ToolExecutor can execute concrete tool operations given authorized inputs."""
    from shamu.security.firewall import ToolExecutor
    from shamu.security.process_guard import WindowsProcessGuard

    test_file = tmp_path / "executor_test.txt"
    executor = ToolExecutor(process_guard=WindowsProcessGuard(workspace_root=tmp_path))

    # Test write_file
    res_w = await executor.execute("write_file", {"path": str(test_file), "content": "hello executor"})
    assert "File written successfully" in res_w
    assert test_file.read_text(encoding="utf-8") == "hello executor"

    # Test read_file
    res_r = await executor.execute("read_file", {"path": str(test_file)})
    assert res_r == "hello executor"

