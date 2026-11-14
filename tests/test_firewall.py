from pathlib import Path
import pytest
from shamu.security.firewall import SecurityFirewall, PermissionTier, PermissionDecision
from shamu.security.audit import AuditLogger


def test_permission_tier_classification(tmp_path: Path):
    firewall = SecurityFirewall(workspace_root=tmp_path)

    # Tier 1 Safe actions
    d1 = firewall.evaluate_tool_call("read_file", {"path": str(tmp_path / "doc.txt")})
    assert d1.tier == PermissionTier.TIER_1_AUTO
    assert not d1.requires_confirmation
    assert d1.allowed

    d2 = firewall.evaluate_tool_call("get_system_metrics", {})
    assert d2.tier == PermissionTier.TIER_1_AUTO

    # Tier 2 Audited actions
    d3 = firewall.evaluate_tool_call("write_file", {"path": str(tmp_path / "notes.txt"), "content": "hi"})
    assert d3.tier == PermissionTier.TIER_2_AUDITED
    assert not d3.requires_confirmation
    assert d3.allowed

    # Tier 3 Confirmation Mandatory actions
    d4 = firewall.evaluate_tool_call("powershell_exec", {"command": "Get-Process"})
    assert d4.tier == PermissionTier.TIER_3_CONFIRMATION
    assert d4.requires_confirmation

    d5 = firewall.evaluate_tool_call("delete_file", {"path": str(tmp_path / "notes.txt")})
    assert d5.tier == PermissionTier.TIER_3_CONFIRMATION
    assert d5.requires_confirmation


def test_path_traversal_escalation(tmp_path: Path):
    firewall = SecurityFirewall(workspace_root=tmp_path)

    # Attempting to read outside workspace escalates to Tier 3
    d = firewall.evaluate_tool_call("read_file", {"path": "C:\\Windows\\System32\\cmd.exe"})
    assert d.tier == PermissionTier.TIER_3_CONFIRMATION
    assert d.requires_confirmation
    assert "outside approved workspace" in d.reason.lower()


def test_audit_logger(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_file=log_file)

    logger.log_event(
        tool_name="write_file",
        arguments={"path": "test.txt"},
        tier=PermissionTier.TIER_2_AUDITED,
        approval_status="AUTO_APPROVED",
    )

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "write_file" in content
    assert "TIER_2_AUDITED" in content


@pytest.mark.asyncio
async def test_toctou_junction_race_condition_blocked(tmp_path: Path):
    """Regression test: TOCTOU directory junction created during approval must be detected and blocked in Step 5."""
    import sys
    try:
        import _winapi
        has_junction = hasattr(_winapi, "CreateJunction") and sys.platform == "win32"
    except ImportError:
        has_junction = False

    from shamu.protocols import PermissionDeniedError
    from shamu.security.approval_manager import ApprovalManager, ApprovalResponse
    from shamu.security.audit_ledger import AuditLedger
    from shamu.security.firewall import ExecutionFirewall

    ws = tmp_path / "workspace"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    sub = ws / "sub"
    sub.mkdir()

    audit_dir = ws / ".shamu" / "audit"
    cp_dir = ws / ".shamu" / "checkpoints"
    audit_dir.mkdir(parents=True)
    cp_dir.mkdir(parents=True)
    ledger = AuditLedger(
        log_path=audit_dir / "audit.jsonl",
        checkpoint_path=cp_dir / "checkpoints.db",
        master_key=b"F" * 32,
    )

    appr_mgr = ApprovalManager()

    async def attacking_approval_handler(req):
        # Simulate TOCTOU race: During human approval, replace sub with junction pointing to outside
        if has_junction:
            sub.rmdir()
            _winapi.CreateJunction(str(outside), str(sub))
        return ApprovalResponse(request_id=req.request_id, approved=True)

    appr_mgr.register_handler(attacking_approval_handler)

    firewall = ExecutionFirewall(
        workspace_root=ws,
        approval_manager=appr_mgr,
        audit_ledger=ledger,
    )

    target_file = sub / "pwned.txt"
    outside_file = outside / "pwned.txt"

    if has_junction:
        with pytest.raises(PermissionDeniedError, match="outside approved workspace"):
            await firewall.execute_tool(
                tool_name="write_file",
                arguments={"path": str(target_file), "content": "malicious write"},
            )

        # Confirm the file was NEVER written to the outside destination
        assert not outside_file.exists()
    else:
        # Fallback verification: mock path_guard to simulate TOCTOU violation at Step 5
        original_auth = firewall.path_guard.authorize_path
        call_count = 0

        def flipper(path_str, mode="read"):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                from shamu.security.path_guard import PathSecurityViolationError
                raise PathSecurityViolationError("Simulated TOCTOU junction flip to outside root")
            return original_auth(path_str, mode=mode)

        firewall.path_guard.authorize_path = flipper

        with pytest.raises(PermissionDeniedError, match="outside approved workspace"):
            await firewall.execute_tool(
                tool_name="write_file",
                arguments={"path": str(target_file), "content": "malicious write"},
            )
        assert not target_file.exists()

    ledger.close()


def _create_test_firewall(ws: Path, memory_store=None):
    from shamu.security.approval_manager import ApprovalManager, ApprovalResponse
    from shamu.security.audit_ledger import AuditLedger
    from shamu.security.firewall import ExecutionFirewall

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


@pytest.mark.asyncio
async def test_kill_process_protection_and_execution(tmp_path: Path):
    import asyncio
    import os
    import sys
    import subprocess
    import time
    from shamu.protocols import PermissionDeniedError, ToolExecutionError
    from shamu.security.firewall import ProcessProtectionError

    firewall = _create_test_firewall(tmp_path)

    # 1. Protection validation: pid <= 4 must be blocked
    with pytest.raises((PermissionDeniedError, ToolExecutionError)) as exc_info:
        await firewall.execute_tool("kill_process", {"pid": 4})
    assert isinstance(exc_info.value, ProcessProtectionError)

    with pytest.raises((PermissionDeniedError, ToolExecutionError)) as exc_info:
        await firewall.execute_tool("kill_process", {"pid": 0})
    assert isinstance(exc_info.value, ProcessProtectionError)

    # 2. Protection validation: current process and parent process must be blocked
    with pytest.raises((PermissionDeniedError, ToolExecutionError)) as exc_info:
        await firewall.execute_tool("kill_process", {"pid": os.getpid()})
    assert isinstance(exc_info.value, ProcessProtectionError)

    if hasattr(os, "getppid"):
        with pytest.raises((PermissionDeniedError, ToolExecutionError)) as exc_info:
            await firewall.execute_tool("kill_process", {"pid": os.getppid()})
        assert isinstance(exc_info.value, ProcessProtectionError)

    # 3. Graceful handling of non-existent PID
    result_nonexistent = await firewall.execute_tool("kill_process", {"pid": 9999999})
    assert "not found" in result_nonexistent

    # 4. Successful termination of real disposable child process
    sub = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        pid = sub.pid
        result_kill = await asyncio.wait_for(
            firewall.execute_tool("kill_process", {"pid": pid}),
            timeout=5.0,
        )
        assert "terminated successfully" in result_kill
        # Confirm process is actually terminated
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
async def test_focus_application(tmp_path: Path):
    firewall = _create_test_firewall(tmp_path)

    # Test non-existent window title returns informative message without error
    res_missing = await firewall.execute_tool(
        "focus_application",
        {"window_title": "DefinitivelyNonExistentWindow_739105"},
    )
    assert "no window found" in res_missing.lower() or "focus_application is only supported" in res_missing.lower()

    # Test existing window focus if available
    res_existing = await firewall.execute_tool(
        "focus_application",
        {"window_title": "S.H.A.M.U"},
    )
    assert isinstance(res_existing, str)


@pytest.mark.asyncio
async def test_search_local_history(tmp_path: Path):
    # 1. Unconfigured memory store
    fw_no_mem = _create_test_firewall(tmp_path)
    res_no_mem = await fw_no_mem.execute_tool(
        "search_local_history",
        {"query": "architectural design"},
    )
    assert "not configured" in res_no_mem.lower() or "unavailable" in res_no_mem.lower()

    # 2. Configured memory store with search_lexical
    class MockMemoryStore:
        async def search_lexical(self, query: str):
            return [{"id": "mem-1", "content": f"Found history for query: {query}"}]

    fw_mem = _create_test_firewall(tmp_path, memory_store=MockMemoryStore())
    res_mem = await fw_mem.execute_tool(
        "search_local_history",
        {"query": "authentication"},
    )
    assert len(res_mem) == 1
    assert "Found history for query: authentication" in res_mem[0]["content"]


@pytest.mark.asyncio
async def test_async_disk_io_offloading(tmp_path: Path):
    firewall = _create_test_firewall(tmp_path)

    # 1. write_file
    target_file = tmp_path / "io_test.txt"
    res_w = await firewall.execute_tool(
        "write_file",
        {"path": str(target_file), "content": "initial line\n"},
    )
    assert "File written successfully" in res_w
    assert target_file.read_text(encoding="utf-8") == "initial line\n"

    # 2. append_file
    res_a = await firewall.execute_tool(
        "append_file",
        {"path": str(target_file), "content": "appended line\n"},
    )
    assert "Appended successfully" in res_a
    assert target_file.read_text(encoding="utf-8") == "initial line\nappended line\n"

    # 3. read_file
    res_r = await firewall.execute_tool(
        "read_file",
        {"path": str(target_file)},
    )
    assert res_r == "initial line\nappended line\n"

    # 4. list_directory
    res_l = await firewall.execute_tool(
        "list_directory",
        {"path": str(tmp_path)},
    )
    assert "io_test.txt" in res_l

    # 5. delete_file
    res_d = await firewall.execute_tool(
        "delete_file",
        {"path": str(target_file)},
    )
    assert "Deleted file" in res_d
    assert not target_file.exists()

    # 6. delete_file when file does not exist
    res_d2 = await firewall.execute_tool(
        "delete_file",
        {"path": str(target_file)},
    )
    assert "File not found" in res_d2



