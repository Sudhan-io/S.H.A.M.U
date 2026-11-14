"""Windows process execution guard with Win32 Job Object sandboxing and environment sanitization."""

import asyncio
import ctypes
from ctypes import wintypes
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from shamu.protocols import PermissionDeniedError, ShamuError

# WinAPI constants
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JobObjectExtendedLimitInformation = 9
CREATE_SUSPENDED = 0x00000004
TH32CS_SNAPTHREAD = 0x00000004
THREAD_SUSPEND_RESUME = 0x0002

if sys.platform == "win32":
    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", ctypes.c_long),
            ("tpDeltaPri", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
        ]

_PROHIBITED_POWERSHELL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Secondary shells and interpreters
    (
        re.compile(
            r"\b(?:powershell|pwsh|cmd|bash|wsl|sh|zsh|wscript|cscript|mshta|rundll32|regsvr32|certutil|bitsadmin)(?:\.exe)?\b",
            re.IGNORECASE,
        ),
        "secondary shell, interpreter, or binary proxy invocation",
    ),
    # Process / script invokers
    (
        re.compile(
            r"\b(?:Start-Process|saps|start|Invoke-Item|ii|Invoke-Expression|iex|Invoke-Command|icm|Invoke-WmiMethod|Invoke-CimMethod)\b",
            re.IGNORECASE,
        ),
        "process/script invoker or dynamic execution cmdlet",
    ),
    # Script file invocation
    (
        re.compile(
            r"\b[a-zA-Z0-9_\-\.\/\\]+\.(?:ps1|bat|cmd|vbs|js|wsf|hta)\b",
            re.IGNORECASE,
        ),
        "script file execution",
    ),
    # Dynamic invocation operator (&)
    (
        re.compile(r"&", re.IGNORECASE),
        "dynamic call/invocation operator '&'",
    ),
    # Variables, environment variables, and subexpressions ($)
    (
        re.compile(r"\$", re.IGNORECASE),
        "dynamic variable reference or subexpression '$'",
    ),
    # String concatenation operator (+)
    (
        re.compile(r"['\"][^'\"]*['\"]\s*\+|\+\s*['\"]", re.IGNORECASE),
        "dynamic string concatenation expression",
    ),
    # Dot-sourcing operator
    (
        re.compile(
            r"(?:^|[;\s|&])\.\s+[^\s]",
            re.IGNORECASE,
        ),
        "dot-sourcing operator",
    ),
    # Pipeline iteration or member invocation (ForEach-Object / % / foreach)
    (
        re.compile(
            r"\b(?:ForEach-Object|foreach)\b|(?:^|[|;&\s])%(?:\s|$)",
            re.IGNORECASE,
        ),
        "pipeline iteration or member-invocation cmdlet ('ForEach-Object' / '%')",
    ),
    # Filesystem cmdlets (file operations must use dedicated PathGuard tools)
    (
        re.compile(
            r"\b(?:Get-Content|gc|cat|type|Set-Content|sc|Remove-Item|rm|del|erase|ri|Copy-Item|cp|copy|cpi|Move-Item|mv|move|mi|Add-Content|ac|New-Item|ni|Clear-Content|clc|Get-Acl|Set-Acl)\b",
            re.IGNORECASE,
        ),
        "filesystem cmdlet (file operations must use dedicated PathGuard tools)",
    ),
    # Download or web request cradles
    (
        re.compile(
            r"\b(?:Invoke-WebRequest|iwr|curl|wget|Invoke-RestMethod|irm|Start-BitsTransfer|DownloadFile|DownloadString)\b",
            re.IGNORECASE,
        ),
        "download or web request cradle",
    ),
    # Dynamic compilation & COM objects
    (
        re.compile(
            r"\b(?:Add-Type|New-Object\s+-(?:ComObject|com))\b",
            re.IGNORECASE,
        ),
        "dynamic type or COM object creation",
    ),
    # Execution policy tampering
    (
        re.compile(
            r"(?:^|\s)-(?:ExecutionPolicy|ep)\b|\bSet-ExecutionPolicy\b",
            re.IGNORECASE,
        ),
        "execution policy tampering",
    ),
    # Encoded command bypasses
    (
        re.compile(
            r"(?:^|\s)-(?:EncodedCommand|encoded|enc|e)\b",
            re.IGNORECASE,
        ),
        "encoded command parameter",
    ),
    # Reflection, process, and dangerous .NET types
    (
        re.compile(
            r"\[(?:System\.)?(?:Diagnostics\.Process|Reflection\.|Management\.Automation\.)",
            re.IGNORECASE,
        ),
        "restricted .NET type reflection/process invocation",
    ),
    # Direct .NET filesystem types
    (
        re.compile(
            r"\[(?:System\.)?IO\.(?:File|Directory|Path|FileInfo|DirectoryInfo)\]",
            re.IGNORECASE,
        ),
        "direct .NET filesystem type invocation",
    ),
]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class ProcessResult(BaseModel):
    """Result of subprocess execution under guard."""
    exit_code: int = Field(description="Process exit returncode")
    stdout: str = Field(default="", description="Captured standard output")
    stderr: str = Field(default="", description="Captured standard error")
    duration_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    timed_out: bool = Field(default=False, description="Whether execution was terminated due to timeout")
    truncated: bool = Field(default=False, description="Whether output exceeded byte cap and was truncated")


class WindowsProcessGuard:
    """Subprocess execution controller enforcing allowlists, environment sanitization, and Win32 Job Object sandboxing."""

    DEFAULT_ALLOWLIST: Set[str] = {
        "git", "git.exe",
        "python", "python.exe",
        "uv", "uv.exe",
        "powershell", "powershell.exe",
    }

    DANGEROUS_ENV_VARS: Set[str] = {
        "PYTHONPATH", "PYTHONHOME", "NODE_OPTIONS", "PERLLIB", "PERL5LIB", "RUBYLIB",
    }

    SECRET_ENV_KEYS: Set[str] = {
        "KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "CREDENTIAL",
    }

    SAFE_ENV_VARS: Set[str] = {
        "SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATH", "PATHEXT",
        "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
        "WINDIR", "PROGRAMFILES", "PROGRAMFILES(X86)", "COMMONPROGRAMFILES",
    }

    MAX_OUTPUT_BYTES: int = 64 * 1024  # 64 KB per stream

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        allowlist: Optional[Set[str]] = None,
        trusted_dirs: Optional[List[Path]] = None,
    ) -> None:
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.allowlist = set(allowlist) if allowlist is not None else set(self.DEFAULT_ALLOWLIST)
        # Always allow current active python runtime executable
        current_py_name = Path(sys.executable).name.lower()
        self.allowlist.add(current_py_name)
        self.allowlist.add(str(Path(sys.executable).resolve()))
        self.custom_trusted_dirs = [Path(d).resolve() for d in (trusted_dirs or [])]

    def get_trusted_directories(self) -> List[Path]:
        """Returns the canonical root directories trusted for executable binaries."""
        trusted: List[Path] = []
        if sys.platform == "win32":
            for env_var in ("SystemRoot", "WINDIR"):
                val = os.environ.get(env_var)
                if val:
                    trusted.append(Path(val).resolve())
            for env_var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
                val = os.environ.get(env_var)
                if val:
                    trusted.append(Path(val).resolve())

            # Python Scripts and binary locations
            trusted.append((Path(sys.prefix).resolve() / "Scripts"))
            trusted.append((Path(sys.base_prefix).resolve() / "Scripts"))

            # User-level trusted binary directories
            local_app = os.environ.get("LOCALAPPDATA")
            if local_app:
                trusted.append((Path(local_app) / "Programs").resolve())
            user_prof = os.environ.get("USERPROFILE")
            if user_prof:
                trusted.append((Path(user_prof) / ".cargo" / "bin").resolve())
        else:
            trusted.extend([
                Path("/bin").resolve(),
                Path("/usr/bin").resolve(),
                Path("/usr/local/bin").resolve(),
                Path(sys.prefix).resolve() / "bin",
                Path(sys.base_prefix).resolve() / "bin",
            ])

        # Active python executable directory
        trusted.append(Path(sys.executable).resolve().parent)
        trusted.extend(self.custom_trusted_dirs)

        return [d for d in trusted if d.exists()]

    def _find_executable_in_trusted_path(self, command_name: str) -> Optional[Path]:
        """Resolves a bare command name across system PATH directories, strictly excluding cwd and untrusted locations."""
        pathext_source = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
        pathext = [ext.lower() for ext in pathext_source.split(os.pathsep) if ext]

        name_ext = os.path.splitext(command_name)[1].lower()
        if name_ext in pathext:
            candidates = [command_name]
        else:
            candidates = [command_name + ext for ext in pathext] + [command_name]

        raw_path = os.environ.get("PATH", "")
        path_dirs = [Path(p) for p in raw_path.split(os.pathsep) if p.strip()]

        excluded_dirs = {
            Path.cwd().resolve(),
            self.workspace_root.resolve(),
            Path(tempfile.gettempdir()).resolve(),
        }

        trusted_dirs = self.get_trusted_directories()

        for p_dir in path_dirs:
            try:
                resolved_dir = p_dir.resolve()
            except Exception:
                continue

            # Never search directories inside excluded locations
            if any(resolved_dir == exc or resolved_dir.is_relative_to(exc) for exc in excluded_dirs):
                continue

            # Must be inside a trusted directory
            if not any(resolved_dir == td or resolved_dir.is_relative_to(td) for td in trusted_dirs):
                continue

            for cand in candidates:
                target = resolved_dir / cand
                try:
                    if target.is_file():
                        return target.resolve()
                except OSError:
                    continue

        return None

    def resolve_and_validate_executable(self, executable: str, cwd: Optional[Path] = None) -> Path:
        """Validates and canonicalizes executable against allowlist and trusted directories (SEC-03 remediation)."""
        exe_path = Path(executable)
        exe_name = exe_path.name.lower()

        # 1. Allowlist verification
        active_py = Path(sys.executable).resolve()
        is_allowed = (
            exe_name in self.allowlist
            or executable in self.allowlist
            or executable.lower() in self.allowlist
        )

        try:
            if not is_allowed and exe_path.resolve() == active_py:
                is_allowed = True
        except Exception:
            pass

        if not is_allowed:
            raise PermissionDeniedError(f"Executable '{executable}' is not in the approved allowlist.")

        # 2. Canonical path resolution
        has_dir_part = (os.path.sep in executable) or (bool(os.path.altsep) and os.path.altsep in executable)
        if has_dir_part:
            effective_cwd = Path(cwd or self.workspace_root).resolve()
            cand_path = (effective_cwd / exe_path).resolve() if not exe_path.is_absolute() else exe_path.resolve()
            if not cand_path.is_file():
                raise PermissionDeniedError(f"Executable '{executable}' not found or is not a regular file.")
        else:
            # Bare command name
            if exe_name in ("python.exe", "python"):
                cand_path = active_py
            else:
                found = self._find_executable_in_trusted_path(executable)
                if not found:
                    raise PermissionDeniedError(f"Executable '{executable}' could not be located in trusted PATH.")
                cand_path = found

        # 3. Trusted location enforcement
        # Active python executable is always trusted
        if cand_path == active_py:
            return cand_path

        # Explicitly authorized canonical paths in allowlist
        for item in self.allowlist:
            if (os.path.sep in item) or (bool(os.path.altsep) and os.path.altsep in item):
                try:
                    if Path(item).resolve() == cand_path:
                        return cand_path
                except Exception:
                    pass

        # Explicitly authorized custom trusted directories
        if any(cand_path == cd or cand_path.is_relative_to(cd) for cd in self.custom_trusted_dirs):
            return cand_path

        # Check against system trusted directories
        trusted_dirs = self.get_trusted_directories()
        in_trusted_dir = any(cand_path == td or cand_path.is_relative_to(td) for td in trusted_dirs)

        # Check untrusted locations: temporary dirs and workspace root
        temp_dir = Path(tempfile.gettempdir()).resolve()
        in_temp = cand_path == temp_dir or cand_path.is_relative_to(temp_dir)
        in_workspace = (cand_path == self.workspace_root or cand_path.is_relative_to(self.workspace_root)) and cand_path != active_py

        if not in_trusted_dir or in_temp or in_workspace:
            raise PermissionDeniedError(
                f"Executable '{executable}' resolves to untrusted path '{cand_path}'. Path substitution is prohibited."
            )

        return cand_path

    def _create_job_object(self):
        """Creates and configures a Windows Job Object with KILL_ON_JOB_CLOSE and 2 GB memory ceiling."""
        if sys.platform != "win32":
            return None

        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return None

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_JOB_MEMORY
            info.JobMemoryLimit = 2 * 1024 * 1024 * 1024  # 2 GB limit

            res = kernel32.SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not res:
                kernel32.CloseHandle(job)
                return None
            return job
        except Exception:
            return None

    def _assign_process_to_job(self, job_handle, pid: int) -> bool:
        """Assigns child process PID to the active Job Object."""
        if sys.platform != "win32" or not job_handle:
            return False

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h_proc = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h_proc:
            return False

        try:
            assigned = kernel32.AssignProcessToJobObject(job_handle, h_proc)
            return bool(assigned)
        finally:
            kernel32.CloseHandle(h_proc)

    def _resume_process_threads(self, pid: int) -> bool:
        """Resumes threads of a child process created in a suspended state."""
        if sys.platform != "win32":
            return True

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        if snap == -1 or snap == 0xFFFFFFFFFFFFFFFF:
            return False

        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(THREADENTRY32)
        success = kernel32.Thread32First(snap, ctypes.byref(entry))
        resumed = 0
        while success:
            if entry.th32OwnerProcessID == pid:
                h_thread = kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                if h_thread:
                    kernel32.ResumeThread(h_thread)
                    kernel32.CloseHandle(h_thread)
                    resumed += 1
            success = kernel32.Thread32Next(snap, ctypes.byref(entry))
        kernel32.CloseHandle(snap)
        return resumed > 0

    def build_sanitized_env(self, custom_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Constructs a clean environment without ambient secrets or execution hijack variables."""
        clean_env: Dict[str, str] = {}

        for k, v in os.environ.items():
            k_upper = k.upper()
            if k_upper in self.SAFE_ENV_VARS:
                # Retain safe system variables unless they match a secret pattern
                if not any(sec in k_upper for sec in self.SECRET_ENV_KEYS):
                    clean_env[k] = v

        # Filter custom environment if provided
        if custom_env:
            for k, v in custom_env.items():
                k_upper = k.upper()
                if k_upper not in self.DANGEROUS_ENV_VARS and not any(sec in k_upper for sec in self.SECRET_ENV_KEYS):
                    clean_env[k] = v

        return clean_env

    async def run_command(
        self,
        executable: str,
        args: List[str],
        cwd: Optional[Path] = None,
        timeout_sec: float = 30.0,
        env_override: Optional[Dict[str, str]] = None,
        stdin_input: Optional[str] = None,
    ) -> ProcessResult:
        """Executes a command under strict executable allowlist, canonical path validation, Job Object sandboxing, and output capping."""
        # 1. Canonical executable validation & resolution (SEC-03 remediation)
        validated_exe = self.resolve_and_validate_executable(executable, cwd=cwd)

        work_dir = str(cwd or self.workspace_root)
        clean_env = self.build_sanitized_env(env_override)

        # Initialize Windows Job Object
        job_handle = self._create_job_object()
        if sys.platform == "win32" and not job_handle:
            raise ShamuError("Failed to create or assign Windows Job Object: could not initialize job sandbox.")

        creationflags = CREATE_SUSPENDED if (sys.platform == "win32" and job_handle) else 0
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if sys.platform == "win32" else None

        t0 = time.perf_counter()
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                str(validated_exe),
                *args,
                cwd=work_dir,
                env=clean_env,
                stdin=asyncio.subprocess.PIPE if stdin_input is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                close_fds=True,
                creationflags=creationflags,
            )

            # Assign to Job Object while process is suspended, eliminating the race condition (SEC-04)
            if sys.platform == "win32" and job_handle:
                assigned = False
                try:
                    sub = proc._transport.get_extra_info("subprocess")
                    if sub and hasattr(sub, "_handle") and sub._handle and kernel32:
                        assigned = bool(kernel32.AssignProcessToJobObject(job_handle, int(sub._handle)))
                except Exception:
                    assigned = False

                if not assigned:
                    assigned = self._assign_process_to_job(job_handle, proc.pid)

                if not assigned:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                    except Exception:
                        pass
                    if hasattr(proc, "_transport") and proc._transport:
                        try:
                            proc._transport.close()
                        except Exception:
                            pass
                    await asyncio.sleep(0.01)
                    raise ShamuError("Failed to create or assign Windows Job Object: could not assign process.")

                # Resume the suspended process now that it is securely bounded by the Job Object
                if not self._resume_process_threads(proc.pid):
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                    except Exception:
                        pass
                    if hasattr(proc, "_transport") and proc._transport:
                        try:
                            proc._transport.close()
                        except Exception:
                            pass
                    await asyncio.sleep(0.01)
                    raise ShamuError("Failed to resume suspended child process after Job Object assignment.")

            input_bytes = stdin_input.encode("utf-8") if stdin_input is not None else None
            try:
                raw_stdout, raw_stderr = await asyncio.wait_for(
                    proc.communicate(input=input_bytes),
                    timeout=timeout_sec,
                )
                duration_ms = (time.perf_counter() - t0) * 1000.0
                timed_out = False
                await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                duration_ms = (time.perf_counter() - t0) * 1000.0
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except Exception:
                    pass
                if hasattr(proc, "_transport") and proc._transport:
                    try:
                        proc._transport.close()
                    except Exception:
                        pass
                await asyncio.sleep(0.01)
                return ProcessResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command execution timed out after {timeout_sec:.1f} seconds.",
                    duration_ms=duration_ms,
                    timed_out=True,
                )

            # Enforce output stream byte ceiling
            truncated = False
            if len(raw_stdout) > self.MAX_OUTPUT_BYTES:
                raw_stdout = raw_stdout[:self.MAX_OUTPUT_BYTES] + b"\n[Output truncated at 64 KB]"
                truncated = True
            if len(raw_stderr) > self.MAX_OUTPUT_BYTES:
                raw_stderr = raw_stderr[:self.MAX_OUTPUT_BYTES] + b"\n[Output truncated at 64 KB]"
                truncated = True

            out_text = raw_stdout.decode("utf-8", errors="replace")
            err_text = raw_stderr.decode("utf-8", errors="replace")

            try:
                await proc.wait()
            except Exception:
                pass

            return ProcessResult(
                exit_code=proc.returncode if proc.returncode is not None else 0,
                stdout=out_text,
                stderr=err_text,
                duration_ms=duration_ms,
                timed_out=timed_out,
                truncated=truncated,
            )
        finally:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except Exception:
                    pass
                if hasattr(proc, "_transport") and proc._transport:
                    try:
                        proc._transport.close()
                    except Exception:
                        pass
                await asyncio.sleep(0.01)
            if kernel32 and job_handle:
                kernel32.CloseHandle(job_handle)

    def validate_powershell_command(self, command: str, cwd: Optional[Path] = None) -> None:
        """Validates a PowerShell command against prohibited constructs and workspace boundary escapes."""
        if not command or not command.strip():
            raise PermissionDeniedError("PowerShell command cannot be empty.")

        # 1. Lexical checks for prohibited commands, cmdlets, and bypass parameters
        for pattern, reason in _PROHIBITED_POWERSHELL_PATTERNS:
            if pattern.search(command):
                raise PermissionDeniedError(
                    f"PowerShell command contains prohibited construct ({reason}): {command.strip()}"
                )

        # 2. Check for absolute paths, root-relative paths, and relative directory traversal outside the workspace
        # Matches drive-letter paths (e.g. C:\... or C:/...), UNC paths (\\...), root-relative paths (\Windows\System32... or /Windows/...), or parent traversal (../ or ..\)
        base_dir = (cwd or self.workspace_root).resolve()
        path_matches = re.finditer(
            r"(?:[A-Za-z]:[\\/][^\s'\";|&]+|[\\/][A-Za-z0-9_.\-]+[\\/][^\s'\";|&]*|\\\\[^\s'\";|&]+|\.\.[\\/][^\s'\";|&]*|\\[A-Za-z0-9_.\-]+)",
            command,
        )
        for m in path_matches:
            raw_path = m.group(0).rstrip("'\")}]")
            try:
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = (base_dir / candidate).resolve()
                else:
                    candidate = candidate.resolve()

                if not candidate.is_relative_to(self.workspace_root):
                    raise PermissionDeniedError(
                        f"PowerShell command references path outside workspace: '{raw_path}'"
                    )
            except PermissionDeniedError:
                raise
            except Exception:
                # If path resolution fails, reject defensively
                raise PermissionDeniedError(
                    f"PowerShell command contains invalid or unresolvable path: '{raw_path}'"
                )

    async def run_powershell(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout_sec: float = 30.0,
    ) -> ProcessResult:
        """Executes a PowerShell command using structured arguments under AST allowlist and strict pre-execution validation."""
        # 1. Pre-execution security validation
        self.validate_powershell_command(command, cwd=cwd)

        # 2. In-Engine AST Verification and ConstrainedLanguage Execution via stdin
        ast_wrapper = (
            "$ErrorActionPreference = 'Stop'; "
            "$cmd = [Console]::In.ReadToEnd(); "
            "$tokens = $null; $errors = $null; "
            "$ast = [System.Management.Automation.Language.Parser]::ParseInput($cmd, [ref]$tokens, [ref]$errors); "
            "if ($errors.Count -gt 0) { throw ('Syntax error in PowerShell command: ' + $errors[0].Message) }; "
            "$allowedNodes = @('ScriptBlockAst','NamedBlockAst','PipelineAst','CommandAst','CommandExpressionAst','StringConstantExpressionAst',"
            "'CommandParameterAst','ConstantExpressionAst','ArrayLiteralAst','BinaryExpressionAst','ParenExpressionAst'); "
            "$allowedCmds = @('Get-Process','Get-Service','Get-Date','Get-Host','Write-Output','Write-Host',"
            "'Select-Object','Sort-Object','Where-Object','Measure-Object','Out-String'); "
            "foreach ($n in $ast.FindAll({ $true }, $true)) { "
            "  if ($allowedNodes -notcontains $n.GetType().Name) { throw ('AST node violation: ' + $n.GetType().Name) }; "
            "  if ($n -is [System.Management.Automation.Language.CommandAst]) { "
            "    $cName = $n.GetCommandName(); "
            "    if (-not $cName -or $allowedCmds -notcontains $cName) { throw ('Command not allowed: ' + $cName) }; "
            "  }; "
            "}; "
            "$sb = $ast.GetScriptBlock(); "
            "$ExecutionContext.SessionState.LanguageMode = [System.Management.Automation.PSLanguageMode]::ConstrainedLanguage; "
            "& $sb"
        )

        ps_args = [
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy", "Restricted",
            "-Command", ast_wrapper,
        ]
        return await self.run_command(
            executable="powershell.exe",
            args=ps_args,
            cwd=cwd,
            timeout_sec=timeout_sec,
            stdin_input=command,
        )
