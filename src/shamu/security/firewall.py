"""Execution Security Firewall: Central authorization gateway orchestrating path validation, tiered approval, process sandboxing, and tamper-evident audit logging."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set
from pydantic import BaseModel, Field

from shamu.protocols import PermissionDeniedError, ToolExecutionError
from shamu.types import PermissionTier, PolicyOutcome
from shamu.security.approval_manager import ApprovalManager
from shamu.security.audit_ledger import AuditIntegrityError, AuditLedger
from shamu.security.path_guard import PathSecurityViolationError, WindowsPathGuard
from shamu.security.process_guard import WindowsProcessGuard
from shamu.security.tool_registry import (
    ToolCapability,
    ToolRegistry,
)

RegistryPermissionTier = PermissionTier


class ProcessProtectionError(PermissionDeniedError, ToolExecutionError):
    """Raised when an attempt is made to kill a protected system or current process."""

    def __init__(
        self,
        pid: int,
        process_name: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        self.pid = pid
        self.process_name = process_name
        self.reason = reason

        details = f"Cannot terminate protected system or current process (PID: {pid}"
        if process_name:
            details += f", name: {process_name}"
        details += ")."
        if reason:
            details += f" Reason: {reason}"

        ToolExecutionError.__init__(
            self,
            "kill_process",
            details,
        )
        PermissionDeniedError.__init__(
            self,
            details,
        )


CRITICAL_SYSTEM_PROCESSES: Set[str] = {
    # System Architecture & Kernel Session Managers
    "system",
    "system idle process",
    "registry",
    "smss.exe",
    "csrss.exe",
    "wininit.exe",
    "services.exe",
    "lsass.exe",
    "winlogon.exe",
    "dwm.exe",
    # Service Host & Font Driver
    "svchost.exe",
    "fontdrvhost.exe",
    # Shell & Security Isolation
    "sihost.exe",
    "lsaiso.exe",
    "securityhealthservice.exe",
    "explorer.exe",
    "explorer",
    # Memory Compression
    "memory compression",
}


def is_critical_process_name(name: Optional[str]) -> bool:
    """Checks whether a process name matches any protected critical Windows system binary."""
    if not name or not isinstance(name, str):
        return False
    clean = name.strip().lower()
    if clean in CRITICAL_SYSTEM_PROCESSES:
        return True
    if clean.endswith(".exe"):
        if clean[:-4] in CRITICAL_SYSTEM_PROCESSES:
            return True
    else:
        if f"{clean}.exe" in CRITICAL_SYSTEM_PROCESSES:
            return True
    return False


def validate_process_target(pid: int) -> None:
    """Validates that a PID does not correspond to a protected system process,
    the current process, or the parent terminal process.

    Enforces fail-closed semantics: if process identity cannot be verified
    or access is denied, ProcessProtectionError is raised.
    """
    if pid <= 4 or pid == os.getpid() or (hasattr(os, "getppid") and pid == os.getppid()):
        raise ProcessProtectionError(
            pid,
            reason="Cannot terminate core system (PID <= 4), current process, or parent terminal.",
        )

    import psutil

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Non-existent process is handled at execution time (returns not found)
        return
    except psutil.AccessDenied as exc:
        raise ProcessProtectionError(
            pid,
            reason=f"Access denied inspecting process identity (fail-closed): {exc}",
        ) from exc
    except Exception as exc:
        raise ProcessProtectionError(
            pid,
            reason=f"Unable to inspect process (fail-closed): {exc}",
        ) from exc

    try:
        if not proc.is_running():
            return
        name = proc.name()
    except psutil.NoSuchProcess:
        return
    except psutil.AccessDenied as exc:
        raise ProcessProtectionError(
            pid,
            reason=f"Access denied inspecting process name (fail-closed): {exc}",
        ) from exc
    except Exception as exc:
        raise ProcessProtectionError(
            pid,
            reason=f"Failed to inspect process name (fail-closed): {exc}",
        ) from exc

    if not name or not isinstance(name, str):
        raise ProcessProtectionError(
            pid,
            reason="Process identity could not be verified: empty process name (fail-closed).",
        )

    if is_critical_process_name(name):
        raise ProcessProtectionError(
            pid,
            process_name=name,
            reason=f"Process '{name}' is a protected Windows critical system process.",
        )

    try:
        exe_path = proc.exe()
        if exe_path and is_critical_process_name(Path(exe_path).name):
            raise ProcessProtectionError(
                pid,
                process_name=Path(exe_path).name,
                reason=f"Process binary '{Path(exe_path).name}' is a protected Windows critical system process.",
            )
    except psutil.NoSuchProcess:
        return
    except (psutil.AccessDenied, Exception):
        pass


class PermissionDecision(BaseModel):
    """Result of legacy security policy evaluation."""
    allowed: bool
    tier: PermissionTier
    requires_confirmation: bool
    reason: str


class AuthorizedToolCall(BaseModel):
    """Specification of an authorized tool call ready for execution dispatch."""
    tool_name: str
    arguments: Dict[str, Any]
    capability: ToolCapability
    tier: PermissionTier
    decision: str = "ALLOW"
    approval_id: Optional[str] = None
    raw_paths: Dict[str, str] = Field(default_factory=dict)


TIER_1_TOOLS: Set[str] = {
    "read_file",
    "list_directory",
    "get_system_metrics",
    "search_local_history",
}

TIER_2_TOOLS: Set[str] = {
    "write_file",
    "append_file",
    "focus_application",
}

TIER_3_TOOLS: Set[str] = {
    "powershell_exec",
    "delete_file",
    "kill_process",
}


class PolicyAuthorizer:
    """Central authorization engine separating security policy authorization from concrete execution."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        path_guard: Optional[WindowsPathGuard] = None,
        process_guard: Optional[WindowsProcessGuard] = None,
        approval_manager: Optional[ApprovalManager] = None,
        audit_ledger: Optional[AuditLedger] = None,
        workspace_root: Optional[Path] = None,
        firewall: Optional["ExecutionFirewall"] = None,
    ) -> None:
        self._firewall = firewall
        self._registry = registry
        self._path_guard = path_guard
        self._process_guard = process_guard
        self._approval_manager = approval_manager
        self._audit_ledger = audit_ledger
        self._workspace_root = workspace_root

    @property
    def registry(self) -> ToolRegistry:
        if self._firewall is not None:
            return self._firewall.registry
        if self._registry is None:
            self._registry = ToolRegistry.create_default()
        return self._registry

    @property
    def path_guard(self) -> WindowsPathGuard:
        if self._firewall is not None:
            return self._firewall.path_guard
        if self._path_guard is None:
            self._path_guard = WindowsPathGuard(workspace_root=self.workspace_root)
        return self._path_guard

    @property
    def process_guard(self) -> WindowsProcessGuard:
        if self._firewall is not None:
            return self._firewall.process_guard
        if self._process_guard is None:
            self._process_guard = WindowsProcessGuard(workspace_root=self.workspace_root)
        return self._process_guard

    @property
    def approval_manager(self) -> ApprovalManager:
        if self._firewall is not None:
            return self._firewall.approval_manager
        if self._approval_manager is None:
            self._approval_manager = ApprovalManager()
        return self._approval_manager

    @property
    def audit_ledger(self) -> AuditLedger:
        if self._firewall is not None:
            return self._firewall.audit_ledger
        if self._audit_ledger is None:
            audit_dir = self.workspace_root / ".shamu" / "audit"
            cp_dir = self.workspace_root / ".shamu" / "checkpoints"
            audit_dir.mkdir(parents=True, exist_ok=True)
            cp_dir.mkdir(parents=True, exist_ok=True)
            self._audit_ledger = AuditLedger(
                log_path=audit_dir / "audit.jsonl",
                checkpoint_path=cp_dir / "checkpoints.db",
            )
        return self._audit_ledger

    @property
    def workspace_root(self) -> Path:
        if self._firewall is not None:
            return self._firewall.workspace_root
        if self._workspace_root is None:
            self._workspace_root = Path.cwd().resolve()
        return self._workspace_root

    def evaluate_policy(self, tool_name: str, arguments: Dict[str, Any]) -> PermissionDecision:
        """Evaluates tool name and arguments against security policies (backward compatibility)."""
        # 1. Prohibited tool check
        if tool_name.startswith("__") or tool_name in ("format_disk", "reg_delete", "disable_firewall", "elevate_uac"):
            return PermissionDecision(
                allowed=False,
                tier=PermissionTier.PROHIBITED,
                requires_confirmation=False,
                reason=f"Action '{tool_name}' is permanently prohibited by security policy.",
            )

        cap = self.registry.get(tool_name)
        if cap and (cap.is_prohibited or cap.tier.canonical() == PermissionTier.PROHIBITED):
            return PermissionDecision(
                allowed=False,
                tier=PermissionTier.PROHIBITED,
                requires_confirmation=False,
                reason=f"Action '{tool_name}' is permanently prohibited by security policy.",
            )

        # Process protection check for kill_process
        if tool_name == "kill_process" and "pid" in arguments:
            try:
                pid_val = int(arguments["pid"])
                validate_process_target(pid_val)
            except ProcessProtectionError as exc:
                return PermissionDecision(
                    allowed=False,
                    tier=PermissionTier.PROHIBITED,
                    requires_confirmation=False,
                    reason=str(exc),
                )
            except (ValueError, TypeError):
                pass

        # 2. Check path containment for filesystem tools
        path_keys = cap.path_arg_names if (cap and cap.path_arg_names) else (["path"] if "path" in arguments else [])
        for k in path_keys:
            if k in arguments and arguments[k] is not None:
                path_val = str(arguments[k])
                if not self.path_guard.is_safe_path(path_val, mode="read"):
                    return PermissionDecision(
                        allowed=True,
                        tier=PermissionTier.TIER_3_CONFIRMATION,
                        requires_confirmation=True,
                        reason=f"Target path '{path_val}' is outside approved workspace ({self.workspace_root}).",
                    )

        # 3. Static tier evaluation
        if cap is not None:
            c_tier = cap.tier.canonical()
            if c_tier == PermissionTier.TIER_1_READ_ONLY:
                return PermissionDecision(
                    allowed=True,
                    tier=PermissionTier.TIER_1_AUTO,
                    requires_confirmation=False,
                    reason="Safe read-only operation.",
                )
            if c_tier == PermissionTier.TIER_2_MUTATION:
                return PermissionDecision(
                    allowed=True,
                    tier=PermissionTier.TIER_2_AUDITED,
                    requires_confirmation=False,
                    reason="Low-risk reversible workspace operation.",
                )
            return PermissionDecision(
                allowed=True,
                tier=PermissionTier.TIER_3_CONFIRMATION,
                requires_confirmation=True,
                reason=f"Action '{tool_name}' carries high-risk or external system impact.",
            )

        if tool_name in TIER_1_TOOLS:
            return PermissionDecision(
                allowed=True,
                tier=PermissionTier.TIER_1_AUTO,
                requires_confirmation=False,
                reason="Safe read-only operation.",
            )

        if tool_name in TIER_2_TOOLS:
            return PermissionDecision(
                allowed=True,
                tier=PermissionTier.TIER_2_AUDITED,
                requires_confirmation=False,
                reason="Low-risk reversible workspace operation.",
            )

        # 4. High-risk or unknown tools default to Tier 3
        return PermissionDecision(
            allowed=True,
            tier=PermissionTier.TIER_3_CONFIRMATION,
            requires_confirmation=True,
            reason=f"Action '{tool_name}' carries high-risk or external system impact.",
        )

    async def authorize_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: str = "default",
        turn_id: str = "turn-0",
    ) -> AuthorizedToolCall:
        """Central authorization policy evaluation enforcing registry lookup, schema validation, path guard, process guard, and tiered human approval."""
        # 1. Fail-closed audit ledger state check
        if self.audit_ledger.is_locked:
            raise AuditIntegrityError("Execution firewall is locked: audit ledger integrity violation detected.")

        # 2. Capability Lookup & Prohibited Check
        cap = self.registry.get(tool_name)
        if cap is None:
            self.audit_ledger.append_event(
                session_id=session_id,
                turn_id=turn_id,
                tool_name=tool_name,
                arguments=arguments,
                tier="UNKNOWN",
                decision="DENY",
                outcome="DENIED",
                error=f"Unregistered tool: {tool_name}",
            )
            raise PermissionDeniedError(f"Tool '{tool_name}' is not registered.")

        if cap.is_prohibited or cap.tier.canonical() == PermissionTier.PROHIBITED:
            self.audit_ledger.append_event(
                session_id=session_id,
                turn_id=turn_id,
                tool_name=tool_name,
                arguments=arguments,
                tier=cap.tier,
                decision="DENY",
                outcome="DENIED",
                error=f"Permanently prohibited tool: {tool_name}",
            )
            raise PermissionDeniedError(f"Action '{tool_name}' is permanently prohibited by security policy.")

        # Validate arguments schema
        valid, err_msg = self.registry.validate_call(tool_name, arguments)
        if not valid:
            self.audit_ledger.append_event(
                session_id=session_id,
                turn_id=turn_id,
                tool_name=tool_name,
                arguments=arguments,
                tier=cap.tier,
                decision="DENY",
                outcome="DENIED",
                error=f"Schema validation error: {err_msg}",
            )
            raise PermissionDeniedError(f"Argument validation failed for tool '{tool_name}': {err_msg}")

        # 3. Path & Process Security Authorization
        c_tier = cap.tier.canonical()
        mode = "read" if c_tier == PermissionTier.TIER_1_READ_ONLY else "write"
        path_keys = cap.path_arg_names or (["path"] if "path" in arguments else [])
        raw_paths = {k: str(arguments[k]) for k in path_keys if k in arguments and arguments[k] is not None}
        for k in path_keys:
            if k in arguments and arguments[k] is not None:
                try:
                    auth_path = self.path_guard.authorize_path(str(arguments[k]), mode=mode)
                    arguments[k] = str(auth_path)
                except PathSecurityViolationError as exc:
                    self.audit_ledger.append_event(
                        session_id=session_id,
                        turn_id=turn_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        tier=cap.tier,
                        decision="DENY",
                        outcome="DENIED",
                        error=str(exc),
                    )
                    raise PermissionDeniedError(
                        f"Target path is outside approved workspace: {exc}"
                    ) from exc

        if tool_name == "kill_process" and "pid" in arguments:
            try:
                pid_val = int(arguments["pid"])
                await asyncio.to_thread(validate_process_target, pid_val)
            except ProcessProtectionError as exc:
                self.audit_ledger.append_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    tier=cap.tier,
                    decision="DENY",
                    outcome="DENIED",
                    error=str(exc),
                )
                raise

        # 4. Tiered Approval Engine
        approval_id: Optional[str] = None
        decision = "ALLOW"
        if c_tier in (PermissionTier.TIER_2_MUTATION, PermissionTier.TIER_3_HIGH_IMPACT):
            decision = "REQUIRE_APPROVAL"
            try:
                token = await self.approval_manager.request_approval(
                    tool_name=tool_name,
                    arguments=arguments,
                    tier=cap.tier,
                    reason=cap.description,
                    requires_double_confirmation=cap.requires_double_confirmation,
                )
                self.approval_manager.verify_and_redeem_token(token, tool_name, arguments)
                approval_id = token.request_id
            except Exception as exc:
                self.audit_ledger.append_event(
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    tier=cap.tier,
                    decision=decision,
                    outcome="DENIED",
                    error=str(exc),
                )
                raise

        return AuthorizedToolCall(
            tool_name=tool_name,
            arguments=arguments,
            capability=cap,
            tier=cap.tier,
            decision=decision,
            approval_id=approval_id,
            raw_paths=raw_paths,
        )


class ToolExecutor:
    """Dispatches concrete tool operations against the filesystem, OS processes, and memory store."""

    def __init__(
        self,
        process_guard: Optional[WindowsProcessGuard] = None,
        memory_store: Optional[Any] = None,
        custom_executors: Optional[Dict[str, Callable[[Dict[str, Any]], Any]]] = None,
        firewall: Optional["ExecutionFirewall"] = None,
    ) -> None:
        self._firewall = firewall
        self._process_guard = process_guard
        self._memory_store = memory_store
        self._custom_executors = custom_executors if custom_executors is not None else {}

    @property
    def process_guard(self) -> WindowsProcessGuard:
        if self._firewall is not None:
            return self._firewall.process_guard
        if self._process_guard is None:
            self._process_guard = WindowsProcessGuard()
        return self._process_guard

    @property
    def memory_store(self) -> Optional[Any]:
        if self._memory_store is not None:
            return self._memory_store
        if self._firewall is not None:
            return self._firewall.memory_store
        return None

    @memory_store.setter
    def memory_store(self, val: Optional[Any]) -> None:
        self._memory_store = val

    @property
    def custom_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        if self._firewall is not None:
            return self._firewall._custom_executors
        return self._custom_executors

    def register_custom_executor(self, tool_name: str, fn: Callable[[Dict[str, Any]], Any]) -> None:
        self.custom_executors[tool_name] = fn

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if tool_name in self.custom_executors:
            res = self.custom_executors[tool_name](arguments)
            if hasattr(res, "__await__"):
                return await res
            return res

        if tool_name == "read_file":
            p = Path(arguments["path"])
            stat = await asyncio.to_thread(p.stat)
            if stat.st_size > 10 * 1024 * 1024:
                raise ToolExecutionError(tool_name, "File size exceeds 10 MB limit.")
            return await asyncio.to_thread(p.read_text, encoding="utf-8", errors="replace")

        if tool_name == "write_file":
            p = Path(arguments["path"])
            content = arguments["content"]

            def _write():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")

            await asyncio.to_thread(_write)
            return f"File written successfully: {p}"

        if tool_name == "append_file":
            p = Path(arguments["path"])
            content = arguments["content"]

            def _append():
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a", encoding="utf-8") as f:
                    f.write(content)

            await asyncio.to_thread(_append)
            return f"Appended successfully to: {p}"

        if tool_name == "list_directory":
            target_dir = Path(arguments.get("path", "."))

            def _list():
                return [f.name for f in target_dir.iterdir()][:1000]

            return await asyncio.to_thread(_list)

        if tool_name == "get_system_metrics":
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "ram_percent": psutil.virtual_memory().percent,
            }

        if tool_name == "search_local_history":
            query = arguments["query"]
            store = self.memory_store
            if store is None:
                return "Local memory store is not configured or unavailable."

            if hasattr(store, "search_lexical"):
                search_fn = store.search_lexical
            elif hasattr(store, "search_memories"):
                search_fn = store.search_memories
            elif hasattr(store, "search"):
                search_fn = store.search
            else:
                search_fn = None

            if search_fn is not None:
                try:
                    res = search_fn(query)
                    matches = await res if hasattr(res, "__await__") else res
                    return matches
                except Exception as exc:
                    return f"Error searching local history: {exc}"
            return "Local memory store does not support search."

        if tool_name == "focus_application":
            window_title = arguments["window_title"]
            return await asyncio.to_thread(ExecutionFirewall._focus_window, window_title)

        if tool_name == "powershell_exec":
            cmd_str = arguments["command"]
            return await self.process_guard.run_powershell(cmd_str)

        if tool_name == "delete_file":
            p = Path(arguments["path"])

            def _delete():
                if p.exists():
                    p.unlink()
                    return True
                return False

            deleted = await asyncio.to_thread(_delete)
            if deleted:
                return f"Deleted file: {p}"
            return f"File not found: {p}"

        if tool_name == "kill_process":
            pid = int(arguments["pid"])
            if pid <= 4 or pid == os.getpid() or (hasattr(os, "getppid") and pid == os.getppid()):
                raise ProcessProtectionError(
                    pid,
                    reason="Cannot terminate core system (PID <= 4), current process, or parent terminal.",
                )

            def _kill():
                import psutil
                try:
                    proc = psutil.Process(pid)
                except psutil.NoSuchProcess:
                    return f"Process {pid} not found (may have already terminated)."
                except psutil.AccessDenied as exc:
                    raise ProcessProtectionError(
                        pid,
                        reason=f"Access denied inspecting process identity (fail-closed): {exc}",
                    ) from exc
                except Exception as exc:
                    raise ProcessProtectionError(
                        pid,
                        reason=f"Unable to inspect process (fail-closed): {exc}",
                    ) from exc

                # Fail-closed process identity validation
                try:
                    if not proc.is_running():
                        return f"Process {pid} not found (may have already terminated)."
                    create_time = proc.create_time()
                    name = proc.name()
                except psutil.NoSuchProcess:
                    return f"Process {pid} not found (may have already terminated)."
                except psutil.AccessDenied as exc:
                    raise ProcessProtectionError(
                        pid,
                        reason=f"Access denied inspecting process name (fail-closed): {exc}",
                    ) from exc
                except Exception as exc:
                    raise ProcessProtectionError(
                        pid,
                        reason=f"Failed to inspect process name (fail-closed): {exc}",
                    ) from exc

                if not name or not isinstance(name, str):
                    raise ProcessProtectionError(
                        pid,
                        reason="Process identity could not be verified: empty process name (fail-closed).",
                    )

                if is_critical_process_name(name):
                    raise ProcessProtectionError(
                        pid,
                        process_name=name,
                        reason=f"Process '{name}' is a protected Windows critical system process.",
                    )

                try:
                    exe_path = proc.exe()
                    if exe_path and is_critical_process_name(Path(exe_path).name):
                        raise ProcessProtectionError(
                            pid,
                            process_name=Path(exe_path).name,
                            reason=f"Process binary '{Path(exe_path).name}' is a protected Windows critical system process.",
                        )
                except psutil.NoSuchProcess:
                    return f"Process {pid} not found (may have already terminated)."
                except (psutil.AccessDenied, Exception):
                    pass

                # Post-validation TOCTOU / PID-reuse check and termination
                try:
                    if not proc.is_running() or proc.create_time() != create_time:
                        return f"Process {pid} not found (may have already terminated)."

                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        if proc.is_running() and proc.create_time() == create_time:
                            proc.kill()
                            try:
                                proc.wait(timeout=3)
                            except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                                pass
                    return f"Process {pid} terminated successfully."
                except psutil.NoSuchProcess:
                    return f"Process {pid} not found (may have already terminated)."
                except psutil.AccessDenied as exc:
                    raise ToolExecutionError(
                        "kill_process",
                        f"Access denied terminating process {pid}: {exc}",
                    ) from exc

            return await asyncio.to_thread(_kill)

        return f"Tool '{tool_name}' executed."


class ExecutionFirewall:
    """Central Security Firewall orchestrating capability validation, path authorization, tiered human approval, process sandboxing, and HMAC audit chaining."""

    memory_store: Optional[Any] = None

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        registry: Optional[ToolRegistry] = None,
        path_guard: Optional[WindowsPathGuard] = None,
        process_guard: Optional[WindowsProcessGuard] = None,
        approval_manager: Optional[ApprovalManager] = None,
        audit_ledger: Optional[AuditLedger] = None,
        enable_two_phase_audit: bool = True,
        memory_store: Optional[Any] = None,
        authorizer: Optional[PolicyAuthorizer] = None,
        executor: Optional[ToolExecutor] = None,
    ) -> None:
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.registry = registry or ToolRegistry.create_default()
        self.path_guard = path_guard or WindowsPathGuard(workspace_root=self.workspace_root)
        self.process_guard = process_guard or WindowsProcessGuard(workspace_root=self.workspace_root)
        self.approval_manager = approval_manager or ApprovalManager()
        self.enable_two_phase_audit = enable_two_phase_audit
        self.memory_store = memory_store

        if audit_ledger is None:
            audit_dir = self.workspace_root / ".shamu" / "audit"
            cp_dir = self.workspace_root / ".shamu" / "checkpoints"
            audit_dir.mkdir(parents=True, exist_ok=True)
            cp_dir.mkdir(parents=True, exist_ok=True)
            self.audit_ledger = AuditLedger(
                log_path=audit_dir / "audit.jsonl",
                checkpoint_path=cp_dir / "checkpoints.db",
            )
        else:
            self.audit_ledger = audit_ledger

        self._custom_executors: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

        self.authorizer = authorizer or PolicyAuthorizer(
            registry=self.registry,
            path_guard=self.path_guard,
            process_guard=self.process_guard,
            approval_manager=self.approval_manager,
            audit_ledger=self.audit_ledger,
            workspace_root=self.workspace_root,
            firewall=self,
        )

        self.executor = executor or ToolExecutor(
            process_guard=self.process_guard,
            memory_store=self.memory_store,
            custom_executors=self._custom_executors,
            firewall=self,
        )

    def register_custom_executor(self, tool_name: str, fn: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a custom execution callable for a tool."""
        self._custom_executors[tool_name] = fn
        self.executor.register_custom_executor(tool_name, fn)

    def get_tool_definitions(self):
        """Returns authorized ToolDefinitions from the underlying registry."""
        return self.registry.get_tool_definitions()

    def _is_within_workspace(self, path_str: str) -> bool:
        """Checks if a path resides within the workspace root."""
        return self.path_guard.is_safe_path(path_str, mode="read")

    def evaluate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> PermissionDecision:
        """Evaluates tool name and arguments against security policies (backward compatibility)."""
        return self.authorizer.evaluate_policy(tool_name, arguments)

    async def authorize_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: str = "default",
        turn_id: str = "turn-0",
    ) -> AuthorizedToolCall:
        """Central authorization entrypoint validating policies, boundaries, and approvals."""
        return await self.authorizer.authorize_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            session_id=session_id,
            turn_id=turn_id,
        )

    async def _dispatch_tool_execution(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Dispatches concrete tool execution through the concrete ToolExecutor."""
        return await self.executor.execute(tool_name, arguments)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: str = "default",
        turn_id: str = "turn-0",
    ) -> Any:
        """Central execution gateway enforcing authorization, path safety, interactive approvals, sandboxing, and HMAC audit chaining."""
        # 1. Authorize tool call through policy authorizer
        auth_call = await self.authorize_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            session_id=session_id,
            turn_id=turn_id,
        )

        cap = auth_call.capability
        decision = auth_call.decision
        approval_id = auth_call.approval_id
        raw_paths = auth_call.raw_paths

        # 2. Dispatch Protected Execution (Two-Phase Write-Ahead Logging: Phase 1 Intent)
        if self.enable_two_phase_audit:
            self.audit_ledger.append_event(
                session_id=session_id,
                turn_id=turn_id,
                tool_name=tool_name,
                arguments=arguments,
                tier=cap.tier,
                decision=decision,
                approval_id=approval_id,
                outcome="PENDING",
                action_type="EXECUTE_INTENT",
            )

        t0 = time.perf_counter()
        outcome = "SUCCESS"
        error_msg = None
        result: Any = None

        try:
            # Re-authorize path immediately prior to file operations (post-approval in Step 5) to eliminate TOCTOU directory junction race conditions
            c_tier = cap.tier.canonical()
            mode = "read" if c_tier == PermissionTier.TIER_1_READ_ONLY else "write"
            path_keys = cap.path_arg_names or (["path"] if "path" in arguments else [])
            for k in path_keys:
                if k in arguments and arguments[k] is not None:
                    targets_to_check = []
                    if k in raw_paths:
                        targets_to_check.append(raw_paths[k])
                    if str(arguments[k]) not in targets_to_check:
                        targets_to_check.append(str(arguments[k]))
                    for p_to_check in targets_to_check:
                        try:
                            auth_path = self.path_guard.authorize_path(str(p_to_check), mode=mode)
                            arguments[k] = str(auth_path)
                        except PathSecurityViolationError as exc:
                            raise PermissionDeniedError(
                                f"Target path is outside approved workspace: {exc}"
                            ) from exc

            result = await self._dispatch_tool_execution(tool_name, arguments)
        except Exception as exc:
            outcome = "FAILURE"
            error_msg = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            self.audit_ledger.append_event(
                session_id=session_id,
                turn_id=turn_id,
                tool_name=tool_name,
                arguments=arguments,
                tier=cap.tier,
                decision=decision,
                approval_id=approval_id,
                outcome=outcome,
                duration_ms=duration_ms,
                error=error_msg,
                action_type="EXECUTE_COMPLETE" if self.enable_two_phase_audit else "tool_execution",
            )

        return result

    @staticmethod
    def _focus_window(window_title: str) -> str:
        """Finds application window matching window_title (case-insensitive substring) and brings it to foreground."""
        import sys
        if sys.platform != "win32":
            return f"focus_application is only supported on Windows platforms (current: {sys.platform})."

        target_lower = window_title.lower()
        matched_hwnd = None
        matched_title = ""

        # 1. Try win32gui if available
        try:
            import win32gui
            def _enum_win32gui(hwnd, _):
                nonlocal matched_hwnd, matched_title
                if matched_hwnd is None and win32gui.IsWindowVisible(hwnd):
                    text = win32gui.GetWindowText(hwnd)
                    if text and target_lower in text.lower():
                        matched_hwnd = hwnd
                        matched_title = text
                        return False
                return True

            try:
                win32gui.EnumWindows(_enum_win32gui, None)
            except Exception:
                pass

            if matched_hwnd:
                try:
                    win32gui.ShowWindow(matched_hwnd, 9)  # SW_RESTORE
                    win32gui.SetForegroundWindow(matched_hwnd)
                except Exception:
                    pass
                return f"Application window '{matched_title}' focused successfully."
        except ImportError:
            pass

        # 2. Fallback to ctypes on Windows
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            def _enum_ctypes(hwnd, _):
                nonlocal matched_hwnd, matched_title
                if matched_hwnd is None and user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        if target_lower in buff.value.lower():
                            matched_hwnd = hwnd
                            matched_title = buff.value
                            return False
                return True

            cb = WNDENUMPROC(_enum_ctypes)
            user32.EnumWindows(cb, 0)

            if matched_hwnd is not None:
                try:
                    user32.ShowWindow(matched_hwnd, 9)  # SW_RESTORE
                    user32.SetForegroundWindow(matched_hwnd)
                except Exception:
                    pass
                return f"Application window '{matched_title}' focused successfully."
        except Exception as exc:
            return f"Error focusing application window: {exc}"

        return f"No window found matching '{window_title}'."


# Backwards compatibility alias
SecurityFirewall = ExecutionFirewall
