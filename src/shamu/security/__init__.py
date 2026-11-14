"""SHAMU Security and Firewall Subsystem."""

from shamu.security.approval_manager import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalToken,
    TokenSecurityError,
)
from shamu.security.audit import AuditLogger
from shamu.security.audit_ledger import (
    AuditIntegrityError,
    AuditLedger,
    AuditRecord,
    AuditVerificationResult,
)
from shamu.security.firewall import (
    AuthorizedToolCall,
    ExecutionFirewall,
    PermissionDecision,
    PermissionTier,
    PolicyAuthorizer,
    ProcessProtectionError,
    SecurityFirewall,
    TIER_1_TOOLS,
    TIER_2_TOOLS,
    TIER_3_TOOLS,
    ToolExecutor,
)
from shamu.security.path_guard import (
    PathSecurityViolationError,
    WindowsPathGuard,
)
from shamu.security.process_guard import (
    ProcessResult,
    WindowsProcessGuard,
)
from shamu.security.tool_registry import (
    PolicyOutcome,
    ToolCapability,
    ToolRegistry,
)

__all__ = [
    "SecurityFirewall",
    "ExecutionFirewall",
    "PolicyAuthorizer",
    "ToolExecutor",
    "AuthorizedToolCall",
    "ProcessProtectionError",
    "PermissionTier",
    "PermissionDecision",
    "AuditLogger",
    "AuditLedger",
    "AuditRecord",
    "AuditVerificationResult",
    "AuditIntegrityError",
    "WindowsPathGuard",
    "PathSecurityViolationError",
    "WindowsProcessGuard",
    "ProcessResult",
    "ToolRegistry",
    "ToolCapability",
    "PolicyOutcome",
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalToken",
    "TokenSecurityError",
    "TIER_1_TOOLS",
    "TIER_2_TOOLS",
    "TIER_3_TOOLS",
]
