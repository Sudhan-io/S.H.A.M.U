"""Immutable JSON-Lines Audit Logging for SHAMU actions."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from shamu.security.firewall import PermissionTier


class AuditLogger:
    """Logs tool execution requests, security decisions, and outcomes."""

    def __init__(self, log_file: Optional[Path] = None):
        if log_file is None:
            log_dir = Path.home() / ".shamu"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = log_dir / "audit.jsonl"
        else:
            self.log_file = log_file
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tier: PermissionTier,
        approval_status: str,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Appends a structured event entry to the audit log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "arguments": arguments,
            "tier": tier.value if hasattr(tier, "value") else str(tier),
            "approval_status": approval_status,
            "duration_ms": duration_ms,
            "error": error,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

