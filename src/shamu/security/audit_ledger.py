"""Tamper-Evident HMAC-SHA-256 Audit Ledger with Tail Checkpointing & DPAPI Key Storage."""

from __future__ import annotations

import atexit
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import weakref
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
import ctypes
from ctypes import wintypes

try:
    import keyring
except Exception:
    keyring = None

from shamu.protocols import ShamuError
from shamu.security.tool_registry import PermissionTier

_active_ledgers = weakref.WeakSet()


def _cleanup_all_ledgers() -> None:
    """Closes all active audit ledger checkpoint connections deterministically upon process exit."""
    for ledger in list(_active_ledgers):
        try:
            ledger.close()
        except Exception:
            pass


atexit.register(_cleanup_all_ledgers)

class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _dpapi_protect(data: bytes) -> bytes:
    """Encrypts bytes using Windows DPAPI CryptProtectData."""
    if sys.platform != "win32":
        raise OSError("DPAPI is only supported on Windows.")

    data_in = _DATA_BLOB()
    data_in.cbData = len(data)
    data_in.pbData = ctypes.cast(ctypes.create_string_buffer(data, len(data)), ctypes.POINTER(ctypes.c_byte))
    data_out = _DATA_BLOB()

    # dwFlags = 1: CRYPTPROTECT_UI_FORBIDDEN
    res = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(data_in),
        "shamu_master_key",
        None,
        None,
        None,
        1,
        ctypes.byref(data_out),
    )
    if not res:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(data_out.pbData, data_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(data_out.pbData)


def _dpapi_unprotect(data: bytes) -> bytes:
    """Decrypts bytes using Windows DPAPI CryptUnprotectData."""
    if sys.platform != "win32":
        raise OSError("DPAPI is only supported on Windows.")

    data_in = _DATA_BLOB()
    data_in.cbData = len(data)
    data_in.pbData = ctypes.cast(ctypes.create_string_buffer(data, len(data)), ctypes.POINTER(ctypes.c_byte))
    data_out = _DATA_BLOB()

    # dwFlags = 1: CRYPTPROTECT_UI_FORBIDDEN
    res = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(data_in),
        None,
        None,
        None,
        None,
        1,
        ctypes.byref(data_out),
    )
    if not res:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(data_out.pbData, data_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(data_out.pbData)

SENSITIVE_KEY_SUBSTRINGS = (
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "auth",
    "authorization",
    "bearer",
    "private_key",
    "credential",
    "cookie",
)

SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"ghp_[a-zA-Z0-9]{36}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"),
]


def sanitize_audit_arguments(data: Any) -> Any:
    """Recursively scrub sensitive keys and secret token patterns from audit arguments."""
    if isinstance(data, dict):
        sanitized: Dict[str, Any] = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(s in k_lower for s in SENSITIVE_KEY_SUBSTRINGS):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_audit_arguments(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_audit_arguments(item) for item in data]
    elif isinstance(data, str):
        val = data
        for pattern in SECRET_VALUE_PATTERNS:
            val = pattern.sub("[REDACTED]", val)
        return val
    return data



class AuditIntegrityError(ShamuError):
    """Raised when audit ledger tampering, corruption, or truncation is detected."""
    pass


@dataclass
class AuditRecord:
    """An individual cryptographically authenticated audit log record."""
    sequence_id: int
    timestamp: str
    session_id: str
    turn_id: str
    tool_name: str
    action_type: str
    sanitized_arguments: Dict[str, Any]
    tier: str
    decision: str
    approval_id: Optional[str] = None
    outcome: str = "SUCCESS"
    duration_ms: Optional[float] = None
    error_category: Optional[str] = None
    prev_hash: str = "0" * 64
    current_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def payload_dict(self) -> Dict[str, Any]:
        """Extract canonical payload fields for HMAC computation."""
        d = self.to_dict()
        d.pop("prev_hash", None)
        d.pop("current_hash", None)
        return d


@dataclass
class AuditVerificationResult:
    """Outcome of an audit ledger integrity verification scan."""
    is_valid: bool
    verified_records: int
    violation_reason: Optional[str] = None
    tampered_sequence_id: Optional[int] = None


class AuditLedger:
    """Append-only, tamper-evident audit ledger chained via HMAC-SHA-256."""

    KEYRING_SERVICE = "shamu_audit_ledger"
    KEYRING_USERNAME = "master_key"
    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        log_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        master_key: Optional[bytes] = None,
        auto_verify: bool = True,
        key_path: Optional[Path] = None,
    ) -> None:
        if log_path is None:
            base_dir = Path.home() / ".shamu" / "audit"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = base_dir / "audit.jsonl"
        else:
            self.log_path = Path(log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if checkpoint_path is None:
            # SEC-10: Store checkpoints in a segregated secondary directory from the append log
            # to protect against co-located directory deletion and rollback attacks.
            base_dir = self.log_path.parent.parent / "checkpoints" if self.log_path.parent.name == "audit" else self.log_path.parent
            self.checkpoint_path = base_dir / "checkpoints.db"
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_path = Path(checkpoint_path)
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.key_path = Path(key_path) if key_path is not None else None
        self.master_key = master_key or self.get_or_create_master_key(key_path=self.key_path)
        self.is_locked = False
        self._lock = threading.Lock()
        self._checkpoint_conn: Optional[sqlite3.Connection] = None
        self._last_record_cache: Optional[Tuple[int, str]] = None

        _active_ledgers.add(self)

        self._init_checkpoint_db()

        # SEC-06: Automatically verify ledger integrity on startup
        if auto_verify:
            try:
                self.verify_integrity()
            except AuditIntegrityError:
                pass

    @classmethod
    def get_or_create_master_key(cls, key_path: Optional[Path] = None) -> bytes:
        """Retrieve master key from Windows Credential Manager / DPAPI via keyring or fallback to DPAPI file."""
        target_file = Path(key_path) if key_path is not None else (Path.home() / ".shamu" / "master.key")

        # 1. Try keyring if available
        if keyring is not None:
            try:
                stored_hex = keyring.get_password(cls.KEYRING_SERVICE, cls.KEYRING_USERNAME)
                if stored_hex:
                    return bytes.fromhex(stored_hex)
            except Exception:
                pass

        # 2. Try loading from DPAPI-protected file .shamu/master.key
        if target_file.exists():
            try:
                encrypted_data = target_file.read_bytes()
                if encrypted_data:
                    decrypted_key = _dpapi_unprotect(encrypted_data)
                    if len(decrypted_key) == 32:
                        return decrypted_key
            except Exception:
                pass

        # 3. Generate a new 32-byte key
        new_key = secrets.token_bytes(32)

        # 4. Try storing in keyring first
        keyring_saved = False
        if keyring is not None:
            try:
                keyring.set_password(cls.KEYRING_SERVICE, cls.KEYRING_USERNAME, new_key.hex())
                keyring_saved = True
            except Exception:
                keyring_saved = False

        # 5. If keyring fails or is unavailable, use Windows DPAPI via CryptProtectData to store
        if not keyring_saved:
            try:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                encrypted_blob = _dpapi_protect(new_key)
                target_file.write_bytes(encrypted_blob)
            except Exception:
                pass

        return new_key

    def _get_checkpoint_conn(self) -> sqlite3.Connection:
        if self._checkpoint_conn is None:
            self._checkpoint_conn = sqlite3.connect(self.checkpoint_path, check_same_thread=False)
            self._checkpoint_conn.execute("PRAGMA journal_mode=WAL;")
            self._checkpoint_conn.execute("PRAGMA synchronous=NORMAL;")
            self._checkpoint_conn.execute("PRAGMA busy_timeout=30000;")
        return self._checkpoint_conn

    def _init_checkpoint_db(self) -> None:
        """Ensure the secondary checkpoint table exists."""
        with self._lock:
            conn = self._get_checkpoint_conn()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_checkpoints (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        sequence_id INTEGER NOT NULL,
                        current_hash TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                cursor.close()

    def _get_tail_checkpoint(self) -> Optional[Tuple[int, str]]:
        """Get the last recorded checkpoint (sequence_id, current_hash)."""
        if not self.checkpoint_path.exists():
            return None
        try:
            conn = self._get_checkpoint_conn()
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT sequence_id, current_hash FROM audit_checkpoints WHERE id = 1")
                row = cursor.fetchone()
                if row:
                    return int(row[0]), str(row[1])
                return None
            finally:
                cursor.close()
        except Exception:
            return None

    def _save_tail_checkpoint(self, sequence_id: int, current_hash: str) -> None:
        """Update the secondary checkpoint store with the latest sequence and hash."""
        now_str = datetime.now(timezone.utc).isoformat()
        conn = self._get_checkpoint_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO audit_checkpoints (id, sequence_id, current_hash, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    sequence_id = excluded.sequence_id,
                    current_hash = excluded.current_hash,
                    updated_at = excluded.updated_at
                """,
                (sequence_id, current_hash, now_str),
            )
            conn.commit()
        finally:
            cursor.close()

    def close(self) -> None:
        """Closes secondary checkpoint connection if open."""
        with self._lock:
            _active_ledgers.discard(self)
            if self._checkpoint_conn is not None:
                try:
                    self._checkpoint_conn.close()
                except Exception:
                    pass
                self._checkpoint_conn = None

    def __enter__(self) -> "AuditLedger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _calculate_hmac(self, prev_hash: str, payload_dict: Dict[str, Any]) -> str:
        """Compute deterministic HMAC-SHA-256 over prev_hash and canonical JSON payload."""
        canonical_bytes = json.dumps(payload_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        hmac_input = prev_hash.encode("utf-8") + canonical_bytes
        return hmac.new(self.master_key, hmac_input, hashlib.sha256).hexdigest()

    def _get_last_record(self) -> Optional[Tuple[int, str]]:
        """Read the last record in the current log file to determine sequence and prev_hash."""
        if self._last_record_cache is not None:
            return self._last_record_cache

        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return None

        last_line = None
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped

        if not last_line:
            return None

        try:
            data = json.loads(last_line)
            res = (int(data["sequence_id"]), str(data["current_hash"]))
            self._last_record_cache = res
            return res
        except Exception:
            return None

    def append_event(
        self,
        session_id: str,
        turn_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        tier: PermissionTier | str,
        decision: str,
        approval_id: Optional[str] = None,
        outcome: str = "SUCCESS",
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        action_type: str = "tool_execution",
    ) -> AuditRecord:
        """Appends a cryptographically authenticated event to the audit ledger."""
        with self._lock:
            if self.is_locked:
                raise AuditIntegrityError("Audit ledger is locked due to integrity violation.")

            last_rec = self._get_last_record()
            if last_rec is None:
                seq_id = 1
                prev_hash = self.GENESIS_HASH
            else:
                seq_id = last_rec[0] + 1
                prev_hash = last_rec[1]

            tier_str = tier.value if hasattr(tier, "value") else str(tier)
            timestamp_str = datetime.now(timezone.utc).isoformat()
            sanitized_args = sanitize_audit_arguments(arguments)

            record = AuditRecord(
                sequence_id=seq_id,
                timestamp=timestamp_str,
                session_id=session_id,
                turn_id=turn_id,
                tool_name=tool_name,
                action_type=action_type,
                sanitized_arguments=sanitized_args,
                tier=tier_str,
                decision=decision,
                approval_id=approval_id,
                outcome=outcome,
                duration_ms=duration_ms,
                error_category=error,
                prev_hash=prev_hash,
                current_hash="",
            )

            record.current_hash = self._calculate_hmac(prev_hash, record.payload_dict())

            # Append to log file atomically
            line_json = json.dumps(record.to_dict()) + "\n"
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line_json)
                f.flush()

            # Update secondary checkpoint
            self._save_tail_checkpoint(record.sequence_id, record.current_hash)
            self._last_record_cache = (record.sequence_id, record.current_hash)

            return record

    def verify_integrity(self) -> AuditVerificationResult:
        """Scan the audit log, recompute HMAC chain, and verify against secondary checkpoint."""
        with self._lock:
            self._last_record_cache = None
            checkpoint = self._get_tail_checkpoint()

            if not self.log_path.exists() or self.log_path.stat().st_size == 0:
                if checkpoint is not None and checkpoint[0] > 0:
                    self.is_locked = True
                    return AuditVerificationResult(
                        is_valid=False,
                        verified_records=0,
                        violation_reason=f"Audit log is empty but checkpoint indicates {checkpoint[0]} records exist (tail truncation detected).",
                        tampered_sequence_id=1,
                    )
                return AuditVerificationResult(is_valid=True, verified_records=0)

            expected_seq = 1
            expected_prev_hash = self.GENESIS_HASH
            verified_count = 0
            last_current_hash = ""

            with open(self.log_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue

                    try:
                        record_dict = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        self.is_locked = True
                        return AuditVerificationResult(
                            is_valid=False,
                            verified_records=verified_count,
                            violation_reason=f"Malformed JSON at line {line_idx}: {exc}",
                            tampered_sequence_id=expected_seq,
                        )

                    seq = record_dict.get("sequence_id")
                    if seq != expected_seq:
                        self.is_locked = True
                        return AuditVerificationResult(
                            is_valid=False,
                            verified_records=verified_count,
                            violation_reason=f"Sequence continuity broken at line {line_idx}: expected {expected_seq}, got {seq}.",
                            tampered_sequence_id=expected_seq,
                        )

                    prev_h = record_dict.get("prev_hash")
                    if prev_h != expected_prev_hash:
                        self.is_locked = True
                        return AuditVerificationResult(
                            is_valid=False,
                            verified_records=verified_count,
                            violation_reason=f"Hash chain broken at sequence {seq}: expected prev_hash '{expected_prev_hash}', got '{prev_h}'.",
                            tampered_sequence_id=seq,
                        )

                    # Verify HMAC
                    payload = dict(record_dict)
                    payload.pop("prev_hash", None)
                    actual_current_h = payload.pop("current_hash", None)

                    expected_current_h = self._calculate_hmac(prev_h, payload)
                    if not hmac.compare_digest(str(actual_current_h), expected_current_h):
                        self.is_locked = True
                        return AuditVerificationResult(
                            is_valid=False,
                            verified_records=verified_count,
                            violation_reason=f"HMAC mutation detected at sequence {seq}.",
                            tampered_sequence_id=seq,
                        )

                    last_current_hash = str(actual_current_h)
                    expected_prev_hash = last_current_hash
                    expected_seq += 1
                    verified_count += 1

            # Checkpoint tail verification
            if checkpoint is None:
                # AUDIT-03: If log contains records but checkpoint does not exist or has no checkpoint record,
                # fail integrity with AuditIntegrityError and lock ledger.
                if verified_count > 0:
                    self.is_locked = True
                    raise AuditIntegrityError(
                        f"Audit log contains {verified_count} records but checkpoint database or record is missing (tampering/truncation alarm)."
                    )
            else:
                cp_seq, cp_hash = checkpoint
                if cp_seq > verified_count:
                    self.is_locked = True
                    return AuditVerificationResult(
                        is_valid=False,
                        verified_records=verified_count,
                        violation_reason=f"Log truncation detected: checkpoint sequence {cp_seq} > verified log sequence {verified_count}.",
                        tampered_sequence_id=verified_count + 1,
                    )
                if cp_seq == verified_count and cp_hash != last_current_hash:
                    self.is_locked = True
                    return AuditVerificationResult(
                        is_valid=False,
                        verified_records=verified_count,
                        violation_reason=f"Checkpoint hash mismatch at sequence {cp_seq}: expected '{cp_hash}', got '{last_current_hash}'.",
                        tampered_sequence_id=cp_seq,
                    )

            if verified_count > 0:
                self._last_record_cache = (verified_count, last_current_hash)

            return AuditVerificationResult(is_valid=True, verified_records=verified_count)
