import json
from pathlib import Path
import pytest
from shamu.security.tool_registry import PermissionTier
from shamu.security.audit_ledger import (
    AuditLedger,
    AuditRecord,
    AuditVerificationResult,
    AuditIntegrityError,
)


def test_sequential_append_and_hash_calculation(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    cp_file = tmp_path / "checkpoints.db"
    key = b"A" * 32

    ledger = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)

    r1 = ledger.append_event(
        session_id="sess-1",
        turn_id="turn-1",
        tool_name="read_file",
        arguments={"path": "a.txt"},
        tier=PermissionTier.TIER_1_READ_ONLY,
        decision="ALLOW",
    )

    assert r1.sequence_id == 1
    assert r1.prev_hash == "0" * 64
    assert len(r1.current_hash) == 64

    r2 = ledger.append_event(
        session_id="sess-1",
        turn_id="turn-2",
        tool_name="write_file",
        arguments={"path": "b.txt", "content": "hello"},
        tier=PermissionTier.TIER_2_MUTATION,
        decision="REQUIRE_APPROVAL",
        approval_id="appr-123",
    )

    assert r2.sequence_id == 2
    assert r2.prev_hash == r1.current_hash

    r3 = ledger.append_event(
        session_id="sess-1",
        turn_id="turn-3",
        tool_name="powershell_exec",
        arguments={"command": "dir"},
        tier=PermissionTier.TIER_3_HIGH_IMPACT,
        decision="REQUIRE_APPROVAL",
        approval_id="appr-456",
    )

    assert r3.sequence_id == 3
    assert r3.prev_hash == r2.current_hash

    # Verify entire log is valid
    res = ledger.verify_integrity()
    assert res.is_valid is True
    assert res.verified_records == 3
    assert res.violation_reason is None


def test_detect_line_mutation_and_lock(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    cp_file = tmp_path / "checkpoints.db"
    key = b"K" * 32

    ledger = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)

    for i in range(1, 4):
        ledger.append_event(
            session_id="sess-1",
            turn_id=f"turn-{i}",
            tool_name="read_file",
            arguments={"path": f"file_{i}.txt"},
            tier=PermissionTier.TIER_1_READ_ONLY,
            decision="ALLOW",
        )

    # Tamper with record 2 by changing the arguments
    lines = log_file.read_text(encoding="utf-8").splitlines()
    rec2 = json.loads(lines[1])
    rec2["sanitized_arguments"]["path"] = "tampered_file.txt"
    lines[1] = json.dumps(rec2)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = ledger.verify_integrity()
    assert res.is_valid is False
    assert res.tampered_sequence_id == 2
    assert "mutation" in res.violation_reason.lower() or "hmac" in res.violation_reason.lower()

    # Fail-closed lock: subsequent append raises AuditIntegrityError
    with pytest.raises(AuditIntegrityError, match="locked"):
        ledger.append_event(
            session_id="sess-1",
            turn_id="turn-4",
            tool_name="read_file",
            arguments={"path": "file_4.txt"},
            tier=PermissionTier.TIER_1_READ_ONLY,
            decision="ALLOW",
        )


def test_detect_line_deletion(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    cp_file = tmp_path / "checkpoints.db"
    key = b"D" * 32

    ledger = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)

    for i in range(1, 4):
        ledger.append_event(
            session_id="sess-1",
            turn_id=f"turn-{i}",
            tool_name="read_file",
            arguments={"path": f"file_{i}.txt"},
            tier=PermissionTier.TIER_1_READ_ONLY,
            decision="ALLOW",
        )

    # Delete line 2 (seq 2)
    lines = log_file.read_text(encoding="utf-8").splitlines()
    del lines[1]
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = ledger.verify_integrity()
    assert res.is_valid is False
    assert res.tampered_sequence_id == 3 or "sequence" in res.violation_reason.lower() or "hash" in res.violation_reason.lower()


def test_detect_line_insertion(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    cp_file = tmp_path / "checkpoints.db"
    key = b"I" * 32

    ledger = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)

    for i in range(1, 3):
        ledger.append_event(
            session_id="sess-1",
            turn_id=f"turn-{i}",
            tool_name="read_file",
            arguments={"path": f"file_{i}.txt"},
            tier=PermissionTier.TIER_1_READ_ONLY,
            decision="ALLOW",
        )

    # Insert a rogue forged record
    lines = log_file.read_text(encoding="utf-8").splitlines()
    forged = {
        "sequence_id": 99,
        "timestamp": "2026-10-30T00:00:00Z",
        "session_id": "sess-1",
        "turn_id": "turn-fake",
        "tool_name": "cmd_exec",
        "action_type": "shell_execution",
        "sanitized_arguments": {"command": "whoami"},
        "tier": "TIER_3_HIGH_IMPACT",
        "decision": "ALLOW",
        "approval_id": None,
        "outcome": "SUCCESS",
        "duration_ms": 1.0,
        "error_category": None,
        "prev_hash": "0" * 64,
        "current_hash": "deadbeef" * 8,
    }
    lines.insert(1, json.dumps(forged))
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = ledger.verify_integrity()
    assert res.is_valid is False


def test_detect_tail_truncation_via_checkpoint(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    cp_file = tmp_path / "checkpoints.db"
    key = b"T" * 32

    ledger = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)

    for i in range(1, 6):
        ledger.append_event(
            session_id="sess-1",
            turn_id=f"turn-{i}",
            tool_name="read_file",
            arguments={"path": f"file_{i}.txt"},
            tier=PermissionTier.TIER_1_READ_ONLY,
            decision="ALLOW",
        )

    # Checkpoint has sequence 5. Now truncate records 4 and 5 from audit.jsonl
    lines = log_file.read_text(encoding="utf-8").splitlines()
    lines = lines[:3]
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = ledger.verify_integrity()
    assert res.is_valid is False
    assert "truncat" in res.violation_reason.lower()


def test_detect_empty_log_with_active_checkpoint(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    cp_file = tmp_path / "checkpoints.db"
    key = b"E" * 32

    ledger = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)

    ledger.append_event(
        session_id="sess-1",
        turn_id="turn-1",
        tool_name="read_file",
        arguments={"path": "file.txt"},
        tier=PermissionTier.TIER_1_READ_ONLY,
        decision="ALLOW",
    )

    # Completely wipe log file
    log_file.write_text("", encoding="utf-8")

    res = ledger.verify_integrity()
    assert res.is_valid is False
    assert "truncat" in res.violation_reason.lower() or "empty" in res.violation_reason.lower()


def test_canonical_json_key_order_deterministic(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    cp_file = tmp_path / "checkpoints.db"
    key = b"C" * 32

    ledger = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)

    # Different key insertion orders in arguments
    r1 = ledger.append_event(
        session_id="sess-1",
        turn_id="turn-1",
        tool_name="write_file",
        arguments={"z": 1, "a": 2, "m": 3},
        tier=PermissionTier.TIER_2_MUTATION,
        decision="ALLOW",
    )

    assert ledger.verify_integrity().is_valid is True


def test_dpapi_master_key_retrieval(tmp_path: Path, monkeypatch):
    log_file = tmp_path / "audit.jsonl"
    cp_file = tmp_path / "checkpoints.db"

    fake_vault = {}

    def mock_get_password(service, username):
        return fake_vault.get(f"{service}:{username}")

    def mock_set_password(service, username, password):
        fake_vault[f"{service}:{username}"] = password

    import keyring
    monkeypatch.setattr(keyring, "get_password", mock_get_password)
    monkeypatch.setattr(keyring, "set_password", mock_set_password)

    # First instance creates key
    ledger1 = AuditLedger(log_path=log_file, checkpoint_path=cp_file)
    key1 = ledger1.master_key

    assert len(key1) == 32

    # Second instance retrieves existing key
    ledger2 = AuditLedger(log_path=log_file, checkpoint_path=cp_file)
    key2 = ledger2.master_key

    assert key1 == key2


def test_startup_integrity_verification_locks_ledger(tmp_path: Path):
    """SEC-06: Tampered or truncated audit ledger must fail closed on startup."""
    log_file = tmp_path / "audit.jsonl"
    cp_file = tmp_path / "checkpoints.db"
    key = b"S" * 32

    # Step 1: Create legitimate ledger and log 3 records
    ledger1 = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)
    for i in range(1, 4):
        ledger1.append_event(
            session_id="sess-1",
            turn_id=f"turn-{i}",
            tool_name="read_file",
            arguments={"path": f"doc_{i}.txt"},
            tier=PermissionTier.TIER_1_READ_ONLY,
            decision="ALLOW",
        )
    ledger1.close()

    # Step 2: Tamper with log file while process is stopped (truncate last record)
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    log_file.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    # Step 3: Startup new AuditLedger instance
    ledger2 = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)
    try:
        assert ledger2.is_locked is True, "AuditLedger should be locked immediately on startup when log is truncated"
        with pytest.raises(AuditIntegrityError, match="locked"):
            ledger2.append_event(
                session_id="sess-1",
                turn_id="turn-new",
                tool_name="read_file",
                arguments={"path": "doc_new.txt"},
                tier=PermissionTier.TIER_1_READ_ONLY,
                decision="ALLOW",
            )
    finally:
        ledger2.close()


def test_sensitive_argument_sanitization(tmp_path: Path):
    """SEC-11: Sensitive argument keys and secret tokens must be redacted in audit ledger."""
    log_file = tmp_path / "audit.jsonl"
    cp_file = tmp_path / "checkpoints.db"
    key = b"M" * 32

    ledger = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)
    try:
        rec = ledger.append_event(
            session_id="sess-sec11",
            turn_id="turn-1",
            tool_name="api_request",
            arguments={
                "url": "https://api.example.com/v1",
                "api_key": "sk-secretkey1234567890abcdef12345",
                "nested": {
                    "password": "supersecretpassword",
                    "bearer_token": "Bearer token-abc-xyz-12345",
                    "normal_param": "regular_value",
                },
                "headers": [
                    {"Authorization": "Bearer secret-token-value-99999"},
                    {"User-Agent": "ShamuBot/1.0"},
                ],
                "cmd": "curl -H 'Authorization: Bearer sk-secretkey1234567890abcdef12345' https://api.example.com",
            },
            tier=PermissionTier.TIER_2_MUTATION,
            decision="ALLOW",
        )

        sanitized = rec.sanitized_arguments
        # Direct sensitive keys redacted
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["nested"]["password"] == "[REDACTED]"
        assert sanitized["nested"]["bearer_token"] == "[REDACTED]"
        assert sanitized["nested"]["normal_param"] == "regular_value"

        # List of dicts sensitive key redacted
        assert sanitized["headers"][0]["Authorization"] == "[REDACTED]"
        assert sanitized["headers"][1]["User-Agent"] == "ShamuBot/1.0"

        # Secret regex pattern inside string redacted
        assert "sk-secretkey1234567890abcdef12345" not in sanitized["cmd"]
        assert "[REDACTED]" in sanitized["cmd"]

        # Ensure raw file on disk has NO sensitive tokens
        raw_log = log_file.read_text(encoding="utf-8")
        assert "sk-secretkey1234567890abcdef12345" not in raw_log
        assert "supersecretpassword" not in raw_log
        assert "secret-token-value-99999" not in raw_log

        # Cryptographic chain must verify cleanly with sanitized payload
        v_res = ledger.verify_integrity()
        assert v_res.is_valid is True
    finally:
        ledger.close()


def test_audit_checkpoint_directory_segregation(tmp_path: Path):
    """SEC-10: Checkpoints should default to a segregated secondary directory from the audit log."""
    log_file = tmp_path / "audit" / "audit.jsonl"
    ledger = AuditLedger(log_path=log_file)
    try:
        assert ledger.checkpoint_path.parent != ledger.log_path.parent
        assert ledger.checkpoint_path.parent.name == "checkpoints"
    finally:
        ledger.close()


def test_dpapi_protect_and_unprotect_roundtrip():
    """Verify Windows DPAPI CryptProtectData and CryptUnprotectData direct roundtrip."""
    from shamu.security.audit_ledger import _dpapi_protect, _dpapi_unprotect

    original_secret = b"\x01\x02\x03\x04" * 8
    encrypted_blob = _dpapi_protect(original_secret)
    assert encrypted_blob != original_secret

    decrypted_secret = _dpapi_unprotect(encrypted_blob)
    assert decrypted_secret == original_secret


def test_dpapi_master_key_file_fallback_on_keyring_failure(tmp_path: Path, monkeypatch):
    """Regression test: when keyring fails/unavailable, DPAPI stores/loads master.key persistently."""
    from shamu.security import audit_ledger

    # Simulate keyring failure
    monkeypatch.setattr(audit_ledger, "keyring", None)

    key_file = tmp_path / ".shamu" / "master.key"
    log_file = tmp_path / "audit" / "audit.jsonl"
    cp_file = tmp_path / "checkpoints" / "checkpoints.db"

    # Startup instance 1: Generates key and protects with DPAPI into key_file
    ledger1 = AuditLedger(log_path=log_file, checkpoint_path=cp_file, key_path=key_file)
    key1 = ledger1.master_key
    assert len(key1) == 32
    assert key_file.exists()

    # The file on disk must be encrypted by DPAPI, not raw plaintext key
    raw_file_content = key_file.read_bytes()
    assert raw_file_content != key1

    # Append an event
    r1 = ledger1.append_event(
        session_id="sess-1",
        turn_id="turn-1",
        tool_name="read_file",
        arguments={"path": "doc1.txt"},
        tier=PermissionTier.TIER_1_READ_ONLY,
        decision="ALLOW",
    )
    assert r1.sequence_id == 1
    ledger1.close()

    # Startup instance 2 (simulating process restart with keyring still unavailable):
    # It must reload the persistent DPAPI-protected key instead of an ephemeral key
    ledger2 = AuditLedger(log_path=log_file, checkpoint_path=cp_file, key_path=key_file)
    try:
        assert ledger2.master_key == key1
        assert ledger2.is_locked is False
        v_res = ledger2.verify_integrity()
        assert v_res.is_valid is True
        assert v_res.verified_records == 1

        # Append another event to ensure ledger remains fully operational
        r2 = ledger2.append_event(
            session_id="sess-1",
            turn_id="turn-2",
            tool_name="write_file",
            arguments={"path": "doc2.txt", "content": "persisted"},
            tier=PermissionTier.TIER_2_MUTATION,
            decision="ALLOW",
        )
        assert r2.sequence_id == 2
        assert r2.prev_hash == r1.current_hash
    finally:
        ledger2.close()


def test_missing_checkpoint_with_active_log_raises_integrity_error(tmp_path: Path):
    """AUDIT-03: If audit.jsonl has records but checkpoints.db is deleted, trigger lock and raise AuditIntegrityError."""
    import sqlite3

    log_file = tmp_path / "audit" / "audit.jsonl"
    cp_file = tmp_path / "checkpoints" / "checkpoints.db"
    key = b"M" * 32

    # Step 1: Initialize ledger and log legitimate events
    ledger1 = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)
    for i in range(1, 4):
        ledger1.append_event(
            session_id="sess-1",
            turn_id=f"turn-{i}",
            tool_name="read_file",
            arguments={"path": f"doc_{i}.txt"},
            tier=PermissionTier.TIER_1_READ_ONLY,
            decision="ALLOW",
        )
    ledger1.close()

    # Step 2: Delete checkpoint database entirely (simulating attacker wiping checkpoint store)
    cp_file.unlink()
    assert not cp_file.exists()

    # Step 3: Verify that verify_integrity() triggers is_locked=True and raises AuditIntegrityError
    ledger2 = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key, auto_verify=False)
    try:
        assert ledger2.is_locked is False
        with pytest.raises(AuditIntegrityError, match="tampering/truncation alarm"):
            ledger2.verify_integrity()
        assert ledger2.is_locked is True

        # Ensure ledger is locked and blocks subsequent appends
        with pytest.raises(AuditIntegrityError, match="locked"):
            ledger2.append_event(
                session_id="sess-1",
                turn_id="turn-4",
                tool_name="read_file",
                arguments={"path": "doc_4.txt"},
                tier=PermissionTier.TIER_1_READ_ONLY,
                decision="ALLOW",
            )
    finally:
        ledger2.close()

    # Step 4: Verify that checkpoint database with no checkpoint record also triggers lock and raises
    conn = sqlite3.connect(cp_file)
    conn.execute("DELETE FROM audit_checkpoints")
    conn.commit()
    conn.close()

    ledger3 = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key, auto_verify=False)
    try:
        with pytest.raises(AuditIntegrityError, match="tampering/truncation alarm"):
            ledger3.verify_integrity()
        assert ledger3.is_locked is True
    finally:
        ledger3.close()

    # Step 5: Verify that startup with auto_verify=True locks the ledger immediately
    ledger4 = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key, auto_verify=True)
    try:
        assert ledger4.is_locked is True
        with pytest.raises(AuditIntegrityError, match="locked"):
            ledger4.append_event(
                session_id="sess-1",
                turn_id="turn-5",
                tool_name="read_file",
                arguments={"path": "doc_5.txt"},
                tier=PermissionTier.TIER_1_READ_ONLY,
                decision="ALLOW",
            )
        with pytest.raises(AuditIntegrityError, match="tampering/truncation alarm"):
            ledger4.verify_integrity()
    finally:
        ledger4.close()





