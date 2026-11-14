"""Adversarial regression test suite for CANON-P0-02 (Filesystem Security & Secret Protection).

Enforces fail-closed protection of:
- .shamu internal application state (master.key, audit.db, checkpoints)
- .env and environment variable secret configurations
- .git repository internals (config, HEAD, hooks, objects)
- Private cryptographic keys (master.key, id_rsa, id_ed25519, *.key, *.pem)
- Credential files (credentials.json, service_account.json, aws_credentials.ini)
- Audit secrets and ledgers

Ensures that Tier-1 autonomous tools (e.g. read_file) and mutation tools (write_file)
cannot access, exfiltrate, or tamper with sensitive application state under any
evasion strategy (traversal, case variance, alternate data streams, or NTFS junctions).
"""

import os
import sys
from pathlib import Path
import pytest

from shamu.protocols import PermissionDeniedError
from shamu.security.firewall import ExecutionFirewall
from shamu.security.path_guard import WindowsPathGuard, PathSecurityViolationError

# Real WinAPI junction support
try:
    import _winapi
    HAS_WINAPI = hasattr(_winapi, "CreateJunction")
except ImportError:
    HAS_WINAPI = False


@pytest.fixture
def secure_workspace(tmp_path: Path):
    """Initializes a disposable mock workspace with fake sensitive credentials and safe files."""
    ws = tmp_path / "mock_workspace"
    ws.mkdir()

    # 1. Fake disposable .shamu internal state
    shamu_dir = ws / ".shamu"
    shamu_dir.mkdir()
    (shamu_dir / "master.key").write_bytes(b"DISPOSABLE_TEST_HMAC_MASTER_KEY_32B")
    (shamu_dir / "audit.db").write_text("DISPOSABLE_SQLITE_AUDIT_LEDGER")
    (shamu_dir / "audit.key").write_bytes(b"DISPOSABLE_AUDIT_KEY_MATERIAL_123")

    # 2. Fake disposable .git repository directory
    git_dir = ws / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n\trepositoryformatversion = 0\n[remote \"origin\"]\n\turl = https://token:ghp_fake123@github.com/fake/repo.git\n")
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

    # 3. Fake disposable environment secret files
    (ws / ".env").write_text("OPENAI_API_KEY=sk-fake-disposable-key-12345\nANTHROPIC_API_KEY=sk-ant-fake-67890\n")
    (ws / ".env.local").write_text("LOCAL_SECRET_TOKEN=xyz987\n")
    (ws / ".env.production").write_text("DATABASE_URL=postgres://user:pass@prod.db:5432/main\n")
    (ws / "secrets.env").write_text("PROD_API_KEY=top_secret_fake\n")

    # 4. Fake disposable SSH & TLS private keys
    (ws / "id_rsa").write_text("-----BEGIN RSA PRIVATE KEY-----\nMIIEfake\n-----END RSA PRIVATE KEY-----\n")
    (ws / "id_ed25519").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nb3Blfake\n-----END OPENSSH PRIVATE KEY-----\n")
    (ws / "master.key").write_bytes(b"FAKEMASTERKEYFORTESTING123456789")
    (ws / "server.pem").write_text("-----BEGIN PRIVATE KEY-----\nfake_pem\n-----END PRIVATE KEY-----\n")
    (ws / "deploy.key").write_text("fake_deploy_key_material")

    # 5. Fake disposable credentials
    (ws / "credentials.json").write_text('{"client_secret": "fake_client_secret_xyz"}')
    (ws / "service_account.json").write_text('{"type": "service_account", "private_key": "fake_gcp_key"}')
    (ws / "aws_credentials.ini").write_text("[default]\naws_access_key_id = AKIAFAKE\naws_secret_access_key = fake_secret\n")

    # 6. Legitimate workspace files
    (ws / "README.md").write_text("# Mock Project\nThis is safe documentation.\n")
    (ws / "doc.txt").write_text("Safe public documentation content.\n")
    (ws / "config.json").write_text('{"theme": "dark", "version": "1.0.0"}')
    (ws / ".gitignore").write_text(".env\n.shamu/\n*.key\n")

    src_dir = ws / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("print('Hello world!')\n")

    return ws


# ==============================================================================
# 1. Tier-1 Autonomous Tool Execution (read_file) Denial Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_read_file_rejects_dot_shamu_secrets(secure_workspace: Path):
    """CANON-P0-02: read_file must fail-closed when targeting .shamu/master.key, audit.db, or audit.key."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    shamu_targets = [
        secure_workspace / ".shamu" / "master.key",
        secure_workspace / ".shamu" / "audit.db",
        secure_workspace / ".shamu" / "audit.key",
        ".shamu/master.key",
        r".shamu\audit.db",
    ]

    for target in shamu_targets:
        with pytest.raises(PermissionDeniedError) as exc_info:
            await firewall.execute_tool(
                tool_name="read_file",
                arguments={"path": str(target)},
                session_id="test-session",
                turn_id="turn-1",
            )
        assert "outside approved workspace" in str(exc_info.value) or "protected" in str(exc_info.value)


@pytest.mark.asyncio
async def test_read_file_rejects_dotenv_configurations(secure_workspace: Path):
    """CANON-P0-02: read_file must fail-closed when targeting .env, .env.local, .env.production, etc."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    env_targets = [
        secure_workspace / ".env",
        secure_workspace / ".env.local",
        secure_workspace / ".env.production",
        secure_workspace / "secrets.env",
        ".env",
        ".env.local",
    ]

    for target in env_targets:
        with pytest.raises(PermissionDeniedError) as exc_info:
            await firewall.execute_tool(
                tool_name="read_file",
                arguments={"path": str(target)},
                session_id="test-session",
                turn_id="turn-2",
            )
        assert "outside approved workspace" in str(exc_info.value) or "protected" in str(exc_info.value) or "sensitive" in str(exc_info.value)


@pytest.mark.asyncio
async def test_read_file_rejects_git_repository_internals(secure_workspace: Path):
    """CANON-P0-02: read_file must fail-closed when targeting .git/config, .git/HEAD, etc."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    git_targets = [
        secure_workspace / ".git" / "config",
        secure_workspace / ".git" / "HEAD",
        ".git/config",
        r".git\HEAD",
    ]

    for target in git_targets:
        with pytest.raises(PermissionDeniedError) as exc_info:
            await firewall.execute_tool(
                tool_name="read_file",
                arguments={"path": str(target)},
                session_id="test-session",
                turn_id="turn-3",
            )
        assert "outside approved workspace" in str(exc_info.value) or "protected" in str(exc_info.value)


@pytest.mark.asyncio
async def test_read_file_rejects_private_keys(secure_workspace: Path):
    """CANON-P0-02: read_file must fail-closed when targeting SSH private keys and TLS keyfiles."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    key_targets = [
        secure_workspace / "id_rsa",
        secure_workspace / "id_ed25519",
        secure_workspace / "master.key",
        secure_workspace / "server.pem",
        secure_workspace / "deploy.key",
        "id_rsa",
        "master.key",
    ]

    for target in key_targets:
        with pytest.raises(PermissionDeniedError) as exc_info:
            await firewall.execute_tool(
                tool_name="read_file",
                arguments={"path": str(target)},
                session_id="test-session",
                turn_id="turn-4",
            )
        assert "outside approved workspace" in str(exc_info.value) or "protected" in str(exc_info.value) or "sensitive" in str(exc_info.value)


@pytest.mark.asyncio
async def test_read_file_rejects_credentials_and_service_accounts(secure_workspace: Path):
    """CANON-P0-02: read_file must fail-closed when targeting credentials.json and service accounts."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    cred_targets = [
        secure_workspace / "credentials.json",
        secure_workspace / "service_account.json",
        secure_workspace / "aws_credentials.ini",
        "credentials.json",
    ]

    for target in cred_targets:
        with pytest.raises(PermissionDeniedError) as exc_info:
            await firewall.execute_tool(
                tool_name="read_file",
                arguments={"path": str(target)},
                session_id="test-session",
                turn_id="turn-5",
            )
        assert "outside approved workspace" in str(exc_info.value) or "protected" in str(exc_info.value) or "sensitive" in str(exc_info.value)


# ==============================================================================
# 2. Legitimate Workspace Access Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_read_file_permits_legitimate_workspace_files(secure_workspace: Path):
    """CANON-P0-02: read_file must succeed on legitimate user documentation, code, and non-sensitive configs."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    safe_targets = [
        (secure_workspace / "README.md", "Mock Project"),
        (secure_workspace / "doc.txt", "Safe public documentation content."),
        (secure_workspace / "config.json", "theme"),
        (secure_workspace / ".gitignore", ".env"),
        (secure_workspace / "src" / "main.py", "Hello world!"),
    ]

    for path_obj, expected_substring in safe_targets:
        content = await firewall.execute_tool(
            tool_name="read_file",
            arguments={"path": str(path_obj)},
            session_id="test-session",
            turn_id="turn-safe",
        )
        assert expected_substring in content


# ==============================================================================
# 3. Adversarial Evasion Vector Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_adversarial_traversal_evasion_blocked(secure_workspace: Path):
    """CANON-P0-02: Path traversal attempts to reach secrets via relative segments must fail."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    traversal_attacks = [
        "src/../.env",
        r"src\..\.shamu\master.key",
        "src/../../.git/config",
        r"src\..\..\outside\secret.txt",
        "./.shamu/../.shamu/master.key",
        "src/../id_rsa",
    ]

    for attack in traversal_attacks:
        with pytest.raises(PermissionDeniedError):
            await firewall.execute_tool(
                tool_name="read_file",
                arguments={"path": attack},
                session_id="test-session",
                turn_id="turn-attack",
            )


@pytest.mark.asyncio
async def test_adversarial_case_variance_evasion_blocked(secure_workspace: Path):
    """CANON-P0-02: Windows NTFS case-insensitive variants must all be strictly blocked."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    case_attacks = [
        ".ENV",
        ".Env",
        ".GIT/config",
        ".Git/CONFIG",
        ".SHAMU/MASTER.KEY",
        ".sHaMu/audit.db",
        "ID_RSA",
        "Id_Ed25519",
        "CrEdEnTiAlS.jSoN",
    ]

    for attack in case_attacks:
        with pytest.raises(PermissionDeniedError):
            await firewall.execute_tool(
                tool_name="read_file",
                arguments={"path": attack},
                session_id="test-session",
                turn_id="turn-case",
            )


@pytest.mark.asyncio
async def test_list_directory_rejects_protected_directories(secure_workspace: Path):
    """CANON-P0-02: list_directory must reject enumerating .shamu and .git."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    for protected_dir in [".shamu", ".git", str(secure_workspace / ".shamu"), str(secure_workspace / ".git")]:
        with pytest.raises(PermissionDeniedError):
            await firewall.execute_tool(
                tool_name="list_directory",
                arguments={"path": protected_dir},
                session_id="test-session",
                turn_id="turn-list",
            )


@pytest.mark.asyncio
async def test_write_file_rejects_overwriting_secrets(secure_workspace: Path):
    """CANON-P0-02: write_file must be blocked when attempting to overwrite or plant sensitive files."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    secret_writes = [
        (secure_workspace / ".env", "EVIL_SECRET=true"),
        (secure_workspace / ".shamu" / "master.key", "FORGED_KEY"),
        (secure_workspace / ".git" / "hooks" / "evil.bat", "echo pwned"),
        ("id_rsa", "fake_overwrite"),
    ]

    for path_target, content in secret_writes:
        with pytest.raises(PermissionDeniedError):
            await firewall.execute_tool(
                tool_name="write_file",
                arguments={"path": str(path_target), "content": content},
                session_id="test-session",
                turn_id="turn-write",
            )


@pytest.mark.skipif(not (sys.platform == "win32" and HAS_WINAPI), reason="Real Windows NTFS junction test")
@pytest.mark.asyncio
async def test_ntfs_junction_to_secrets_blocked(secure_workspace: Path):
    """CANON-P0-02: Directory junctions pointing to .shamu must be resolved to physical path and rejected."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    shamu_dir = secure_workspace / ".shamu"
    junction_alias = secure_workspace / "innocent_shamu_alias"
    try:
        _winapi.CreateJunction(str(shamu_dir), str(junction_alias))
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"Junction creation not supported: {e}")

    # Attempt to read master.key through the junction
    junction_path = str(junction_alias / "master.key")
    with pytest.raises(PermissionDeniedError):
        await firewall.execute_tool(
            tool_name="read_file",
            arguments={"path": junction_path},
            session_id="test-session",
            turn_id="turn-junction",
        )


# ==============================================================================
# 4. Audit Logging Verification on Rejection
# ==============================================================================

@pytest.mark.asyncio
async def test_denial_events_recorded_in_audit_ledger(secure_workspace: Path):
    """CANON-P0-02: All rejected secret access attempts must be recorded as DENY / DENIED in audit ledger."""
    firewall = ExecutionFirewall(workspace_root=secure_workspace)

    with pytest.raises(PermissionDeniedError):
        await firewall.execute_tool(
            tool_name="read_file",
            arguments={"path": ".env"},
            session_id="audit-test-session",
            turn_id="turn-audit-1",
        )

    # Verify audit ledger
    verify_res = firewall.audit_ledger.verify_integrity()
    assert verify_res.is_valid is True
    assert verify_res.verified_records >= 1

    # Inspect the logged record from log_path
    import json
    lines = firewall.audit_ledger.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    last_record = json.loads(lines[-1])
    assert last_record["decision"] == "DENY"
    assert last_record["outcome"] == "DENIED"
    assert last_record["tool_name"] == "read_file"
