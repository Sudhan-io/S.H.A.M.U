import os
import sys
import pytest
from pathlib import Path
from shamu.security.path_guard import WindowsPathGuard, PathSecurityViolationError

# Real WinAPI junction support
try:
    import _winapi
    HAS_WINAPI = hasattr(_winapi, "CreateJunction")
except ImportError:
    HAS_WINAPI = False


@pytest.fixture
def workspace_dirs(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    return ws, outside


def test_allowed_path_within_workspace(workspace_dirs):
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    test_file = ws / "subdir" / "file.txt"
    test_file.parent.mkdir()
    test_file.write_text("safe content")

    canon = guard.authorize_path(str(test_file), mode="read")
    assert canon == test_file.resolve()
    assert guard.is_safe_path(str(test_file), mode="read") is True


def test_path_traversal_double_dot(workspace_dirs):
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    traversal_path = str(ws / "subdir" / ".." / ".." / "outside" / "secret.txt")
    with pytest.raises(PathSecurityViolationError, match="outside approved workspace"):
        guard.authorize_path(traversal_path, mode="read")

    assert guard.is_safe_path(traversal_path, mode="read") is False


def test_mixed_case_and_separators(workspace_dirs):
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    test_file = ws / "data.txt"
    test_file.write_text("hello")

    # Change case and separators
    path_str = str(test_file).replace("\\", "/").lower()
    canon = guard.authorize_path(path_str, mode="read")
    assert os.path.normcase(str(canon)) == os.path.normcase(str(test_file.resolve()))


def test_reject_dos_device_names(workspace_dirs):
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    dos_devices = ["CON", "PRN", "AUX", "NUL", "COM1", "COM9", "LPT1", "LPT9", "NUL.txt", "con.dat", "aux.json"]
    for dev in dos_devices:
        target = str(ws / dev)
        with pytest.raises(PathSecurityViolationError, match="reserved DOS device"):
            guard.authorize_path(target, mode="write")
        assert guard.is_safe_path(target, mode="write") is False


def test_reject_alternate_data_streams(workspace_dirs):
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    ads_paths = [
        str(ws / "file.txt:hidden.exe"),
        str(ws / "notes.txt:$DATA"),
        str(ws / "test.txt:stream"),
    ]
    for p in ads_paths:
        with pytest.raises(PathSecurityViolationError, match="Alternate Data Stream"):
            guard.authorize_path(p, mode="write")
        assert guard.is_safe_path(p, mode="write") is False


def test_reject_null_byte_injection(workspace_dirs):
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    null_path = str(ws / "safe.txt") + "\x00.exe"
    with pytest.raises(PathSecurityViolationError, match="null byte"):
        guard.authorize_path(null_path, mode="read")
    assert guard.is_safe_path(null_path, mode="read") is False


def test_reject_unc_and_device_namespaces(workspace_dirs):
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    bad_paths = [
        r"\\remote_server\share\file.txt",
        r"\\?\Volume{b75e2c83-0000-0000-0000-100000000000}\secret.txt",
        r"\\.\PhysicalDrive0",
    ]
    for p in bad_paths:
        with pytest.raises(PathSecurityViolationError, match="UNC or raw device namespace"):
            guard.authorize_path(p, mode="read")
        assert guard.is_safe_path(p, mode="read") is False


@pytest.mark.skipif(not (sys.platform == "win32" and HAS_WINAPI), reason="Real Windows NTFS junction test")
def test_real_ntfs_junction_escape(workspace_dirs):
    ws, outside = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    secret_file = outside / "secret.env"
    secret_file.write_text("TOP_SECRET=true")

    junction_point = ws / "junction_link"
    try:
        # Create a real NTFS junction inside workspace pointing to outside
        _winapi.CreateJunction(str(outside), str(junction_point))
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"Junction creation not supported on filesystem: {e}")

    # Now attempt to read secret.env through the in-workspace junction path
    escaped_path = str(junction_point / "secret.env")

    with pytest.raises(PathSecurityViolationError, match="outside approved workspace"):
        guard.authorize_path(escaped_path, mode="read")

    assert guard.is_safe_path(escaped_path, mode="read") is False


def test_nonexistent_target_within_workspace(workspace_dirs):
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    # File and subfolder do not exist yet
    future_file = ws / "nested" / "future_output.txt"
    canon = guard.authorize_path(str(future_file), mode="write")
    assert canon == (ws.resolve() / "nested" / "future_output.txt")
    assert guard.is_safe_path(str(future_file), mode="write") is True


def test_nonexistent_target_beneath_escaping_ancestor(workspace_dirs):
    ws, outside = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    future_outside = str(ws / ".." / "outside" / "nested" / "new_file.txt")
    with pytest.raises(PathSecurityViolationError, match="outside approved workspace"):
        guard.authorize_path(future_outside, mode="write")


def test_empty_and_whitespace_paths(workspace_dirs):
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    for invalid in ["", "   ", "\t\n"]:
        with pytest.raises(PathSecurityViolationError, match="empty or whitespace"):
            guard.authorize_path(invalid, mode="read")
        assert guard.is_safe_path(invalid, mode="read") is False


def test_reject_extended_dos_devices(workspace_dirs):
    """SEC-09: CONIN$, CONOUT$, COM0, LPT0 must be strictly blocked as reserved devices."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    extended_devices = [
        "CONIN$", "CONOUT$", "conin$", "conout$",
        "COM0", "LPT0", "com0", "lpt0",
        "conin$.txt", "CONOUT$.dat", "COM0.log"
    ]
    for dev in extended_devices:
        target = str(ws / dev)
        with pytest.raises(PathSecurityViolationError, match="reserved DOS device"):
            guard.authorize_path(target, mode="read")
        assert guard.is_safe_path(target, mode="read") is False


def test_reject_trailing_dots_and_spaces(workspace_dirs):
    """NEW-SEC-04: Paths with trailing dots or spaces in components must be rejected to prevent NTFS evasion."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    evasion_paths = [
        str(ws / "file.txt."),
        str(ws / "secret.doc "),
        str(ws / "folder." / "file.txt"),
        str(ws / "dir " / "file.txt"),
    ]
    for p in evasion_paths:
        with pytest.raises(PathSecurityViolationError, match="trailing dot or space"):
            guard.authorize_path(p, mode="write")
        assert guard.is_safe_path(p, mode="write") is False


def test_reject_clock_dos_device(workspace_dirs):
    """Regression test: CLOCK$ reserved DOS device must be strictly blocked."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    clock_devices = [
        "CLOCK$", "clock$", "Clock$",
        "CLOCK$.txt", "clock$.dat", "CLOCK$.log"
    ]
    for dev in clock_devices:
        target = str(ws / dev)
        with pytest.raises(PathSecurityViolationError, match="reserved DOS device"):
            guard.authorize_path(target, mode="read")
        assert guard.is_safe_path(target, mode="read") is False


# ==============================================================================
# CANON-P0-02 Secret Protection & Protected Directory Tests
# ==============================================================================

def test_reject_dot_shamu_directory_and_contents(workspace_dirs):
    """CANON-P0-02: .shamu application state and all nested files must be blocked."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    shamu_dir = ws / ".shamu"
    shamu_dir.mkdir()
    (shamu_dir / "master.key").write_bytes(b"SUPER_SECRET_KEY")
    (shamu_dir / "audit.db").write_text("sqlite format")
    nested_dir = shamu_dir / "checkpoints"
    nested_dir.mkdir()
    (nested_dir / "state.json").write_text("{}")

    shamu_targets = [
        str(shamu_dir),
        str(shamu_dir / "master.key"),
        str(shamu_dir / "audit.db"),
        str(nested_dir / "state.json"),
        ".shamu",
        r".shamu\master.key",
        ".shamu/audit.db",
        r"subdir\..\.shamu\master.key",
    ]
    for target in shamu_targets:
        with pytest.raises(PathSecurityViolationError, match="protected"):
            guard.authorize_path(target, mode="read")
        assert guard.is_safe_path(target, mode="read") is False
        assert guard.is_safe_path(target, mode="write") is False


def test_reject_dot_git_directory_and_contents(workspace_dirs):
    """CANON-P0-02: .git repository state, objects, and hooks must be blocked."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    git_dir = ws / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\nrepositoryformatversion = 0\n")
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "pre-commit").write_text("#!/bin/sh\n")

    git_targets = [
        str(git_dir),
        str(git_dir / "config"),
        str(git_dir / "HEAD"),
        str(hooks_dir / "pre-commit"),
        ".git",
        r".git\config",
        ".git/HEAD",
        r"foo\..\.git\config",
    ]
    for target in git_targets:
        with pytest.raises(PathSecurityViolationError, match="protected"):
            guard.authorize_path(target, mode="read")
        assert guard.is_safe_path(target, mode="read") is False
        assert guard.is_safe_path(target, mode="write") is False


def test_reject_dotenv_variants(workspace_dirs):
    """CANON-P0-02: .env and all variations must be blocked."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    env_files = [
        ".env",
        ".env.local",
        ".env.production",
        ".env.staging",
        ".env.development",
        ".env.vault",
        ".env.backup",
        "secrets.env",
        "app.env",
    ]
    for name in env_files:
        target_file = ws / name
        target_file.write_text("API_SECRET=12345")

        with pytest.raises(PathSecurityViolationError, match="protected|sensitive"):
            guard.authorize_path(str(target_file), mode="read")
        with pytest.raises(PathSecurityViolationError, match="protected|sensitive"):
            guard.authorize_path(name, mode="read")
        assert guard.is_safe_path(str(target_file), mode="read") is False


def test_reject_private_keys_and_certificates(workspace_dirs):
    """CANON-P0-02: Private keys, certificates, and keyfiles must be blocked."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    key_names = [
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
        "master.key",
        "deploy.key",
        "server.pem",
        "client.p12",
        "tls.pkcs12",
        "backup.kdbx",
        "identity.pfx",
    ]
    for name in key_names:
        key_file = ws / name
        key_file.write_bytes(b"-----BEGIN PRIVATE KEY-----")

        with pytest.raises(PathSecurityViolationError, match="protected|sensitive"):
            guard.authorize_path(str(key_file), mode="read")
        with pytest.raises(PathSecurityViolationError, match="protected|sensitive"):
            guard.authorize_path(name, mode="read")
        assert guard.is_safe_path(str(key_file), mode="read") is False


def test_reject_credentials_and_secrets_files(workspace_dirs):
    """CANON-P0-02: Credential config files and secrets must be blocked."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    cred_names = [
        "credentials",
        "credentials.json",
        "credentials.ini",
        "credentials.xml",
        "aws_credentials.ini",
        "service_account.json",
        "service-account.json",
        "secrets.json",
        "secrets.yaml",
        "audit.db",
        "audit.key",
        "audit_secrets.json",
    ]
    for name in cred_names:
        cred_file = ws / name
        cred_file.write_text('{"token": "xyz"}')

        with pytest.raises(PathSecurityViolationError, match="protected|sensitive"):
            guard.authorize_path(str(cred_file), mode="read")
        assert guard.is_safe_path(str(cred_file), mode="read") is False


def test_case_variance_defense_in_depth(workspace_dirs):
    """CANON-P0-02: Mixed-case variants of protected paths must be blocked."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    case_variants = [
        ".ENV",
        ".Env",
        ".eNv.LoCaL",
        ".GIT/config",
        ".Git/HEAD",
        ".SHAMU/MASTER.KEY",
        ".sHaMu/audit.db",
        "ID_RSA",
        "Id_Ed25519",
        "MASTER.KEY",
        "CrEdEnTiAlS.jSoN",
    ]
    for cv in case_variants:
        with pytest.raises(PathSecurityViolationError, match="protected|sensitive"):
            guard.authorize_path(cv, mode="read")
        assert guard.is_safe_path(cv, mode="read") is False


def test_nonexistent_sensitive_targets_rejected_fail_closed(workspace_dirs):
    """CANON-P0-02: Writing nonexistent sensitive files must be blocked before creation."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    future_secrets = [
        ws / ".env",
        ws / ".shamu" / "future.key",
        ws / ".git" / "hooks" / "evil.ps1",
        ws / "stolen_id_rsa",
        ws / "new_master.key",
    ]
    for target in future_secrets:
        assert not target.exists()
        with pytest.raises(PathSecurityViolationError, match="protected|sensitive"):
            guard.authorize_path(str(target), mode="write")
        assert guard.is_safe_path(str(target), mode="write") is False


@pytest.mark.skipif(not (sys.platform == "win32" and HAS_WINAPI), reason="Real Windows NTFS junction test")
def test_junction_to_protected_directory_blocked(workspace_dirs):
    """CANON-P0-02: NTFS junction to .shamu or .git must resolve physically and be blocked."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    real_shamu = ws / ".shamu"
    real_shamu.mkdir()
    secret_key = real_shamu / "master.key"
    secret_key.write_bytes(b"SECRET_KEY_123")

    junction_alias = ws / "innocent_folder"
    try:
        _winapi.CreateJunction(str(real_shamu), str(junction_alias))
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"Junction creation not supported: {e}")

    # Accessing master.key through the innocent_folder junction
    target = str(junction_alias / "master.key")
    with pytest.raises(PathSecurityViolationError, match="protected|sensitive"):
        guard.authorize_path(target, mode="read")
    assert guard.is_safe_path(target, mode="read") is False


def test_legitimate_workspace_files_remain_accessible(workspace_dirs):
    """CANON-P0-02: Legitimate source code, docs, and configs must remain accessible."""
    ws, _ = workspace_dirs
    guard = WindowsPathGuard(workspace_root=ws)

    safe_files = [
        ws / "main.py",
        ws / "doc.txt",
        ws / "config.json",
        ws / "README.md",
        ws / ".gitignore",
        ws / "keyboard.py",
        ws / "src" / "utils" / "helper.py",
    ]
    for sf in safe_files:
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text("legitimate content")

        canon = guard.authorize_path(str(sf), mode="read")
        assert canon == sf.resolve()
        assert guard.is_safe_path(str(sf), mode="read") is True
        assert guard.is_safe_path(str(sf), mode="write") is True




