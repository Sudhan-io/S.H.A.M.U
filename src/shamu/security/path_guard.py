"""Windows filesystem security guard enforcing handle-based canonicalization, boundary containment, and secret protection."""

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from shamu.protocols import ShamuError

# Windows reserved DOS device names (case-insensitive, with or without extension)
DOS_DEVICE_RE = re.compile(r"^(CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|CLOCK\$|COM[0-9]|LPT[0-9])(\..*)?$", re.IGNORECASE)

# Protected directory names (matched case-insensitively against components)
PROTECTED_DIRECTORY_NAMES = frozenset({
    ".shamu",
    ".git",
    ".ssh",
    ".gnupg",
    ".gpg",
    ".aws",
    ".azure",
    ".gcp",
    ".kube",
    ".docker",
    ".vault",
})

# Protected exact file names (matched case-insensitively)
PROTECTED_EXACT_FILENAMES = frozenset({
    ".env",
    "master.key",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "credentials",
    "credentials.json",
    "credentials.ini",
    "credentials.xml",
    "service_account.json",
    "service-account.json",
    "audit.db",
    "audit.key",
    "audit_ledger.db",
    "audit_secrets",
    "audit_secrets.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "secrets.ini",
    "secrets.toml",
    "htpasswd",
})

# Protected sensitive file extensions (matched case-insensitively)
PROTECTED_EXTENSIONS = frozenset({
    ".key",
    ".pem",
    ".pkcs12",
    ".pfx",
    ".p12",
    ".kdbx",
    ".keystore",
    ".jks",
})

# Protected filename prefixes (matched case-insensitively)
PROTECTED_PREFIXES = (
    ".env.",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
)


class PathSecurityViolationError(ShamuError):
    """Raised when a path fails canonical containment, uses reserved names, attempts traversal, or targets secrets."""
    pass


class WindowsPathGuard:
    """Centralized filesystem path authorizer designed for Windows NTFS/ReFS security semantics."""

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        allowed_roots: Optional[List[Path]] = None,
        protected_dirs: Optional[Iterable[str]] = None,
        protected_files: Optional[Iterable[str]] = None,
        protected_extensions: Optional[Iterable[str]] = None,
        protected_prefixes: Optional[Iterable[str]] = None,
    ) -> None:
        raw_ws = (workspace_root or Path.cwd()).resolve()
        self.workspace_root = Path(self._clean_extended_prefix(os.path.realpath(raw_ws)))

        if allowed_roots:
            self.allowed_roots = [
                Path(self._clean_extended_prefix(os.path.realpath(r)))
                for r in allowed_roots
            ]
        else:
            self.allowed_roots = [self.workspace_root]

        self.protected_dirs: Set[str] = (
            {d.lower() for d in protected_dirs}
            if protected_dirs is not None
            else set(PROTECTED_DIRECTORY_NAMES)
        )
        self.protected_files: Set[str] = (
            {f.lower() for f in protected_files}
            if protected_files is not None
            else set(PROTECTED_EXACT_FILENAMES)
        )
        self.protected_extensions: Set[str] = (
            {e.lower() for e in protected_extensions}
            if protected_extensions is not None
            else set(PROTECTED_EXTENSIONS)
        )
        self.protected_prefixes: Tuple[str, ...] = (
            tuple(p.lower() for p in protected_prefixes)
            if protected_prefixes is not None
            else PROTECTED_PREFIXES
        )

    @staticmethod
    def _clean_extended_prefix(path_str: str) -> str:
        """Strips the Win32 \\?\\ extended prefix if present."""
        if path_str.startswith(r"\\?\UNC\\"):
            return r"\\" + path_str[8:]
        if path_str.startswith(r"\\?\\"):
            return path_str[4:]
        return path_str

    def _validate_syntax(self, path_str: str) -> None:
        """Validates raw string format against null bytes, UNC paths, DOS devices, and alternate streams."""
        if not path_str or not path_str.strip():
            raise PathSecurityViolationError("Path is empty or whitespace.")

        if "\x00" in path_str:
            raise PathSecurityViolationError("Path contains prohibited null byte.")

        # UNC paths (\\server\share) and raw device namespaces (\\.\, \\?\)
        if path_str.startswith(("\\\\", "//")):
            raise PathSecurityViolationError("UNC or raw device namespaces are strictly prohibited.")

        # Alternate Data Streams (ADS): colon anywhere other than single-letter drive specifier (C:)
        clean = path_str
        if len(clean) >= 2 and clean[1] == ":" and clean[0].isalpha():
            clean = clean[2:]
        if ":" in clean:
            raise PathSecurityViolationError(f"Path contains prohibited Alternate Data Stream specifier: '{path_str}'.")

        # Reserved DOS device names and component format validation
        # Normalize separators and inspect each component
        norm_parts = re.split(r"[\\/]+", path_str)
        for part in norm_parts:
            if not part:
                continue

            # Trailing dots or spaces in path components (NTFS silent truncation evasion)
            if part not in (".", "..") and (part.endswith(".") or part.endswith(" ")):
                raise PathSecurityViolationError(f"Path component has prohibited trailing dot or space: '{part}'.")

            if DOS_DEVICE_RE.match(part):
                raise PathSecurityViolationError(f"Path targets reserved DOS device name: '{part}'.")

    def canonicalize(self, path_str: str) -> Path:
        """Resolves target path to its true canonical physical destination, dereferencing symlinks and junctions."""
        self._validate_syntax(path_str)

        raw_path = Path(path_str)
        if not raw_path.is_absolute():
            candidate = self.workspace_root / raw_path
        else:
            candidate = raw_path

        # If the file or directory exists on disk, os.path.realpath calls GetFinalPathNameByHandleW
        # which dereferences all intermediate symlinks, junctions, and short 8.3 aliases.
        if candidate.exists():
            resolved_str = os.path.realpath(str(candidate))
            canonical = Path(self._clean_extended_prefix(resolved_str))
        else:
            # For nonexistent target paths (e.g. creating a new file or directory),
            # walk up to find the nearest existing ancestor, canonicalize that ancestor,
            # and verify the relative non-existent tail.
            curr = candidate
            missing_parts = []
            while not curr.exists() and curr != curr.parent:
                missing_parts.append(curr.name)
                curr = curr.parent

            if not curr.exists():
                raise PathSecurityViolationError(f"Unverifiable path: ancestor directory does not exist for '{path_str}'.")

            canonical_ancestor = Path(self._clean_extended_prefix(os.path.realpath(str(curr))))
            for part in reversed(missing_parts):
                canonical_ancestor = canonical_ancestor / part
            canonical = canonical_ancestor

        return canonical

    def check_sensitive_path(
        self,
        target: Path,
        original_str: Optional[str] = None,
        relative_to: Optional[Path] = None,
    ) -> Optional[str]:
        """Inspects canonical path and original string for protected directories and sensitive secrets.

        Returns:
            A descriptive violation string if the path targets sensitive secrets/directories, or None if safe.
        """
        # 1. Inspect directory components in original_str (defense-in-depth against pre-canonical evasion)
        if original_str:
            norm_parts = re.split(r"[\\/]+", original_str.strip())
            for part in norm_parts:
                part_clean = part.strip().lower()
                if part_clean in self.protected_dirs:
                    return f"input path targets protected directory '{part}'"

        # 2. Inspect path components in canonical path
        # If relative_to is provided, check components relative to the allowed root;
        # otherwise check all components.
        parts_to_check = target.parts
        if relative_to is not None:
            try:
                parts_to_check = target.relative_to(relative_to).parts
            except ValueError:
                parts_to_check = target.parts

        for part in parts_to_check:
            part_clean = part.strip().lower()
            if part_clean in self.protected_dirs:
                return f"targets protected directory '{part}'"

        # 3. Inspect target file name
        target_name = target.name.strip().lower()

        # Exact match
        if target_name in self.protected_files:
            return f"targets sensitive file '{target.name}'"

        # SSH private keys (e.g. id_rsa, id_ed25519, stolen_id_rsa, id_rsa_prod)
        if any(ssh_term in target_name for ssh_term in ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa")):
            return f"targets SSH private key secret ('{target.name}')"

        # Prefixes
        for prefix in self.protected_prefixes:
            if target_name.startswith(prefix):
                return f"targets sensitive file pattern '{prefix}*' ('{target.name}')"

        # Suffix / extension checks
        if target_name.endswith(".env"):
            return f"targets environment configuration secret ('{target.name}')"

        suffix = target.suffix.lower()
        if suffix in self.protected_extensions:
            return f"targets sensitive cryptographic or secret key file (extension '{suffix}')"

        # Specific credential data files: e.g. aws_credentials.ini, service_account.json, audit_secrets.json
        data_exts = (".json", ".ini", ".xml", ".yaml", ".yml", ".toml", ".txt", ".conf", ".cfg", "")
        if target_name.endswith(data_exts):
            if any(term in target_name for term in ("credential", "service_account", "service-account")):
                return f"targets credential configuration file ('{target.name}')"
            if "audit" in target_name and any(term in target_name for term in ("secret", "key", ".db", "token")):
                return f"targets audit secret file ('{target.name}')"
            if "master" in target_name and "key" in target_name:
                return f"targets master key secret ('{target.name}')"

        # 4. Check original_str basename if provided (defense-in-depth)
        if original_str:
            raw_base = Path(original_str.strip()).name.strip().lower()
            if raw_base in self.protected_files:
                return f"targets sensitive file '{raw_base}'"
            for prefix in self.protected_prefixes:
                if raw_base.startswith(prefix):
                    return f"targets sensitive file pattern '{prefix}*' ('{raw_base}')"
            if raw_base.endswith(".env"):
                return f"targets environment configuration secret ('{raw_base}')"
            if Path(original_str).suffix.lower() in self.protected_extensions:
                return f"targets sensitive cryptographic or secret key file (extension '{Path(original_str).suffix}')"

        return None

    def authorize_path(self, path_str: str, mode: str = "read") -> Path:
        """Authorizes and returns canonical Path if contained strictly within allowed roots
        and does not target protected directories or sensitive application secrets.

        Raises:
            PathSecurityViolationError: If path attempts traversal, escapes permitted roots,
                                        or targets protected directories/sensitive files.
        """
        canonical = self.canonicalize(path_str)
        canon_norm = os.path.normcase(str(canonical))

        # 1. Workspace Boundary Containment Verification
        is_contained = False
        matching_root = None
        for root in self.allowed_roots:
            root_norm = os.path.normcase(str(root))
            # Must either match the root exactly or be a direct descendant beneath root
            if canon_norm == root_norm or canon_norm.startswith(root_norm + os.sep):
                is_contained = True
                matching_root = root
                break

        if not is_contained:
            raise PathSecurityViolationError(
                f"Target path '{path_str}' resolves outside approved workspace ({self.workspace_root})."
            )

        # 2. Defense-in-depth: Protected Directory & Sensitive Secret Protection
        violation = self.check_sensitive_path(canonical, original_str=path_str, relative_to=matching_root)
        if violation:
            raise PathSecurityViolationError(
                f"Access to protected path '{path_str}' is strictly prohibited: {violation}."
            )

        return canonical

    def is_safe_path(self, path_str: str, mode: str = "read") -> bool:
        """Returns True if path passes canonical authorization, False otherwise."""
        try:
            self.authorize_path(path_str, mode=mode)
            return True
        except (PathSecurityViolationError, Exception):
            return False

