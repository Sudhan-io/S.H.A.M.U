# Security Policy

## Security Overview & Project Maturity

S.H.A.M.U. (Secure Heuristic Autonomous Machine Utility) is an **actively developed, pre-1.0 open-source project**.

Because S.H.A.M.U. functions as a local desktop assistant with autonomous tool execution, task planning, and filesystem capabilities, defensive boundaries are designed into the core architecture:
- **3-Tier Execution Firewall:** Categorizes tool actions into Tier 1 (Read-only/autonomous), Tier 2 (Reversible/audited), and Tier 3 (Destructive/interactive human confirmation required).
- **Filesystem Isolation (`WindowsPathGuard`):** Enforces canonical path containment to the designated workspace root, rejecting path traversal attempts, NTFS Alternate Data Streams (`:stream`), and symlink attacks.
- **Process Resource Controls (`WindowsProcessGuard`):** Sandboxes subprocess commands using native Windows Job Objects (configurable execution timeout, 2 GB job memory ceiling for tool execution, and child process-tree termination).
- **Cryptographic Audit Ledger:** Records tool requests, approval responses, and execution outcomes in a tamper-evident SHA-256 chained hash ledger (`.shamu/audit/audit.jsonl`).

**Security Notice:** S.H.A.M.U. is under active development and has not undergone formal third-party security certification. It is not designed or evaluated for multi-tenant servers or untrusted enterprise environments; it is intended for personal desktop use on Windows 10/11.

---

## Supported Versions

Security fixes and defensive updates are provided exclusively for the **latest published development version** on the default branch of this repository. Historical revisions, development checkpoints, and unreleased feature branches are not maintained. Users and contributors should always track the latest revision.

---

## Reporting a Vulnerability

We take the security of local desktop automation seriously. If you discover a vulnerability or security defect, please report it **privately** so it can be investigated and resolved before public disclosure.

### Intended Reporting Channel
- **GitHub Private Vulnerability Reporting:** Once enabled by the repository owner, please submit vulnerability reports privately via the repository's **Security** tab under **Advisories** -> **Report a vulnerability** on GitHub.

> **Note for Project Owner:** GitHub Private Vulnerability Reporting must be enabled in the repository's repository settings under **Settings** -> **Code security and analysis** -> **Private vulnerability reporting** before public launch.

### Responsible Disclosure Guidelines
- **Do not open public GitHub issues or discussions** for suspected vulnerabilities.
- Provide a clear technical summary, steps to reproduce under safe conditions, and the specific subsystem involved (`ExecutionFirewall`, `WindowsPathGuard`, `WindowsProcessGuard`, or `AgentOrchestrator`).
- Allow the maintainer reasonable time to review, reproduce, and release a fix before publishing details.
- Avoid accessing, destroying, or modifying local user data during research.

Thank you for helping keep S.H.A.M.U. and its users safe.
