# Contributing to S.H.A.M.U.

Thank you for your interest in contributing to S.H.A.M.U. (Secure Heuristic Autonomous Machine Utility)!

S.H.A.M.U. is an open-source, local-first personal AI desktop assistant engineered for Windows 10/11. We welcome community bug reports, feature proposals, documentation improvements, and pull requests.

---

## 1. Prerequisites & Development Environment

Because S.H.A.M.U. integrates deeply with Windows platform APIs (`pywin32`, Windows Job Objects, DPAPI keyring, and PySide6), the development environment requires:

- **Operating System:** Windows 10 (Build 19041+) or Windows 11 64-bit.
- **Python:** Python >= 3.10 (64-bit).
- **Git:** Standard Git for Windows.
- **Local LLM Runner (Optional but Recommended):** [Ollama](https://ollama.com/) running locally for live inference testing.

---

## 2. Setting Up Your Development Environment

1. **Clone the Repository:**
   ```powershell
   git clone https://github.com/sudhan-io/S.H.A.M.U.git
   cd S.H.A.M.U
   ```

2. **Create and Activate a Virtual Environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   - **Full Installation (GUI, Speech, Memory, Dev & Test Tools):**
     ```powershell
     pip install -e ".[all]"
     ```
   - **Minimal Headless / Development Only:**
     ```powershell
     pip install -e ".[dev]"
     ```

---

## 3. Running the Automated Test Suite

All tests in S.H.A.M.U. are engineered to run completely offline with mocked endpoints, synthetic test fixtures, and simulated tool environments.

Run the test suite from the repository root:

```powershell
# Run the entire test suite
pytest

# Run tests with concise output
pytest -q

# Run a specific test module (e.g., Security Firewall)
pytest tests/test_firewall_integration.py -v
```

Before submitting a pull request, ensure that **100% of tests pass** with zero failures or unexpected regressions.

---

## 4. Code Contribution Expectations

To maintain code quality and security boundaries:

- **Security-First Architecture:** Any modifications involving tool execution, filesystem access, or external command execution must route through `ExecutionFirewall`, `WindowsPathGuard`, or `WindowsProcessGuard`. Never bypass security boundaries.
- **Offline Test Integrity:** Tests must never make live external network calls (e.g., live Ollama or OpenRouter requests). Use mocks, test doubles, or loopback fixtures.
- **No Hardcoded Secrets or Credentials:** Never commit live API keys, tokens, personal filesystem paths (`C:\Users\...`), or private credentials.
- **Type Annotations & Documentation:** Provide Python type annotations (`typing`) and clear docstrings for public classes and functions.
- **Platform Compatibility:** Maintain compatibility with Windows 10/11 64-bit and dynamic `onedir` packaging in `shamu.spec`.

---

## 5. Submitting a Pull Request

1. **Fork and Branch:** Create a focused feature or fix branch from the repository's published default branch:
   ```powershell
   git checkout -b feature/your-feature-name
   ```
2. **Make Focused Changes:** Keep pull requests focused on a single issue, enhancement, or bugfix.
3. **Verify Locally:**
   - Run the full test suite (`pytest`).
   - Verify that no untracked files or temporary build artifacts are left in the repository (`git status`).
4. **Open a Pull Request:**
   - Provide a clear summary of what the PR does and why.
   - Reference any relevant issues.
   - Include test evidence showing that all unit and integration tests pass.

---

## 6. Security Vulnerabilities

If you discover a security vulnerability, please **do not open a public issue**. Follow the reporting procedure in [SECURITY.md](SECURITY.md) to disclose the issue privately.
