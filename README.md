# S.H.A.M.U. — Secure Heuristic Autonomous Machine Utility

**S.H.A.M.U.** is an open-source, local-first personal AI desktop assistant engineered specifically for **Windows 10 and Windows 11**. 

Engineered with a local-first architecture, S.H.A.M.U. runs large language models on local hardware (via Ollama) with optional cloud failover, alongside neural speech recognition and high-fidelity text-to-speech synthesis. Actual hardware utilization depends on the selected model, runtime, and system configuration. All autonomous actions are governed by an interactive 3-Tier Security Firewall designed to prevent unrestricted shell execution.

---

## Current Development Status

> **Development State:** **Active Pre-1.0 Development (Alpha)**  
> S.H.A.M.U. is actively developed and unfinished. While its core desktop GUI, task planner, voice pipeline, and security firewall are fully functional and supported by an automated test suite with over 580 passing tests, APIs and internal architectures may evolve prior to a tagged stable release. It is currently suitable for developers, security researchers, and early adopters running Windows 10/11 64-bit.

---

## What S.H.A.M.U. Does

- **Native Windows Desktop Experience:** Runs as a floating, dark-themed PySide6 desktop window with a global hotkey (`Alt+Space`), system tray residency, and push-to-talk voice capture.
- **Local-First Model Routing:** Streams conversational responses from local Ollama instances (`qwen2.5:7b-instruct-q4_K_M` by default), with automatic failover to cloud providers (OpenRouter) if the local provider is unreachable, times out, or exceeds context limits, as well as explicit local-only or cloud-only routing policies.
- **Autonomous Task Planning & Execution:** Decomposes complex user requests into directed acyclic graph (DAG) task plans via `TaskPlanner`, executing steps with topological dependency resolution and SQLite checkpoint state recovery.
- **3-Tier Interactive Security Firewall:** Regulates model-generated tool calls through authorization tiers: Tier 1 read-only actions run autonomously, Tier 2 reversible actions are logged to a cryptographic audit ledger, and Tier 3 high-risk operations require explicit human confirmation via an interactive modal prompt.
- **Tiered Memory & Context Compaction:** Maintains a clear separation across memory tiers: an active working conversation context bounded by rolling token compaction; an SQLite FTS5 full-text lexical archive for historical pruned conversation turns; and a curated declarative memory store with FastEmbed semantic embeddings and Reciprocal Rank Fusion (RRF) hybrid retrieval. Conversation archive turns are indexed for lexical search without requiring vector embeddings.
- **Asymmetric Neural Voice Pipeline:** Transcribes user speech via Faster-Whisper with Silero Voice Activity Detection (VAD), and synthesizes spoken responses using Kokoro-82M ONNX (24kHz neural synthesis) with automatic fallbacks to Piper and Windows SAPI5.

---

## Architecture Overview

```
                      ┌───────────────────────────────────────────┐
                      │    PySide6 Desktop GUI / CLI REPL         │
                      │   (Floating Window, System Tray, Hotkey)  │
                      └─────────────────────┬─────────────────────┘
                                            │ User Prompts / Voice Input
                                            ▼
                      ┌───────────────────────────────────────────┐
                      │       Cognitive Agent Orchestrator        │
                      │  (Turn Loop, Memory Injection, Compactor) │
                      └───────┬───────────────────────────┬───────┘
                              │                           │
                 Model Calls  │                           │ Tool Requests
                              ▼                           ▼
┌───────────────────────────────────────────┐   ┌───────────────────────────────────────────┐
│           Model Gateway Router            │   │         3-Tier Security Firewall          │
├─────────────────────┬─────────────────────┤   ├───────────────────────────────────────────┤
│    Local Ollama     │   Cloud OpenRouter  │   │ Tier 1: Autonomous (read_file, list_dir)  │
│  (Local inference)  │ (gemini-2.5-flash)  │   │ Tier 2: Audited (write_file, edit_file)   │
│ (Streaming/Failover)│ (Keyring / Fallback)│   │ Tier 3: Human Approval (powershell_exec)  │
└─────────────────────┴─────────────────────┘   └─────────────┬─────────────────────────────┘
                                                              │
                                                              ▼
                                                ┌───────────────────────────┐
                                                │      Execution Engine     │
                                                │ - WindowsPathGuard        │
                                                │ - WindowsProcessGuard     │
                                                │   (Win32 Job Objects)     │
                                                │ - SHA-256 AuditLedger     │
                                                └───────────────────────────┘
```

---

## Technology Stack

- **Platform:** Windows 10 (Build 19041+) / Windows 11 (64-bit)
- **Runtime:** Python >= 3.10 (64-bit)
- **Desktop UI:** PySide6 (Qt 6 for Python), ctypes / Win32 API integration (`RegisterHotKey`)
- **CLI Interface:** Click, Rich terminal rendering, Argparse
- **Local LLM Inference:** [Ollama](https://ollama.com/) (OpenAI-compatible local HTTP streaming)
- **Cloud Fallback Gateway:** OpenRouter API via HTTPX; credentials stored in Windows Credential Manager via Keyring (DPAPI)
- **Speech Recognition (STT):** Faster-Whisper (INT8 quantized CT2 models on CPU) + Silero VAD
- **Speech Synthesis (TTS):** Kokoro-82M ONNX (primary 24kHz neural TTS), with fallbacks to Piper ONNX and Windows PowerShell SAPI5
- **Task Planner & DAG:** Custom topological task scheduler with SQLite checkpoint storage
- **Memory & Archival:** Active conversation context compactor, SQLite FTS5 full-text conversation archive, and curated declarative memory with FastEmbed (BGE-small-en-v1.5) embeddings & Reciprocal Rank Fusion (RRF) hybrid search
- **Process Isolation:** Windows Job Objects (2 GB memory ceiling for tool execution, 1 GB for TTS child processes, execution timeouts, and process-tree termination via `KILL_ON_JOB_CLOSE`)
- **Packaging:** PyInstaller `onedir` dual-target distribution (`SHAMU.exe` GUI and `shamu-cli.exe` CLI)

---

## Installation & Setup

### Prerequisites

1. **Windows 10 / 11 64-bit**
2. **Python >= 3.10** installed and available on `PATH`.
3. **Ollama (Required for Local Inference; Optional for Cloud-Only Mode):**
   - **Local & Auto-Fallback Modes:** Install from [ollama.com](https://ollama.com/), start the service, and pull the default model:
     ```powershell
     ollama pull qwen2.5:7b-instruct-q4_K_M
     ```
   - **Cloud-Only Mode:** If operating strictly in cloud mode (`shamu --cloud`), local Ollama installation is not required, provided an OpenRouter API key is configured.

### 1. Clone & Set Up Virtual Environment

```powershell
git clone https://github.com/sudhan-io/S.H.A.M.U.git
cd S.H.A.M.U

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Depending on your intended use case, install the appropriate dependency group:

- **Complete Installation (Desktop GUI, Voice, Memory & Dev Tools):**
  ```powershell
  pip install -e ".[all]"
  ```

- **Desktop GUI Only (No Speech / Vector Dependencies):**
  ```powershell
  pip install -e ".[desktop]"
  ```

- **Minimal CLI & Testing Only:**
  ```powershell
  pip install -e ".[dev]"
  ```

### 3. Speech Model Setup (Optional for Neural Voice)

If using the `[speech]` optional extra for Kokoro-82M ONNX neural speech:
1. Ensure `kokoro-v1.0.onnx` and `voices-v1.0.bin` are placed in `~/.shamu/models/` (or `models/` in the repository root).
2. If Kokoro model files are not found, S.H.A.M.U. automatically falls back to Piper (if installed) or Windows SAPI5 PowerShell speech.

---

## Usage

### Desktop GUI Application (`shamu-gui`)

Launch the native PySide6 desktop interface:

```powershell
shamu-gui
```

- **Global Hotkey:** Press `Alt+Space` to summon or minimize the floating window.
- **Push-to-Talk (PTT):** Click the microphone button to initiate voice recording with Silero VAD.
- **System Tray:** Minimize to the system tray for non-intrusive background availability.
- **Security Confirmation:** Destructive operations display an interactive confirmation dialog with the exact proposed tool arguments.

### Interactive Terminal REPL (`shamu`)

Launch the terminal REPL:

```powershell
# Interactive prompt
shamu

# Force offline (disable cloud fallback)
shamu --offline

# Force cloud-only mode (bypasses local Ollama entirely)
shamu --cloud

# Execute a single prompt and exit
shamu -p "Summarize project status"

# Execute an autonomous DAG task plan
shamu -p "/plan Refactor the audio configuration module"

# Launch with voice interaction enabled
shamu -v
```

---

## Configuration

Settings are stored in `~/.shamu/config.json`. If missing, default settings are created automatically.

### Key Settings:
- `local_model`: Ollama model name (default: `"qwen2.5:7b-instruct-q4_K_M"`).
- `fallback_model`: OpenRouter model name (default: `"google/gemini-2.5-flash"`).
- `ollama_url`: URL for local Ollama instance (default: `"http://localhost:11434"`).
- `workspace_root`: Approved workspace root path for tool filesystem operations.

### Cloud Provider API Keys:
If using OpenRouter cloud fallback, provide your API key via environment variable:
```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```
Alternatively, store it securely in Windows Credential Manager via the Python CLI helper or keyring.

---

## Running the Automated Test Suite

S.H.A.M.U. maintains an extensive automated test suite with full offline execution:

```powershell
# Run the full test suite
pytest

# Run tests concisely
pytest -q

# Run specific subsystem tests
pytest tests/test_firewall_integration.py -v
pytest tests/test_process_guard.py -v
pytest tests/test_tasks.py -v
```

---

## Security Architecture & Boundaries

1. **Workspace Confinement (`WindowsPathGuard`):** Path resolution validates canonical, real paths against the configured `workspace_root`. Traversal sequences (`..`), hidden drive prefixes, and NTFS streams (`:stream`) are rejected before filesystem APIs are reached.
2. **Process Resource Sandboxing & Lifecycle (`WindowsProcessGuard` & Speech TTS):**
   - **Tool Subprocess Sandboxing (`WindowsProcessGuard`):** Tool commands run inside an ephemeral Windows Job Object created per command with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and a 2 GB job memory ceiling. The Job Object terminates all associated child processes when the command finishes or times out (default 30 seconds) and the job handle is closed.
   - **Speech Subprocess Lifecycle (`TextToSpeech`):** Fallback speech synthesis subprocesses (Piper and PowerShell SAPI5) run inside a persistent instance Job Object configured with a 1 GB memory ceiling and a 4-process limit (`ActiveProcessLimit = 4`). When an individual synthesis times out, the active subprocess is directly terminated via `process.kill()` and awaited, while the instance Job Object handle remains open across utterances until the assistant shuts down.
3. **PowerShell AST Allowlist:** PowerShell tool execution is strictly validated via the PowerShell Abstract Syntax Tree (AST) parser to block secondary shells, reflection, dynamic invocations (`Invoke-Expression`), and unapproved cmdlets.
4. **Audit Ledger:** Every tool invocation, status update, and execution outcome is appended to an append-only, SHA-256 chained hash ledger at `.shamu/audit/audit.jsonl`.
5. **Responsible Disclosure:** For vulnerability reporting instructions, review [SECURITY.md](SECURITY.md).

---

## Known Limitations

- **Windows Platform Requirement:** Relies on Win32 platform features (Job Objects, `pywin32`, Windows Credential Manager). Linux and macOS environments are not supported.
- **Local Model Prerequisite:** Operating in local-only or auto-fallback modes requires an active Ollama service running locally with the configured model weights. Cloud-only routing (`--cloud`) does not require local Ollama.
- **Model Acquisition:** Neural TTS and local LLM weights must be downloaded separately due to model distribution licensing.

---

## Development Roadmap

- [x] Phase 1: Local LLM Gateway routing and streaming REPL.
- [x] Phase 1: 3-Tier Security Firewall and Win32 Job Object process guards.
- [x] Phase 1: Native PySide6 Desktop GUI, hotkey, and system tray integration.
- [x] Phase 1: DAG Task Planner, topological execution, and SQLite checkpointing.
- [x] Phase 1: Asymmetric voice pipeline (Faster-Whisper STT, Silero VAD, Kokoro ONNX TTS).
- [ ] Phase 2: PyInstaller binary release build pipeline automation.
- [ ] Phase 2: Windows UI automation tool integration (`uiautomation`).
- [ ] Phase 2: Multi-modal vision integration for desktop screen understanding.

---

## License & Attribution

- **S.H.A.M.U. Codebase:** Licensed under the [MIT License](LICENSE). Copyright (c) 2026 Sudhan D.
- **PySide6 (Qt for Python):** S.H.A.M.U. uses PySide6, licensed under the [GNU Lesser General Public License version 3.0 (LGPLv3)](https://www.gnu.org/licenses/lgpl-3.0.html). In binary distributions, standalone shared libraries are packaged dynamically in a separate directory (`_internal/PySide6/`), and users must be provided the ability to relink or replace those libraries. Upstream source code is available at [The Qt Project](https://code.qt.io/cgit/pyside/pyside-setup.git). Applicable third-party license and distribution obligations must be respected.
- **Third-Party Dependencies:** Refer to individual dependencies for their respective licenses (e.g. Apache-2.0, BSD-3-Clause, MIT, MPL-2.0).
