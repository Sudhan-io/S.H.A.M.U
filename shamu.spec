# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification file for building standalone SHAMU Windows executables.

Builds dual targets:
- SHAMU.exe: Standalone Windows GUI application (no CMD console flash)
- shamu-cli.exe: Standalone Windows CLI application (console mode)
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)
from PyInstaller.utils.hooks.qt import get_qt_library_info

block_cipher = None

project_dir = Path.cwd()
src_dir = str(project_dir / "src")

# ---------------------------------------------------------
# Dynamic Assets & Data Files
# ---------------------------------------------------------
datas = []

# Project data assets
lexicon_file = project_dir / "data" / "lexicon.json"
if lexicon_file.exists():
    datas.append((str(lexicon_file), "data"))
data_dir = project_dir / "data"
if data_dir.exists():
    datas.append((str(data_dir), "data"))

# Dynamic ML / Speech assets (Silero VAD ONNX models, etc.)
datas += collect_data_files("faster_whisper")
datas += collect_data_files("onnxruntime")
datas += collect_data_files("_sounddevice_data")
try:
    datas += collect_data_files("_soundfile_data")
except Exception:
    pass

# Project icons / UI assets if present
assets_dir = project_dir / "assets"
if assets_dir.exists():
    datas.append((str(assets_dir), "assets"))

# Deduplicate datas
datas = list({(str(src), dst): (src, dst) for src, dst in datas}.values())

# ---------------------------------------------------------
# Native Binaries & DLLs
# ---------------------------------------------------------
binaries = []
binaries += collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("_sounddevice_data")
try:
    binaries += collect_dynamic_libs("_soundfile_data")
except Exception:
    pass

# PySide6 Qt plugins (styles, platforms, iconengines, imageformats)
try:
    qt_info = get_qt_library_info("PySide6")
    for plugin in ["styles", "platforms", "iconengines", "imageformats"]:
        binaries += qt_info.collect_plugins(plugin)
except Exception:
    pass

# Deduplicate binaries
binaries = list({(str(src), dst): (src, dst) for src, dst in binaries}.values())

# ---------------------------------------------------------
# Hidden Imports
# ---------------------------------------------------------
hiddenimports = [
    # Core shamu modules
    "shamu",
    "shamu.app",
    "shamu.cli",
    "shamu.config",
    "shamu.protocols",
    "shamu.types",
    "shamu.__main__",
    # Agent orchestrator
    "shamu.agent",
    "shamu.agent.orchestrator",
    # Gateway
    "shamu.gateway",
    "shamu.gateway.ollama_provider",
    "shamu.gateway.openrouter_provider",
    "shamu.gateway.router",
    # Memory
    "shamu.memory",
    "shamu.memory.context_compactor",
    "shamu.memory.embedding",
    "shamu.memory.hybrid",
    "shamu.memory.sqlite_store",
    # Security
    "shamu.security",
    "shamu.security.approval_manager",
    "shamu.security.audit",
    "shamu.security.audit_ledger",
    "shamu.security.firewall",
    "shamu.security.path_guard",
    "shamu.security.process_guard",
    "shamu.security.tool_registry",
    # Speech
    "shamu.speech",
    "shamu.speech.audio_capture",
    "shamu.speech.lexicon",
    "shamu.speech.pipeline",
    "shamu.speech.sentence_splitter",
    "shamu.speech.stt",
    "shamu.speech.tts",
    "shamu.speech.vad",
    # UI
    "shamu.ui",
    "shamu.ui.hotkey",
    "shamu.ui.modal",
    "shamu.ui.theme",
    "shamu.ui.tray",
    "shamu.ui.window",
    # Windows native & PyWin32
    "win32api",
    "win32gui",
    "win32con",
    "win32process",
    "win32security",
    "pywintypes",
    "pythoncom",
    # Keyring & WinVault
    "keyring",
    "keyring.backends",
    "keyring.backends.Windows",
    # Third-party native/runtime libs
    "psutil",
    "uiautomation",
    "sqlite3",
    "sounddevice",
    "soundfile",
    "_sounddevice_data",
    "_soundfile_data",
    "onnxruntime",
    "faster_whisper",
    "ctranslate2",
    "rich",
    "httpx",
    "pydantic",
    "numpy",
    "fastembed",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

# Collect all win32ctypes submodules dynamically
try:
    hiddenimports += collect_submodules("win32ctypes")
except Exception:
    pass

# Collect all shamu submodules
try:
    hiddenimports += collect_submodules("shamu")
except Exception:
    pass

hiddenimports = sorted(list(set(hiddenimports)))

# ---------------------------------------------------------
# GUI Analysis & Executable Target (SHAMU.exe)
# ---------------------------------------------------------
a_gui = Analysis(
    ['src/shamu/app.py'],
    pathex=[src_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook', 'scipy'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz_gui = PYZ(a_gui.pure, a_gui.zipped_data, cipher=block_cipher)

exe_gui = EXE(
    pyz_gui,
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name='SHAMU',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Standalone Windows GUI application (no CMD prompt window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ---------------------------------------------------------
# CLI Analysis & Executable Target (shamu-cli.exe)
# ---------------------------------------------------------
a_cli = Analysis(
    ['src/shamu/cli.py'],
    pathex=[src_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook', 'scipy'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)

exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name='shamu-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Standalone Windows Console CLI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ---------------------------------------------------------
# Unified Distribution Folder
# ---------------------------------------------------------
coll = COLLECT(
    exe_gui,
    a_gui.binaries,
    a_gui.zipfiles,
    a_gui.datas,
    exe_cli,
    a_cli.binaries,
    a_cli.zipfiles,
    a_cli.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SHAMU',
)
