"""Automated PyInstaller build script for compiling SHAMU into a native Windows .exe."""

import os
import subprocess
import sys
from pathlib import Path


def build():
    root = Path(__file__).resolve().parent.parent
    spec_file = root / "shamu.spec"

    print("==================================================")
    print(" Compiling SHAMU to Native Windows .exe App")
    print(f" Spec file: {spec_file}")
    print("==================================================")

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_file),
    ]

    result = subprocess.run(pyinstaller_cmd, cwd=str(root))
    if result.returncode != 0:
        print(f"\n[ERROR] Build failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    output_exe = root / "dist" / "SHAMU" / "SHAMU.exe"
    print("\n==================================================")
    print(" Build Succeeded!")
    print(f" Executable: {output_exe}")
    print("==================================================")


if __name__ == "__main__":
    build()

