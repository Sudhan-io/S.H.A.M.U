"""Text-to-Speech Engine supporting Kokoro 82M ONNX (primary), Piper neural synthesis, and Windows SAPI5 fallback."""

import asyncio
import ctypes
from ctypes import wintypes
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, AsyncIterator, Dict, Optional

import numpy as np

from shamu.protocols import TextToSpeechEngine, SpeechProcessingError

logger = logging.getLogger(__name__)

try:
    from shamu.security.process_guard import WindowsProcessGuard
except ImportError:
    WindowsProcessGuard = None  # type: ignore

# WinAPI constants for Job Object management and thread suspension
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JobObjectExtendedLimitInformation = 9
CREATE_SUSPENDED = 0x00000004
TH32CS_SNAPTHREAD = 0x00000004
THREAD_SUSPEND_RESUME = 0x0002

if sys.platform == "win32":

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", ctypes.c_long),
            ("tpDeltaPri", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
        ]


def _get_safe_subprocess_env(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Returns a scrubbed environment map omitting ambient secrets and API keys."""
    safe_keys = {
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "TEMP",
        "TMP",
        "PATH",
        "USERPROFILE",
        "LOCALAPPDATA",
        "COMSPEC",
        "PATHEXT",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "COMMONPROGRAMFILES",
    }
    secret_patterns = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "CREDENTIAL")
    clean_env: Dict[str, str] = {}
    for k, v in os.environ.items():
        k_upper = k.upper()
        if k_upper in safe_keys and not any(sec in k_upper for sec in secret_patterns):
            clean_env[k] = v
    if extra_env:
        for k, v in extra_env.items():
            k_upper = k.upper()
            if not any(sec in k_upper for sec in secret_patterns):
                clean_env[k] = v
    return clean_env


def _find_kokoro_model_path(custom_path: Optional[str] = None) -> Optional[Path]:
    """Finds Kokoro ONNX model file from explicit argument, config, or standard locations."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p.resolve()
        candidate = Path.home() / ".shamu" / p
        if candidate.is_file():
            return candidate.resolve()
        return None

    try:
        from shamu.config import get_config
        cfg_path = get_config().speech.resolved_tts_model_path
        if cfg_path and cfg_path.is_file():
            return cfg_path.resolve()
    except Exception:
        pass

    candidates = [
        Path.home() / ".shamu" / "models" / "kokoro-v1.0.onnx",
        Path.home() / ".shamu" / "models" / "kokoro-v1.0.int8.onnx",
        Path.home() / ".shamu" / "models" / "kokoro.onnx",
        Path("models/kokoro-v1.0.onnx"),
        Path("models/kokoro.onnx"),
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def _find_kokoro_voices_path(custom_path: Optional[str] = None) -> Optional[Path]:
    """Finds Kokoro voices bin file from explicit argument, config, or standard locations."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p.resolve()
        candidate = Path.home() / ".shamu" / p
        if candidate.is_file():
            return candidate.resolve()
        return None

    try:
        from shamu.config import get_config
        cfg_path = get_config().speech.resolved_tts_voices_path
        if cfg_path and cfg_path.is_file():
            return cfg_path.resolve()
    except Exception:
        pass

    candidates = [
        Path.home() / ".shamu" / "models" / "voices-v1.0.bin",
        Path.home() / ".shamu" / "models" / "voices.bin",
        Path("models/voices-v1.0.bin"),
        Path("models/voices.bin"),
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


class TextToSpeech(TextToSpeechEngine):
    """Synthesizes text to speech using Kokoro 82M ONNX (primary), Piper, or Windows SAPI5 (fallback).

    Enforces Win32 Job Object lifecycle limits (KILL_ON_JOB_CLOSE, 1 GB memory ceiling)
    and deterministic pipe transport cleanup to eliminate process leaks.
    """

    def __init__(
        self,
        piper_binary_path: Optional[str] = None,
        model_path: Optional[str] = None,
        voices_path: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        sample_rate: Optional[int] = None,
        engine: Optional[str] = None,
        process_guard: Optional[Any] = None,
        job_handle: Optional[Any] = None,
    ):
        try:
            from shamu.config import get_config
            cfg = get_config().speech
        except Exception:
            cfg = None

        self.piper_path = piper_binary_path or "piper.exe"
        self.model_path = model_path or (str(cfg.resolved_tts_model_path) if cfg and cfg.resolved_tts_model_path else None)
        self.voices_path = voices_path or (str(cfg.resolved_tts_voices_path) if cfg and cfg.resolved_tts_voices_path else None)
        self.voice = voice or (cfg.tts_voice if cfg else "af_heart")
        self.speed = speed if speed is not None else (cfg.tts_speed if cfg else 1.0)
        self.last_fallback_reason: Optional[str] = None
        self._kokoro_instance: Optional[Any] = None

        # Resolve primary active engine
        if engine is not None:
            self.active_engine = engine.lower()
        elif piper_binary_path is not None:
            # Caller explicitly configured Piper (preserves unit test compatibility)
            self.active_engine = "piper"
        else:
            # Auto-selection: Kokoro 82M ONNX primary default if present
            kokoro_m = _find_kokoro_model_path(self.model_path)
            kokoro_v = _find_kokoro_voices_path(self.voices_path)
            if kokoro_m and kokoro_v:
                self.active_engine = "kokoro"
                self.model_path = str(kokoro_m)
                self.voices_path = str(kokoro_v)
            elif Path(self.piper_path).is_file() or bool(shutil.which(self.piper_path)):
                self.active_engine = "piper"
            else:
                self.active_engine = "sapi5"
                self.last_fallback_reason = "Kokoro 82M ONNX model files and Piper binary not found"
                logger.info("TextToSpeech: Kokoro ONNX model files not found; defaulting to SAPI5 (%s)", self.last_fallback_reason)

        # Configure sample rate (24000 for Kokoro, 22050 for Piper, or user override)
        if sample_rate is not None:
            self.sample_rate = sample_rate
        elif self.active_engine == "kokoro":
            self.sample_rate = 24000
        else:
            self.sample_rate = 22050

        if process_guard is not None:
            self.process_guard = process_guard
        else:
            try:
                self.process_guard = WindowsProcessGuard() if WindowsProcessGuard is not None else None
            except Exception:
                self.process_guard = None

        self._active_process: Optional[asyncio.subprocess.Process] = None
        self._stopped = False
        self._job_handle: Optional[Any] = job_handle

    @property
    def is_kokoro_available(self) -> bool:
        """Checks if Kokoro ONNX dependencies and model files are present."""
        try:
            import kokoro_onnx  # noqa: F401
            m = _find_kokoro_model_path(self.model_path)
            v = _find_kokoro_voices_path(self.voices_path)
            return bool(m and v)
        except Exception:
            return False

    @property
    def is_piper_available(self) -> bool:
        """Checks if Piper executable is present on disk or in PATH."""
        if not self.piper_path:
            return False
        return Path(self.piper_path).is_file() or bool(shutil.which(self.piper_path))

    @property
    def is_sapi5_available(self) -> bool:
        """Checks if Windows SAPI5 PowerShell synthesis is supported."""
        return sys.platform == "win32"

    def _get_kokoro(self) -> Optional[Any]:
        """Lazy-loads and caches Kokoro ONNX inference engine."""
        if self._kokoro_instance is not None:
            return self._kokoro_instance
        try:
            import kokoro_onnx
            m_path = _find_kokoro_model_path(self.model_path)
            v_path = _find_kokoro_voices_path(self.voices_path)
            if not m_path or not v_path:
                reason = f"Kokoro ONNX model files missing (model={m_path}, voices={v_path})"
                logger.warning("TextToSpeech: %s", reason)
                self.last_fallback_reason = reason
                return None
            self._kokoro_instance = kokoro_onnx.Kokoro(str(m_path), str(v_path))
            self.model_path = str(m_path)
            self.voices_path = str(v_path)
            logger.info("TextToSpeech: Kokoro 82M ONNX initialized successfully (model=%s, voices=%s)", m_path, v_path)
            return self._kokoro_instance
        except Exception as e:
            reason = f"Kokoro ONNX initialization error: {e}"
            logger.warning("TextToSpeech: %s", reason)
            self.last_fallback_reason = reason
            return None

    def _ensure_job_object(self) -> Optional[Any]:
        """Creates and configures a Windows Job Object with KILL_ON_JOB_CLOSE and memory limit."""
        if sys.platform != "win32":
            return None
        if self._job_handle:
            return self._job_handle

        if self.process_guard and hasattr(self.process_guard, "_create_job_object"):
            try:
                handle = self.process_guard._create_job_object()
                if handle:
                    self._job_handle = handle
                    return self._job_handle
            except Exception:
                pass

        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return None

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            # KILL_ON_JOB_CLOSE (0x2000) | JOB_MEMORY (0x200) | ACTIVE_PROCESS (0x8)
            info.BasicLimitInformation.LimitFlags = (
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | JOB_OBJECT_LIMIT_JOB_MEMORY
                | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            )
            info.JobMemoryLimit = 1024 * 1024 * 1024  # 1 GB ceiling for TTS child processes
            info.BasicLimitInformation.ActiveProcessLimit = 4

            res = kernel32.SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not res:
                kernel32.CloseHandle(job)
                return None
            self._job_handle = job
            return self._job_handle
        except Exception:
            return None

    def _attach_process_to_job(self, proc: Any, job_handle: Any) -> bool:
        """Assigns child process to the active Job Object."""
        if sys.platform != "win32" or not job_handle:
            return True
        if not hasattr(proc, "pid") or not isinstance(proc.pid, int) or proc.pid <= 0:
            return True

        if self.process_guard and hasattr(self.process_guard, "_assign_process_to_job"):
            try:
                if self.process_guard._assign_process_to_job(job_handle, proc.pid):
                    return True
            except Exception:
                pass

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        assigned = False
        try:
            sub = getattr(proc, "_transport", None)
            if sub and hasattr(sub, "get_extra_info"):
                sub_proc = sub.get_extra_info("subprocess")
                if sub_proc and hasattr(sub_proc, "_handle") and sub_proc._handle:
                    assigned = bool(kernel32.AssignProcessToJobObject(job_handle, int(sub_proc._handle)))
        except Exception:
            assigned = False

        if not assigned:
            h_proc = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, proc.pid)
            if h_proc:
                try:
                    assigned = bool(kernel32.AssignProcessToJobObject(job_handle, h_proc))
                finally:
                    kernel32.CloseHandle(h_proc)

        return assigned

    def _resume_process(self, proc: Any) -> bool:
        """Resumes suspended process threads after Job Object assignment."""
        if sys.platform != "win32":
            return True
        if not hasattr(proc, "pid") or not isinstance(proc.pid, int) or proc.pid <= 0:
            return True

        if self.process_guard and hasattr(self.process_guard, "_resume_process_threads"):
            try:
                return bool(self.process_guard._resume_process_threads(proc.pid))
            except Exception:
                pass

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        if snap == -1 or snap == 0xFFFFFFFFFFFFFFFF:
            return False
        try:
            entry = THREADENTRY32()
            entry.dwSize = ctypes.sizeof(THREADENTRY32)
            res = kernel32.Thread32First(snap, ctypes.byref(entry))
            while res:
                if entry.th32OwnerProcessID == proc.pid:
                    h_thread = kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                    if h_thread:
                        try:
                            kernel32.ResumeThread(h_thread)
                        finally:
                            kernel32.CloseHandle(h_thread)
                res = kernel32.Thread32Next(snap, ctypes.byref(entry))
            return True
        finally:
            kernel32.CloseHandle(snap)

    def _get_env(self) -> Dict[str, str]:
        """Constructs an ambient secret-free environment for TTS child execution."""
        if self.process_guard and hasattr(self.process_guard, "build_sanitized_env"):
            try:
                return self.process_guard.build_sanitized_env()
            except Exception:
                pass
        return _get_safe_subprocess_env()

    async def _cleanup_process(self, process: Optional[Any], force_kill: bool = False) -> None:
        """Deterministically terminates child process and closes all pipes, transports, and handles."""
        if process is None:
            return

        # 1. Terminate or Kill if still running
        if hasattr(process, "returncode") and process.returncode is None:
            try:
                if force_kill and hasattr(process, "kill") and not hasattr(process, "terminate"):
                    process.kill()
                elif hasattr(process, "terminate"):
                    process.terminate()
                elif hasattr(process, "kill"):
                    process.kill()
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    if hasattr(process, "kill"):
                        process.kill()
                except Exception:
                    pass

            if hasattr(process, "wait"):
                try:
                    await asyncio.wait_for(process.wait(), timeout=1.5)
                except Exception:
                    try:
                        if hasattr(process, "kill"):
                            process.kill()
                        await asyncio.wait_for(process.wait(), timeout=1.0)
                    except Exception:
                        pass

        # 2. Close stdin stream writer if present
        stdin = getattr(process, "stdin", None)
        if stdin is not None and hasattr(stdin, "close"):
            try:
                is_closing = getattr(stdin, "is_closing", lambda: False)()
                if not is_closing:
                    stdin.close()
                    if hasattr(stdin, "wait_closed"):
                        try:
                            await asyncio.wait_for(stdin.wait_closed(), timeout=0.5)
                        except Exception:
                            pass
            except Exception:
                pass

        # 3. Close pipe transports to eliminate unclosed transport warnings
        transport = getattr(process, "_transport", None)
        if transport is not None and hasattr(transport, "close"):
            try:
                is_closing = getattr(transport, "is_closing", lambda: False)()
                if not is_closing:
                    transport.close()
            except Exception:
                pass

    async def _synthesize_piper(self, text: str, timeout: float = 15.0) -> bytes:
        """Executes Piper as a clean OS subprocess under Win32 Job Object sandboxing."""
        cmd = [self.piper_path]
        if self.model_path:
            cmd.extend(["--model", self.model_path])
        cmd.append("--output-raw")

        job_handle = self._ensure_job_object()
        creationflags = CREATE_SUSPENDED if (sys.platform == "win32" and job_handle) else 0

        process: Optional[asyncio.subprocess.Process] = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._get_env(),
                creationflags=creationflags,
            )
            self._active_process = process

            if sys.platform == "win32" and job_handle:
                assigned = self._attach_process_to_job(process, job_handle)
                if not assigned:
                    await self._cleanup_process(process, force_kill=True)
                    raise SpeechProcessingError("Failed to attach Piper TTS child process to Windows Job Object")
                if not self._resume_process(process):
                    await self._cleanup_process(process, force_kill=True)
                    raise SpeechProcessingError("Failed to resume suspended Piper TTS child process")

            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(input=text.encode("utf-8")),
                    timeout=timeout,
                )
                return stdout
            except asyncio.TimeoutError:
                await self._cleanup_process(process, force_kill=True)
                raise SpeechProcessingError("Piper TTS synthesis timed out")
            finally:
                await self._cleanup_process(process, force_kill=False)
                self._active_process = None
        except Exception as e:
            if process is not None:
                await self._cleanup_process(process, force_kill=True)
            self._active_process = None
            if isinstance(e, SpeechProcessingError):
                raise
            raise SpeechProcessingError(f"Piper TTS synthesis failed: {e}") from e

    async def _speak_windows_sapi(self, text: str, timeout: float = 15.0) -> bytes:
        """Fallback synthesis using native Windows System.Speech synthesizer under Win32 Job Object sandboxing."""
        if not text.strip():
            return b""

        # Pass a fixed script that reads text from stdin to prevent PowerShell injection
        ps_cmd = (
            "$ErrorActionPreference = 'Stop'; "
            "[Console]::InputEncoding = [System.Text.Encoding]::UTF8; "
            "Add-Type -AssemblyName System.Speech; "
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$synthText = [Console]::In.ReadToEnd(); "
            "if ($synthText) { $synth.Speak($synthText) }"
        )

        job_handle = self._ensure_job_object()
        creationflags = CREATE_SUSPENDED if (sys.platform == "win32" and job_handle) else 0

        process: Optional[asyncio.subprocess.Process] = None
        try:
            process = await asyncio.create_subprocess_exec(
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Restricted",
                "-Command",
                ps_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._get_env(),
                creationflags=creationflags,
            )
            self._active_process = process

            if sys.platform == "win32" and job_handle:
                assigned = self._attach_process_to_job(process, job_handle)
                if not assigned:
                    await self._cleanup_process(process, force_kill=True)
                    return b""
                if not self._resume_process(process):
                    await self._cleanup_process(process, force_kill=True)
                    return b""

            try:
                await asyncio.wait_for(
                    process.communicate(input=text.encode("utf-8")),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                await self._cleanup_process(process, force_kill=True)
            finally:
                await self._cleanup_process(process, force_kill=False)
                self._active_process = None
            return b""
        except Exception:
            if process is not None:
                await self._cleanup_process(process, force_kill=True)
            self._active_process = None
            return b""

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Synthesizes text and yields audio bytes.

        Hierarchy:
        1. Kokoro 82M ONNX (primary neural engine when present/configured).
        2. Piper neural synthesis (secondary engine if binary is present).
        3. Windows SAPI5 (explicit native fallback with logged diagnostic reason).
        """
        if not text.strip():
            return

        self._stopped = False

        # 1. Kokoro 82M ONNX Primary Synthesis
        if self.active_engine == "kokoro" or (self.active_engine is None and self.is_kokoro_available):
            kokoro = self._get_kokoro()
            if kokoro is not None:
                try:
                    chunks_yielded = 0
                    stream = kokoro.create_stream(text, voice=self.voice, speed=self.speed)
                    async for samples, sr in stream:
                        if self._stopped:
                            return
                        if samples is not None and len(samples) > 0:
                            int16_data = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
                            chunks_yielded += 1
                            yield int16_data.tobytes()
                    if chunks_yielded > 0:
                        return
                except Exception as e:
                    self.last_fallback_reason = f"Kokoro ONNX synthesis failed: {e}"
                    logger.warning("TextToSpeech: %s", self.last_fallback_reason)

        # 2. Piper Secondary Synthesis
        if self.active_engine == "piper" or self.is_piper_available:
            try:
                audio_bytes = await self._synthesize_piper(text)
                if not self._stopped and audio_bytes:
                    yield audio_bytes
                    return
            except Exception as e:
                self.last_fallback_reason = f"Piper synthesis failed: {e}"
                logger.warning("TextToSpeech: %s", self.last_fallback_reason)

        # 3. Explicit Fallback: Windows SAPI5
        if not self._stopped:
            if not self.last_fallback_reason:
                self.last_fallback_reason = f"Neither Kokoro 82M ONNX nor Piper is available (active={self.active_engine})"
            logger.warning("TextToSpeech: Falling back to Windows SAPI5 TTS: %s", self.last_fallback_reason)
            await self._speak_windows_sapi(text)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Provides full diagnostic visibility into TTS engine status, models, and fallback history."""
        m_path = _find_kokoro_model_path(self.model_path)
        v_path = _find_kokoro_voices_path(self.voices_path)
        return {
            "active_engine": self.active_engine,
            "sample_rate": self.sample_rate,
            "voice": self.voice,
            "speed": self.speed,
            "kokoro_available": bool(m_path and v_path),
            "kokoro_model_path": str(m_path) if m_path else None,
            "kokoro_voices_path": str(v_path) if v_path else None,
            "piper_available": self.is_piper_available,
            "piper_path": self.piper_path,
            "sapi5_available": self.is_sapi5_available,
            "last_fallback_reason": self.last_fallback_reason,
        }

    async def stop(self) -> None:
        """Cancels active speech output immediately (Barge-In) and deterministic cleanup."""
        self._stopped = True
        proc = self._active_process
        self._active_process = None
        if proc is not None:
            await self._cleanup_process(proc, force_kill=True)

    def close(self) -> None:
        """Releases the Win32 Job Object handle and frees underlying resources."""
        if sys.platform == "win32" and self._job_handle:
            try:
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.CloseHandle(self._job_handle)
            except Exception:
                pass
            self._job_handle = None

    def __del__(self) -> None:
        self.close()

    async def __aenter__(self) -> "TextToSpeech":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()
        self.close()
