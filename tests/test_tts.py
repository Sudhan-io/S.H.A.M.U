import pytest
from shamu.speech.sentence_splitter import SentenceSplitter
from shamu.speech.tts import TextToSpeech


def test_sentence_splitter_basic():
    splitter = SentenceSplitter()
    sentences = splitter.feed("Hello world. How are you?")
    assert len(sentences) == 2
    assert sentences[0] == "Hello world."
    assert sentences[1] == "How are you?"

    # Remaining text without punctuation
    sentences2 = splitter.feed(" This is ongoing")
    assert len(sentences2) == 0

    # Flush remaining
    flushed = splitter.flush()
    assert len(flushed) == 1
    assert flushed[0] == "This is ongoing"


def test_sentence_splitter_abbreviations():
    splitter = SentenceSplitter()
    # "Dr. Smith arrived. He is ready."
    sentences = splitter.feed("Dr. Smith arrived. He is ready.")
    assert len(sentences) == 2
    assert sentences[0] == "Dr. Smith arrived."
    assert sentences[1] == "He is ready."


@pytest.mark.asyncio
async def test_tts_synthesize_stream(monkeypatch):
    class MockPiperProcess:
        def __init__(self):
            self.returncode = 0
        async def communicate(self, input):
            return b"WAV_AUDIO_DATA", b""

    tts = TextToSpeech(piper_binary_path="mock_piper.exe")

    async def mock_subprocess(*args, **kwargs):
        return MockPiperProcess()

    import asyncio
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_subprocess)

    chunks = []
    async for chunk in tts.synthesize_stream("Hello world."):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0] == b"WAV_AUDIO_DATA"


@pytest.mark.asyncio
async def test_windows_sapi_safety_and_no_shell_injection(monkeypatch):
    """SEC-08 & NEW-SEC-02: Verify SAPI5 uses stdin piping, no injection in args, and scrubs env."""
    import asyncio

    captured_cmd = None
    captured_stdin = None
    captured_env = None

    class MockSapiProcess:
        def __init__(self, *args, **kwargs):
            nonlocal captured_cmd, captured_env
            captured_cmd = args
            captured_env = kwargs.get("env")
            self.returncode = 0

        async def communicate(self, input=None):
            nonlocal captured_stdin
            captured_stdin = input
            return b"", b""

        def terminate(self):
            pass

        async def wait(self):
            return 0

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-token-12345")
    monkeypatch.setenv("SHAMU_SECRET", "super-secret-password")

    async def mock_exec(*args, **kwargs):
        return MockSapiProcess(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)

    tts = TextToSpeech(piper_binary_path="nonexistent_piper.exe")
    malicious_text = "Testing speech injection $(calc.exe) ' ` ; Start-Process notepad.exe"

    await tts._speak_windows_sapi(malicious_text)

    # 1. Verify text was passed via stdin, not injected into cmdline args
    assert captured_stdin == malicious_text.encode("utf-8")
    for arg in captured_cmd:
        assert "$(calc.exe)" not in str(arg)
        assert "notepad.exe" not in str(arg)

    # 2. Verify sensitive environment variables were stripped (NEW-SEC-02)
    assert captured_env is not None
    assert "OPENROUTER_API_KEY" not in captured_env
    assert "SHAMU_SECRET" not in captured_env
    # Safe system keys should be retained
    assert any(k in captured_env for k in ("SYSTEMROOT", "SystemRoot", "PATH", "Path"))


@pytest.mark.asyncio
async def test_windows_sapi_timeout_handling(monkeypatch):
    """NEW-SEC-03: Verify SAPI5 execution has a timeout and terminates hung process."""
    import asyncio

    terminated = False

    class HangingSapiProcess:
        def __init__(self, *args, **kwargs):
            self.returncode = None

        async def communicate(self, input=None):
            await asyncio.sleep(10.0)
            return b"", b""

        def terminate(self):
            nonlocal terminated
            terminated = True

        async def wait(self):
            return -15

    async def mock_exec(*args, **kwargs):
        return HangingSapiProcess(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)

    tts = TextToSpeech(piper_binary_path="nonexistent_piper.exe")
    # Call with a very short timeout
    await tts._speak_windows_sapi("Hanging test", timeout=0.05)

    assert terminated is True



@pytest.mark.asyncio
async def test_windows_sapi_execution_policy_restricted(monkeypatch):
    """CANON-P1-02: Verify powershell fallback uses -ExecutionPolicy Restricted, never Bypass."""
    import asyncio

    captured_cmd = None

    class MockProcess:
        def __init__(self, *args, **kwargs):
            nonlocal captured_cmd
            captured_cmd = args
            self.returncode = 0

        async def communicate(self, input=None):
            return b"", b""

        def terminate(self):
            pass

        async def wait(self):
            return 0

    async def mock_exec(*args, **kwargs):
        return MockProcess(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)
    tts = TextToSpeech(piper_binary_path="nonexistent.exe")
    await tts._speak_windows_sapi("Testing restricted execution policy")

    assert captured_cmd is not None
    cmd_list = list(captured_cmd)
    assert "-ExecutionPolicy" in cmd_list
    ep_idx = cmd_list.index("-ExecutionPolicy")
    assert cmd_list[ep_idx + 1] == "Restricted"
    assert "Bypass" not in cmd_list
    tts.close()


def test_tts_win32_job_object_lifecycle():
    """CANON-P1-02: Verify TTS initializes and closes Win32 Job Object handle cleanly."""
    import sys

    tts = TextToSpeech()
    if sys.platform == "win32":
        job = tts._ensure_job_object()
        assert job is not None
        assert tts._job_handle == job
        tts.close()
        assert tts._job_handle is None
    else:
        assert tts._ensure_job_object() is None


@pytest.mark.asyncio
async def test_tts_process_attached_to_job_object(monkeypatch):
    """CANON-P1-02: Verify child process is registered with Win32 Job Object and resumed."""
    import asyncio
    import sys

    attached = []
    resumed = []

    class MockProcessWithPid:
        def __init__(self):
            self.pid = 9999
            self.returncode = 0

        async def communicate(self, input=None):
            return b"WAV_DATA", b""

        def terminate(self):
            pass

        async def wait(self):
            return 0

    async def mock_exec(*args, **kwargs):
        return MockProcessWithPid()

    tts = TextToSpeech(piper_binary_path="mock_piper.exe")
    monkeypatch.setattr(tts, "_attach_process_to_job", lambda proc, job: attached.append((proc.pid, job)) or True)
    monkeypatch.setattr(tts, "_resume_process", lambda proc: resumed.append(proc.pid) or True)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)

    chunks = []
    async for chunk in tts.synthesize_stream("Testing process attachment"):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0] == b"WAV_DATA"
    if sys.platform == "win32":
        assert len(attached) == 1
        assert attached[0][0] == 9999
        assert len(resumed) == 1
        assert resumed[0] == 9999
    tts.close()


@pytest.mark.asyncio
async def test_tts_stop_barge_in_cleans_up_process(monkeypatch):
    """CANON-P1-02: Verify stop() (barge-in) immediately cleans up active process and closes transports."""
    terminated = False
    waited = False
    stdin_closed = False
    transport_closed = False

    class MockStdin:
        def is_closing(self):
            return stdin_closed

        def close(self):
            nonlocal stdin_closed
            stdin_closed = True

        async def wait_closed(self):
            pass

    class MockTransport:
        def is_closing(self):
            return transport_closed

        def close(self):
            nonlocal transport_closed
            transport_closed = True

    class MockActiveProcess:
        def __init__(self):
            self.returncode = None
            self.pid = 1234
            self.stdin = MockStdin()
            self._transport = MockTransport()

        def terminate(self):
            nonlocal terminated
            terminated = True

        async def wait(self):
            nonlocal waited
            waited = True
            self.returncode = -15
            return -15

    tts = TextToSpeech()
    proc = MockActiveProcess()
    tts._active_process = proc

    await tts.stop()

    assert tts._stopped is True
    assert tts._active_process is None
    assert terminated is True
    assert waited is True
    assert stdin_closed is True
    assert transport_closed is True
    tts.close()


@pytest.mark.asyncio
async def test_tts_piper_timeout_cleans_up_and_raises(monkeypatch):
    """CANON-P1-02: Verify Piper synthesis timeout terminates child process and cleans pipes."""
    import asyncio
    from shamu.protocols import SpeechProcessingError

    terminated = False
    waited = False

    class HangingPiperProcess:
        def __init__(self):
            self.returncode = None
            self.pid = 5555

        async def communicate(self, input=None):
            await asyncio.sleep(10.0)
            return b"", b""

        def terminate(self):
            nonlocal terminated
            terminated = True

        async def wait(self):
            nonlocal waited
            waited = True
            self.returncode = -15
            return -15

    async def mock_exec(*args, **kwargs):
        return HangingPiperProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)
    tts = TextToSpeech(piper_binary_path="mock_piper.exe")
    monkeypatch.setattr(tts, "_attach_process_to_job", lambda proc, job: True)
    monkeypatch.setattr(tts, "_resume_process", lambda proc: True)

    with pytest.raises(SpeechProcessingError, match="timed out"):
        await tts._synthesize_piper("Hello", timeout=0.05)

    assert terminated is True
    assert waited is True
    tts.close()


@pytest.mark.asyncio
async def test_tts_process_guard_delegation(monkeypatch):
    """CANON-P1-02: Verify custom WindowsProcessGuard integration for env sanitization and job handling."""
    import asyncio
    import sys
    from unittest.mock import MagicMock

    mock_guard = MagicMock()
    mock_guard.build_sanitized_env.return_value = {"SYSTEMROOT": "C:\\Windows", "MOCK_CLEAN": "1"}
    mock_guard._create_job_object.return_value = 12345
    mock_guard._assign_process_to_job.return_value = True
    mock_guard._resume_process_threads.return_value = True

    class MockProc:
        def __init__(self):
            self.pid = 4321
            self.returncode = 0

        async def communicate(self, input=None):
            return b"GUARD_AUDIO", b""

        def terminate(self):
            pass

        async def wait(self):
            return 0

    async def mock_exec(*args, **kwargs):
        return MockProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)

    tts = TextToSpeech(piper_binary_path="mock_piper.exe", process_guard=mock_guard)
    chunks = []
    async for chunk in tts.synthesize_stream("Testing process guard"):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0] == b"GUARD_AUDIO"
    assert mock_guard.build_sanitized_env.called
    if sys.platform == "win32":
        assert mock_guard._create_job_object.called
        assert mock_guard._assign_process_to_job.called
        assert mock_guard._resume_process_threads.called


@pytest.mark.asyncio
async def test_real_windows_sapi_job_object_attachment():
    """CANON-P1-02 (Integration): On Windows, verify actual child process is placed in Win32 Job Object."""
    import sys

    if sys.platform != "win32":
        pytest.skip("Windows-specific integration test")

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    tts = TextToSpeech()
    job_handle = tts._ensure_job_object()
    assert job_handle is not None

    in_job_observed = []

    orig_attach = tts._attach_process_to_job

    def check_and_attach(proc, jh):
        res = orig_attach(proc, jh)
        sub = getattr(proc, "_transport", None)
        h_proc = None
        close_h = False
        if sub and hasattr(sub, "get_extra_info"):
            sub_proc = sub.get_extra_info("subprocess")
            if sub_proc and hasattr(sub_proc, "_handle"):
                h_proc = int(sub_proc._handle)
        if not h_proc and hasattr(proc, "pid"):
            h_proc = kernel32.OpenProcess(0x0400, False, proc.pid)  # PROCESS_QUERY_INFORMATION
            close_h = True

        if h_proc:
            in_job = wintypes.BOOL()
            kernel32.IsProcessInJob(h_proc, jh, ctypes.byref(in_job))
            in_job_observed.append(bool(in_job.value))
            if close_h:
                kernel32.CloseHandle(h_proc)
        return res

    tts._attach_process_to_job = check_and_attach

    await tts._speak_windows_sapi("Test", timeout=5.0)

    assert len(in_job_observed) >= 1
    assert all(in_job_observed)
    tts.close()


@pytest.mark.asyncio
async def test_tts_kokoro_default_selection_and_diagnostics():
    """Verify Kokoro 82M ONNX is selected as the primary default engine when present."""
    tts = TextToSpeech()
    diag = tts.get_diagnostics()

    assert diag["active_engine"] in ("kokoro", "sapi5")
    if diag["kokoro_available"]:
        assert diag["active_engine"] == "kokoro"
        assert tts.sample_rate == 24000
        assert diag["kokoro_model_path"] is not None
        assert diag["kokoro_voices_path"] is not None
    tts.close()


@pytest.mark.asyncio
async def test_tts_kokoro_streaming_synthesis_mock(monkeypatch):
    """Verify Kokoro synthesis streams 16-bit PCM chunks properly."""
    import numpy as np

    class MockKokoroInstance:
        async def create_stream(self, text, voice="af_heart", speed=1.0):
            # Yield two chunks of dummy float32 audio
            yield np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32), 24000
            yield np.array([0.3, -0.3, 0.4, -0.4], dtype=np.float32), 24000

    tts = TextToSpeech(engine="kokoro")
    monkeypatch.setattr(tts, "_get_kokoro", lambda: MockKokoroInstance())

    chunks = []
    async for chunk in tts.synthesize_stream("Hello Kokoro."):
        chunks.append(chunk)

    assert len(chunks) == 2
    # Verify each chunk is raw int16 PCM
    arr1 = np.frombuffer(chunks[0], dtype=np.int16)
    assert len(arr1) == 4
    assert tts.sample_rate == 24000
    tts.close()


@pytest.mark.asyncio
async def test_tts_kokoro_fallback_to_sapi_with_logged_reason(monkeypatch):
    """Verify Kokoro synthesis failures log reasons and explicitly fall back to SAPI5."""
    sapi_called = False
    sapi_text = ""

    async def mock_sapi(text, timeout=15.0):
        nonlocal sapi_called, sapi_text
        sapi_called = True
        sapi_text = text
        return b""

    tts = TextToSpeech(engine="kokoro")
    # Simulate Kokoro engine missing / initialization failure
    monkeypatch.setattr(tts, "_get_kokoro", lambda: None)
    monkeypatch.setattr(tts, "_speak_windows_sapi", mock_sapi)

    chunks = []
    async for chunk in tts.synthesize_stream("Fallback test sentence."):
        chunks.append(chunk)

    assert len(chunks) == 0
    assert sapi_called is True
    assert sapi_text == "Fallback test sentence."
    assert tts.last_fallback_reason is not None
    assert "Kokoro" in tts.last_fallback_reason
    tts.close()


@pytest.mark.asyncio
async def test_tts_kokoro_stop_barge_in(monkeypatch):
    """Verify barge-in stop halts Kokoro streaming generator immediately."""
    import numpy as np
    import asyncio

    class SlowMockKokoro:
        async def create_stream(self, text, voice="af_heart", speed=1.0):
            for i in range(10):
                yield np.array([0.1 * i], dtype=np.float32), 24000
                await asyncio.sleep(0.01)

    tts = TextToSpeech(engine="kokoro")
    monkeypatch.setattr(tts, "_get_kokoro", lambda: SlowMockKokoro())

    chunks = []
    async for chunk in tts.synthesize_stream("Long stream sentence"):
        chunks.append(chunk)
        if len(chunks) == 2:
            await tts.stop()

    assert len(chunks) == 2
    assert tts._stopped is True
    tts.close()


@pytest.mark.asyncio
async def test_tts_explicit_engine_override():
    """Verify explicit engine configuration overrides automatic detection."""
    tts_piper = TextToSpeech(engine="piper")
    assert tts_piper.active_engine == "piper"
    assert tts_piper.sample_rate == 22050
    tts_piper.close()

    tts_sapi = TextToSpeech(engine="sapi5")
    assert tts_sapi.active_engine == "sapi5"
    tts_sapi.close()

    tts_kokoro = TextToSpeech(engine="kokoro")
    assert tts_kokoro.active_engine == "kokoro"
    assert tts_kokoro.sample_rate == 24000
    tts_kokoro.close()


@pytest.mark.asyncio
async def test_tts_real_kokoro_synthesis_if_models_present():
    """Verify real on-disk Kokoro 82M ONNX model synthesis works end-to-end."""
    tts = TextToSpeech()
    if not tts.is_kokoro_available:
        pytest.skip("Kokoro ONNX model files not found on disk")

    assert tts.active_engine == "kokoro"
    chunks = []
    async for chunk in tts.synthesize_stream("SHAMU voice synthesis active."):
        chunks.append(chunk)

    assert len(chunks) >= 1
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 1000
    tts.close()
