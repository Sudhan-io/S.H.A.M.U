import pytest
import numpy as np
from shamu.speech.stt import FasterWhisperSTT


class MockSegment:
    def __init__(self, text):
        self.text = text


class MockWhisperModel:
    def __init__(self, *args, **kwargs):
        assert kwargs.get("device") == "cpu"
        assert kwargs.get("compute_type") == "int8"

    def transcribe(self, audio_array, **kwargs):
        return [MockSegment(" Hello from Faster Whisper")], None


@pytest.mark.asyncio
async def test_stt_transcription_success(monkeypatch):
    monkeypatch.setattr(
        "shamu.speech.stt.WhisperModel",
        MockWhisperModel
    )

    stt = FasterWhisperSTT(model_size="base.en", device="cpu", compute_type="int8")
    dummy_audio = np.zeros(16000, dtype=np.int16).tobytes()

    text = await stt.transcribe(dummy_audio)
    assert text == "Hello from Faster Whisper"


@pytest.mark.asyncio
async def test_stt_empty_audio():
    stt = FasterWhisperSTT()
    text = await stt.transcribe(b"")
    assert text == ""


@pytest.mark.asyncio
async def test_stt_transcription_with_initial_prompt(monkeypatch):
    captured_kwargs = {}

    class CapturingWhisperModel:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)

        def transcribe(self, audio_array, **kwargs):
            captured_kwargs.update(kwargs)
            return [MockSegment(" Hello with S.H.A.M.U.")], None

    monkeypatch.setattr("shamu.speech.stt.WhisperModel", CapturingWhisperModel)

    stt = FasterWhisperSTT(model_size="large-v3-turbo", p_cores=[0, 1, 2, 3])
    dummy_audio = np.zeros(16000, dtype=np.int16).tobytes()
    prompt = "Technical conversation discussing PySide6 and S.H.A.M.U."

    text = await stt.transcribe(dummy_audio, initial_prompt=prompt)
    assert text == "Hello with S.H.A.M.U."
    assert captured_kwargs.get("initial_prompt") == prompt


def test_stt_affinity_safe_fallback(monkeypatch):
    stt = FasterWhisperSTT(p_cores=[999, 1000])  # Non-existent cores
    # Should not raise exception
    stt._apply_cpu_affinity()


@pytest.mark.asyncio
async def test_stt_no_process_cpu_affinity_called(monkeypatch):
    """ST-01: Verify psutil.Process().cpu_affinity is NEVER called during STT inference.
    Process-wide affinity pins the whole python process including Qt GUI thread and asyncio loop."""
    affinity_called = False

    class MockProcess:
        def cpu_affinity(self, *args, **kwargs):
            nonlocal affinity_called
            affinity_called = True
            raise AssertionError("psutil.Process().cpu_affinity should never be called!")

    import shamu.speech.stt as stt_module
    if stt_module.psutil is not None:
        monkeypatch.setattr(stt_module.psutil, "Process", lambda: MockProcess())

    monkeypatch.setattr("shamu.speech.stt.WhisperModel", MockWhisperModel)

    stt = FasterWhisperSTT(model_size="base.en", p_cores=[0, 1, 2, 3], cpu_threads=4)
    # Explicitly test _apply_cpu_affinity
    stt._apply_cpu_affinity()
    assert not affinity_called

    # Test transcribe
    dummy_audio = np.zeros(16000, dtype=np.int16).tobytes()
    res = await stt.transcribe(dummy_audio)
    assert res == "Hello from Faster Whisper"
    assert not affinity_called


def test_stt_ctranslate2_cpu_threads_configuration(monkeypatch):
    """ST-01: Verify CTranslate2 cpu_threads is passed to WhisperModel without whole-process affinity."""
    captured_init_kwargs = {}

    class CapturingModel:
        def __init__(self, model_size, **kwargs):
            captured_init_kwargs.update(kwargs)

    monkeypatch.setattr("shamu.speech.stt.WhisperModel", CapturingModel)

    stt = FasterWhisperSTT(cpu_threads=6)
    model = stt._get_model()
    assert model is not None
    assert captured_init_kwargs.get("cpu_threads") == 6


