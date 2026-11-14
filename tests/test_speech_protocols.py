import pytest
from shamu.protocols import (
    ShamuError,
    AudioDeviceError,
    SpeechProcessingError,
    SpeechToTextEngine,
    TextToSpeechEngine,
    AudioCapture,
)


def test_speech_exceptions():
    err1 = AudioDeviceError("Microphone not found")
    assert isinstance(err1, ShamuError)
    assert "Microphone not found" in str(err1)

    err2 = SpeechProcessingError("Transcription failed")
    assert isinstance(err2, ShamuError)


def test_speech_protocols_runtime_checkable():
    class DummySTT:
        async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
            return "hello"

    class DummyTTS:
        async def synthesize_stream(self, text: str):
            yield b"audio_chunk"

        async def stop(self) -> None:
            pass

    class DummyCapture:
        async def record_utterance(self) -> bytes:
            return b"raw_pcm"

        def is_recording(self) -> bool:
            return False

        def cancel(self) -> None:
            pass

    assert isinstance(DummySTT(), SpeechToTextEngine)
    assert isinstance(DummyTTS(), TextToSpeechEngine)
    assert isinstance(DummyCapture(), AudioCapture)
