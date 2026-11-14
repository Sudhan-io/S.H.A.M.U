import asyncio
import time
import pytest
from typing import AsyncIterator, List, Optional
from shamu.types import ChatMessage, Role, ChatChunk
from shamu.protocols import SpeechToTextEngine, TextToSpeechEngine, AudioCapture, ModelProvider
from shamu.speech.pipeline import VoicePipeline


class MockSTT(SpeechToTextEngine):
    def __init__(self):
        self.received_prompt = None

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        self.received_prompt = initial_prompt
        return "What is the time?"


class MockTTS(TextToSpeechEngine):
    def __init__(self):
        self.spoken_sentences = []

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        self.spoken_sentences.append(text)
        yield b"AUDIO_BYTES"

    async def stop(self) -> None:
        pass


class MockCapture(AudioCapture):
    async def record_utterance(self, timeout_sec: float = 15.0) -> bytes:
        return b"DUMMY_AUDIO_DATA"

    def is_recording(self) -> bool:
        return False

    def cancel(self) -> None:
        pass


class MockGateway(ModelProvider):
    async def stream_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
        yield ChatChunk(text_delta="It is 10 AM. ")
        yield ChatChunk(text_delta="Have a great day!")
        yield ChatChunk(finish_reason="stop")

    async def complete_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
        pass


@pytest.mark.asyncio
async def test_voice_pipeline_full_turn():
    stt = MockSTT()
    tts = MockTTS()
    gateway = MockGateway()
    capture = MockCapture()

    pipeline = VoicePipeline(
        stt=stt,
        tts=tts,
        gateway=gateway,
        capture=capture,
    )

    history = []
    text_deltas = []
    async for delta in pipeline.run_voice_turn(prompt="What is the time?", history=history):
        text_deltas.append(delta)

    full_response = "".join(text_deltas)
    assert "It is 10 AM." in full_response
    assert "Have a great day!" in full_response

    # Verify TTS received sentences incrementally
    assert len(tts.spoken_sentences) == 2
    assert tts.spoken_sentences[0] == "It is 10 AM."
    assert tts.spoken_sentences[1] == "Have a great day!"


@pytest.mark.asyncio
async def test_voice_pipeline_with_dynamic_lexicon(tmp_path):
    from shamu.speech.lexicon import DynamicLexicon

    lexicon = DynamicLexicon(custom_lexicon_path=tmp_path / "lexicon.json")
    stt = MockSTT()
    tts = MockTTS()
    gateway = MockGateway()
    capture = MockCapture()

    pipeline = VoicePipeline(
        stt=stt,
        tts=tts,
        gateway=gateway,
        capture=capture,
        lexicon=lexicon,
    )

    # Calling listen_and_transcribe should pass initial_prompt built from lexicon
    transcribed = await pipeline.listen_and_transcribe()
    assert transcribed == "What is the time?"
    assert stt.received_prompt is not None
    assert "S.H.A.M.U." in stt.received_prompt


@pytest.mark.asyncio
async def test_voice_pipeline_clause_streaming_ttfa():
    class ClauseGateway(ModelProvider):
        async def stream_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
            # Stream clauses before sentence ends
            yield ChatChunk(text_delta="Scanning all system directories now, ")
            yield ChatChunk(text_delta="this will take only a few moments.")
            yield ChatChunk(finish_reason="stop")

        async def complete_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
            pass

    stt = MockSTT()
    tts = MockTTS()
    gateway = ClauseGateway()
    pipeline = VoicePipeline(stt=stt, tts=tts, gateway=gateway)

    chunks = [c async for c in pipeline.run_voice_turn(prompt="Check files")]
    assert len(chunks) == 2
    # Verify clause was dispatched to TTS before full sentence finished
    assert len(tts.spoken_sentences) == 2
    assert tts.spoken_sentences[0] == "Scanning all system directories now,"
    assert tts.spoken_sentences[1] == "this will take only a few moments."


@pytest.mark.asyncio
async def test_voice_pipeline_audio_chunks_dispatched_to_playback():
    """SPCH-01: Audio chunks yielded by TTS must be forwarded to playback, not discarded."""
    played_chunks = []

    def mock_play(chunk: bytes):
        played_chunks.append(chunk)

    stt = MockSTT()
    tts = MockTTS()
    gateway = MockGateway()
    capture = MockCapture()

    pipeline = VoicePipeline(
        stt=stt,
        tts=tts,
        gateway=gateway,
        capture=capture,
        on_audio_chunk=mock_play,
    )

    async for _ in pipeline.run_voice_turn(prompt="What is the time?"):
        pass

    assert len(played_chunks) == 2
    assert played_chunks == [b"AUDIO_BYTES", b"AUDIO_BYTES"]


@pytest.mark.asyncio
async def test_audio_playback_decoupled_from_stream_chat():
    """Verify that slow audio playback does not block stream_chat generator token emission."""
    playback_started = []
    playback_finished = []

    async def slow_play(chunk: bytes):
        playback_started.append(time.perf_counter())
        await asyncio.sleep(0.12)
        playback_finished.append(time.perf_counter())

    class FastGateway(ModelProvider):
        async def stream_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
            yield ChatChunk(text_delta="Sentence one is here. ")
            yield ChatChunk(text_delta="Sentence two is here. ")
            yield ChatChunk(finish_reason="stop")

        async def complete_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
            pass

    stt = MockSTT()
    tts = MockTTS()
    gateway = FastGateway()
    pipeline = VoicePipeline(
        stt=stt,
        tts=tts,
        gateway=gateway,
        on_audio_chunk=slow_play,
    )

    token_receive_times = []
    start_time = time.perf_counter()
    async for _ in pipeline.run_voice_turn(prompt="Hello"):
        token_receive_times.append(time.perf_counter() - start_time)

    # Both tokens are yielded without waiting for slow_play (each 0.12s)
    # Total playback for 2 chunks is >= 0.24s, but token emission finishes almost instantly
    assert len(token_receive_times) == 2
    assert token_receive_times[-1] < 0.10  # Token streaming was NOT blocked by playback
    assert len(playback_finished) == 2


@pytest.mark.asyncio
async def test_sounddevice_wait_in_background_executor(monkeypatch):
    """Verify that default sounddevice playback with sd.wait() runs in background task without blocking tokens."""
    import sys
    import types

    sd_waits = []

    def mock_play(arr, samplerate):
        pass

    def mock_wait():
        time.sleep(0.1)
        sd_waits.append(True)

    def mock_stop():
        pass

    mock_sd = types.ModuleType("sounddevice")
    mock_sd.play = mock_play
    mock_sd.wait = mock_wait
    mock_sd.stop = mock_stop
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)

    class FastGateway(ModelProvider):
        async def stream_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
            yield ChatChunk(text_delta="Sentence one is complete. ")
            yield ChatChunk(text_delta="Sentence two is complete. ")
            yield ChatChunk(finish_reason="stop")

        async def complete_chat(self, messages, tools=None, temperature=0.7, max_tokens=None):
            pass

    pipeline = VoicePipeline(
        stt=MockSTT(),
        tts=MockTTS(),
        gateway=FastGateway(),
    )

    token_receive_times = []
    start = time.perf_counter()
    async for _ in pipeline.run_voice_turn(prompt="Hi"):
        token_receive_times.append(time.perf_counter() - start)

    # Tokens emitted without waiting for sd.wait()
    assert len(token_receive_times) == 2
    assert token_receive_times[-1] < 0.08
    assert len(sd_waits) == 2


@pytest.mark.asyncio
async def test_interrupt_halts_playback_and_drains_queue():
    """Verify interrupt() halts active playback and flushes pending audio chunks."""
    played_chunks = []

    async def slow_play(chunk: bytes):
        played_chunks.append(chunk)
        await asyncio.sleep(0.15)

    stt = MockSTT()
    tts = MockTTS()
    gateway = MockGateway()
    pipeline = VoicePipeline(
        stt=stt,
        tts=tts,
        gateway=gateway,
        on_audio_chunk=slow_play,
    )

    # Enqueue multiple chunks
    await pipeline.enqueue_audio(b"CHUNK1")
    await pipeline.enqueue_audio(b"CHUNK2")
    await pipeline.enqueue_audio(b"CHUNK3")

    # Brief yield to let worker pick up chunk 1
    await asyncio.sleep(0.01)

    # Interrupt barge-in
    await pipeline.interrupt()

    # Audio queue should be cleared and worker task cancelled
    assert pipeline._audio_queue.empty()
    assert pipeline._playback_task is None
    # Subsequent chunks were drained without playing
    assert len(played_chunks) <= 1


class PipelineMockStream:
    def __init__(self, callback, *args, **kwargs):
        self.callback = callback
        self.active = False
        self.closed = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True
        self.active = False

    def emit_chunk(self, val=0):
        import numpy as np
        data = np.full((512, 1), fill_value=val, dtype=np.int16)
        self.callback(data, 512, None, None)


@pytest.mark.asyncio
async def test_pipeline_listen_and_transcribe_short_speech_vad(monkeypatch):
    """CANON-P0-03: VoicePipeline.listen_and_transcribe terminates promptly on trailing silence."""
    import sounddevice
    from shamu.speech.audio_capture import PushToTalkCapture
    from shamu.speech.vad import VoiceActivityDetector, VADState

    created_streams = []

    def mock_stream_factory(*args, **kwargs):
        stream = PipelineMockStream(kwargs.get("callback"))
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(sounddevice, "InputStream", mock_stream_factory)

    vad = VoiceActivityDetector(sample_rate=16000, trailing_silence_ms=64)
    capture = PushToTalkCapture(sample_rate=16000, block_size=512, vad=vad)
    stream = created_streams[0]

    stt = MockSTT()
    pipeline = VoicePipeline(stt=stt, capture=capture, vad=vad)

    stop_feeder = asyncio.Event()

    async def feeder():
        while not capture.is_recording() and not stop_feeder.is_set():
            await asyncio.sleep(0.005)

        # Emit 3 loud speech chunks
        for _ in range(3):
            if stop_feeder.is_set():
                return
            stream.emit_chunk(val=5000)
            await asyncio.sleep(0.01)

        # Emit silence chunks to trigger trailing silence finish
        while not stop_feeder.is_set():
            stream.emit_chunk(val=0)
            await asyncio.sleep(0.01)

    feeder_task = asyncio.create_task(feeder())

    start_time = time.perf_counter()
    try:
        # Full timeout is 10.0 seconds
        result = await pipeline.listen_and_transcribe(timeout_sec=10.0)
        elapsed = time.perf_counter() - start_time

        # Finished promptly on trailing silence (<< 10.0s)
        assert elapsed < 1.0
        assert result == "What is the time?"
        assert vad.state == VADState.FINISHED
        assert not capture.is_recording()
    finally:
        stop_feeder.set()
        await feeder_task
        capture.close()


@pytest.mark.asyncio
async def test_pipeline_listen_and_transcribe_silence_only(monkeypatch):
    """CANON-P0-03: Silence-only input stops within reasonable limit, no endless stall, returns empty string."""
    import sounddevice
    from shamu.speech.audio_capture import PushToTalkCapture
    from shamu.speech.vad import VoiceActivityDetector, VADState

    created_streams = []

    def mock_stream_factory(*args, **kwargs):
        stream = PipelineMockStream(kwargs.get("callback"))
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(sounddevice, "InputStream", mock_stream_factory)

    vad = VoiceActivityDetector(sample_rate=16000, trailing_silence_ms=64)
    capture = PushToTalkCapture(sample_rate=16000, block_size=512, vad=vad)
    stream = created_streams[0]

    stt = MockSTT()
    pipeline = VoicePipeline(stt=stt, capture=capture, vad=vad)

    stop_feeder = asyncio.Event()

    async def feeder():
        while not stop_feeder.is_set():
            stream.emit_chunk(val=0)
            await asyncio.sleep(0.01)

    feeder_task = asyncio.create_task(feeder())

    start_time = time.perf_counter()
    try:
        # Initial silence limit 0.15s, timeout 5.0s
        result = await pipeline.listen_and_transcribe(timeout_sec=5.0, max_initial_silence_sec=0.15)
        elapsed = time.perf_counter() - start_time

        assert elapsed < 0.8
        assert result == ""
        assert vad.state == VADState.WAITING
        assert not capture.is_recording()
    finally:
        stop_feeder.set()
        await feeder_task
        capture.close()


@pytest.mark.asyncio
async def test_pipeline_listen_and_transcribe_long_speech(monkeypatch):
    """CANON-P0-03: Long speech bounds at timeout_sec and transcribes."""
    import sounddevice
    from shamu.speech.audio_capture import PushToTalkCapture
    from shamu.speech.vad import VoiceActivityDetector, VADState

    created_streams = []

    def mock_stream_factory(*args, **kwargs):
        stream = PipelineMockStream(kwargs.get("callback"))
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(sounddevice, "InputStream", mock_stream_factory)

    vad = VoiceActivityDetector(sample_rate=16000, trailing_silence_ms=600)
    capture = PushToTalkCapture(sample_rate=16000, block_size=512, vad=vad)
    stream = created_streams[0]

    stt = MockSTT()
    pipeline = VoicePipeline(stt=stt, capture=capture, vad=vad)

    stop_feeder = asyncio.Event()

    async def feeder():
        while not capture.is_recording() and not stop_feeder.is_set():
            await asyncio.sleep(0.005)
        while not stop_feeder.is_set():
            stream.emit_chunk(val=5000)
            await asyncio.sleep(0.01)

    feeder_task = asyncio.create_task(feeder())

    start_time = time.perf_counter()
    try:
        result = await pipeline.listen_and_transcribe(timeout_sec=0.25)
        elapsed = time.perf_counter() - start_time

        assert elapsed >= 0.20
        assert elapsed < 0.8
        assert result == "What is the time?"
        assert not capture.is_recording()
        assert vad.state == VADState.SPEAKING
    finally:
        stop_feeder.set()
        await feeder_task
        capture.close()


@pytest.mark.asyncio
async def test_pipeline_listen_and_transcribe_cancellation(monkeypatch):
    """CANON-P0-03: Cancellation of listen_and_transcribe cancels capture cleanly."""
    import sounddevice
    from shamu.speech.audio_capture import PushToTalkCapture

    created_streams = []

    def mock_stream_factory(*args, **kwargs):
        stream = PipelineMockStream(kwargs.get("callback"))
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(sounddevice, "InputStream", mock_stream_factory)

    capture = PushToTalkCapture(sample_rate=16000, block_size=512)
    stream = created_streams[0]

    pipeline = VoicePipeline(stt=MockSTT(), capture=capture)

    stop_feeder = asyncio.Event()

    async def feeder():
        while not stop_feeder.is_set():
            stream.emit_chunk(val=0)
            await asyncio.sleep(0.01)

    feeder_task = asyncio.create_task(feeder())

    try:
        task = asyncio.create_task(pipeline.listen_and_transcribe(timeout_sec=10.0))
        await asyncio.sleep(0.05)
        assert capture.is_recording()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not capture.is_recording()
        assert len(capture._chunks) == 0
    finally:
        stop_feeder.set()
        await feeder_task
        capture.close()

