import pytest
import numpy as np
from shamu.protocols import AudioDeviceError
from shamu.speech.audio_capture import PushToTalkCapture, _cleanup_all_captures


class MockInputStream:
    def __init__(self, *args, callback=None, **kwargs):
        self.callback = callback or (args[0] if args else None) or kwargs.get("callback")
        self.active = False
        self.closed = False

    def start(self):
        self.active = True
        # Simulate incoming 16kHz audio chunk of 512 samples
        data = np.zeros((512, 1), dtype=np.int16)
        if self.callback:
            self.callback(data, 512, None, None)

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True
        self.active = False


class ControllableMockStream:
    def __init__(self, *args, callback=None, **kwargs):
        self.callback = callback or (args[0] if args else None) or kwargs.get("callback")
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
        data = np.full((512, 1), fill_value=val, dtype=np.int16)
        if self.callback:
            self.callback(data, 512, None, None)


@pytest.fixture(autouse=True)
def default_mock_sounddevice(monkeypatch):
    """Ensure sounddevice.InputStream defaults to mock to prevent unclosed hardware PortAudio streams."""
    import sounddevice
    monkeypatch.setattr(sounddevice, "InputStream", ControllableMockStream)
    yield
    _cleanup_all_captures()


def test_audio_capture_start_stop(monkeypatch):
    import sounddevice
    monkeypatch.setattr(sounddevice, "InputStream", MockInputStream)

    with PushToTalkCapture(sample_rate=16000) as capture:
        assert not capture.is_recording()

        capture.start()
        assert capture.is_recording()

        audio_bytes = capture.stop()
        assert not capture.is_recording()
        assert len(audio_bytes) == 1024  # 512 samples * 2 bytes/sample


def test_audio_capture_device_error(monkeypatch):
    import sounddevice

    class FailingStream:
        def __init__(self, *args, **kwargs):
            raise sounddevice.PortAudioError("No default input device available")

    monkeypatch.setattr(sounddevice, "InputStream", FailingStream)

    capture = PushToTalkCapture()
    try:
        with pytest.raises(AudioDeviceError) as exc_info:
            capture.start()
        assert "No default input device" in str(exc_info.value)
    finally:
        capture.close()


def test_audio_capture_pre_speech_ring_buffer():
    with PushToTalkCapture(sample_rate=16000, block_size=512, pre_buffer_ms=200) as capture:
        assert capture.pre_buffer_blocks >= 6

        # Feed 10 dummy blocks while idle/not recording
        for i in range(10):
            data = np.full((512, 1), fill_value=i, dtype=np.int16)
            capture._audio_callback(data, 512, None, None)

        # When start() is called, the last pre_buffer_blocks must be prepended
        capture.start()
        assert len(capture._chunks) == capture.pre_buffer_blocks
        first_block = np.frombuffer(capture._chunks[0], dtype=np.int16)
        expected_val = 10 - capture.pre_buffer_blocks
        assert first_block[0] == expected_val

        audio = capture.stop()
        assert len(audio) == capture.pre_buffer_blocks * 512 * 2


def test_audio_capture_thread_safety():
    """SPCH-06: Ensure concurrent callbacks and start/stop calls do not corrupt internal buffers."""
    import threading

    with PushToTalkCapture(sample_rate=16000, block_size=256) as capture:
        stop_event = threading.Event()

        def feeder():
            dummy_data = np.zeros((256, 1), dtype=np.int16)
            while not stop_event.is_set():
                capture._audio_callback(dummy_data, 256, None, None)

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()

        for _ in range(50):
            capture.start()
            capture.stop()

        stop_event.set()
        thread.join(timeout=1.0)
        assert not capture.is_recording()


def test_audio_capture_persistent_stream_preserves_pre_speech(monkeypatch):
    """SP-01: Verify stream is persistently active and pre-speech ring buffer is populated
    while idle, preventing speech clipping upon start()."""
    import sounddevice

    created_streams = []

    def mock_stream_factory(*args, **kwargs):
        stream = ControllableMockStream(kwargs.get("callback"))
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(sounddevice, "InputStream", mock_stream_factory)

    capture = PushToTalkCapture(sample_rate=16000, block_size=512, pre_buffer_ms=200)
    assert len(created_streams) == 1
    stream = created_streams[0]
    assert stream.active  # Started during init for background pre-speech buffering

    # Emit 5 chunks while idle (pre-speech)
    for i in range(5):
        stream.emit_chunk(val=100 + i)

    assert len(capture._ring_buffer) == 5

    # Activate Push-to-Talk
    capture.start()
    assert capture.is_recording()
    # No new stream should have been created (no 150-350ms OS re-init latency)
    assert len(created_streams) == 1

    # Emit speech chunk while recording
    stream.emit_chunk(val=999)

    # Pre-speech chunks + speech chunk should be present
    assert len(capture._chunks) == 6

    # Stop Push-to-Talk
    audio = capture.stop()
    assert not capture.is_recording()
    # Stream must remain active in background for next utterance
    assert stream.active
    assert not stream.closed
    assert len(audio) == 6 * 512 * 2

    # Subsequent idle chunk should populate ring buffer again
    stream.emit_chunk(val=200)
    assert len(capture._ring_buffer) == 1

    # Cleanup
    capture.close()
    assert stream.closed


@pytest.mark.asyncio
async def test_record_utterance_silence_only(monkeypatch):
    """CANON-P0-03 Synthetic Test: Silence only stops within reasonable limit, no endless stall."""
    import time
    import asyncio
    import sounddevice
    from shamu.speech.vad import VoiceActivityDetector, VADState

    created_streams = []

    def mock_stream_factory(*args, **kwargs):
        stream = ControllableMockStream(kwargs.get("callback"))
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(sounddevice, "InputStream", mock_stream_factory)

    vad = VoiceActivityDetector(sample_rate=16000, trailing_silence_ms=64)
    capture = PushToTalkCapture(sample_rate=16000, block_size=512, vad=vad)
    stream = created_streams[0]

    stop_feeder = asyncio.Event()

    async def feeder():
        while not stop_feeder.is_set():
            stream.emit_chunk(val=0)  # Silence (RMS = 0)
            await asyncio.sleep(0.01)

    feeder_task = asyncio.create_task(feeder())

    start_time = time.perf_counter()
    try:
        # Timeout set to 5.0s, but max_initial_silence_sec is 0.15s
        audio = await capture.record_utterance(timeout_sec=5.0, max_initial_silence_sec=0.15)
        elapsed = time.perf_counter() - start_time

        # Must stop promptly when initial silence exceeds threshold, well before 5.0s
        assert elapsed < 0.8
        assert audio == b""
        assert not capture.is_recording()
        assert vad.state == VADState.WAITING
    finally:
        stop_feeder.set()
        await feeder_task
        capture.close()


@pytest.mark.asyncio
async def test_record_utterance_short_speech_terminates_promptly(monkeypatch):
    """CANON-P0-03 Synthetic Test: Short speech followed by silence terminates promptly,
    does NOT wait for full timeout."""
    import time
    import asyncio
    import sounddevice
    from shamu.speech.vad import VoiceActivityDetector, VADState

    created_streams = []

    def mock_stream_factory(*args, **kwargs):
        stream = ControllableMockStream(kwargs.get("callback"))
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(sounddevice, "InputStream", mock_stream_factory)

    # 64ms trailing silence = 2 consecutive silent frames of 32ms
    vad = VoiceActivityDetector(sample_rate=16000, trailing_silence_ms=64)
    capture = PushToTalkCapture(sample_rate=16000, block_size=512, vad=vad)
    stream = created_streams[0]

    stop_feeder = asyncio.Event()

    async def feeder():
        # Step 1: Wait until recording is active
        while not capture.is_recording() and not stop_feeder.is_set():
            await asyncio.sleep(0.005)

        # Step 2: Emit 3 loud speech chunks
        for _ in range(3):
            if stop_feeder.is_set():
                return
            stream.emit_chunk(val=5000)  # RMS 5000 -> speech prob 0.95
            await asyncio.sleep(0.01)

        # Step 3: Emit trailing silence chunks until stopped
        while not stop_feeder.is_set():
            stream.emit_chunk(val=0)  # Silence -> will trigger VADState.FINISHED
            await asyncio.sleep(0.01)

    feeder_task = asyncio.create_task(feeder())

    start_time = time.perf_counter()
    try:
        # Full timeout is 10.0s
        audio = await capture.record_utterance(timeout_sec=10.0)
        elapsed = time.perf_counter() - start_time

        # Must terminate promptly (<< 10.0s) as soon as trailing silence condition is met
        assert elapsed < 1.0
        assert len(audio) > 0
        assert not capture.is_recording()
        assert vad.state == VADState.FINISHED
    finally:
        stop_feeder.set()
        await feeder_task
        capture.close()


@pytest.mark.asyncio
async def test_record_utterance_long_speech_reaches_timeout(monkeypatch):
    """CANON-P0-03 Synthetic Test: Continuous long speech records up to timeout_sec upper bound."""
    import time
    import asyncio
    import sounddevice
    from shamu.speech.vad import VoiceActivityDetector, VADState

    created_streams = []

    def mock_stream_factory(*args, **kwargs):
        stream = ControllableMockStream(kwargs.get("callback"))
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(sounddevice, "InputStream", mock_stream_factory)

    vad = VoiceActivityDetector(sample_rate=16000, trailing_silence_ms=600)
    capture = PushToTalkCapture(sample_rate=16000, block_size=512, vad=vad)
    stream = created_streams[0]

    stop_feeder = asyncio.Event()

    async def feeder():
        while not capture.is_recording() and not stop_feeder.is_set():
            await asyncio.sleep(0.005)

        # Emit continuous speech without trailing silence
        while not stop_feeder.is_set():
            stream.emit_chunk(val=5000)
            await asyncio.sleep(0.01)

    feeder_task = asyncio.create_task(feeder())

    start_time = time.perf_counter()
    try:
        # Upper bound limit is 0.25 seconds
        audio = await capture.record_utterance(timeout_sec=0.25)
        elapsed = time.perf_counter() - start_time

        # Upper bound limit reached
        assert elapsed >= 0.20
        assert elapsed < 0.8
        assert len(audio) > 0
        assert not capture.is_recording()
        assert vad.state == VADState.SPEAKING
    finally:
        stop_feeder.set()
        await feeder_task
        capture.close()


@pytest.mark.asyncio
async def test_record_utterance_cancellation_stops_stream(monkeypatch):
    """CANON-P0-03 Synthetic Test: Cancellation stops stream, cleans up state, and releases buffers."""
    import asyncio
    import sounddevice

    created_streams = []

    def mock_stream_factory(*args, **kwargs):
        stream = ControllableMockStream(kwargs.get("callback"))
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(sounddevice, "InputStream", mock_stream_factory)

    capture = PushToTalkCapture(sample_rate=16000, block_size=512)
    stream = created_streams[0]

    stop_feeder = asyncio.Event()

    async def feeder():
        while not stop_feeder.is_set():
            stream.emit_chunk(val=100)
            await asyncio.sleep(0.01)

    feeder_task = asyncio.create_task(feeder())

    try:
        record_task = asyncio.create_task(capture.record_utterance(timeout_sec=10.0))
        await asyncio.sleep(0.05)
        assert capture.is_recording()

        # Cancel the active recording task
        record_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await record_task

        # Recording must be inactive and internal buffers cleared
        assert not capture.is_recording()
        assert len(capture._chunks) == 0
        # Background stream remains intact for future captures
        assert stream.active
    finally:
        stop_feeder.set()
        await feeder_task
        capture.close()


@pytest.mark.asyncio
async def test_record_utterance_preserves_pre_speech(monkeypatch):
    """CANON-P0-03 Synthetic Test: Ring buffer preserves pre-speech audio frames."""
    import asyncio
    import numpy as np
    import sounddevice
    from shamu.speech.vad import VoiceActivityDetector, VADState

    created_streams = []

    def mock_stream_factory(*args, **kwargs):
        stream = ControllableMockStream(kwargs.get("callback"))
        created_streams.append(stream)
        return stream

    monkeypatch.setattr(sounddevice, "InputStream", mock_stream_factory)

    vad = VoiceActivityDetector(sample_rate=16000, trailing_silence_ms=64)
    capture = PushToTalkCapture(sample_rate=16000, block_size=512, pre_buffer_ms=200, vad=vad)
    stream = created_streams[0]

    # Emit 4 pre-speech chunks with distinct value 777 while idle
    for _ in range(4):
        stream.emit_chunk(val=777)

    assert len(capture._ring_buffer) == 4

    stop_feeder = asyncio.Event()

    async def feeder():
        while not capture.is_recording() and not stop_feeder.is_set():
            await asyncio.sleep(0.005)

        # Emit speech
        stream.emit_chunk(val=5000)
        await asyncio.sleep(0.01)

        # Emit trailing silence
        for _ in range(5):
            stream.emit_chunk(val=0)
            await asyncio.sleep(0.01)

    feeder_task = asyncio.create_task(feeder())

    try:
        audio = await capture.record_utterance(timeout_sec=5.0)
        assert len(audio) > 0

        # Check that pre-speech data (val 777) is preserved at beginning of audio
        arr = np.frombuffer(audio, dtype=np.int16)
        assert arr[0] == 777
        assert vad.state == VADState.FINISHED
    finally:
        stop_feeder.set()
        await feeder_task
        capture.close()




