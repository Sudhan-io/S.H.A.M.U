import numpy as np
import pytest
from pathlib import Path
from shamu.speech.vad import VoiceActivityDetector, VADState


class MockNode:
    def __init__(self, name: str, shape=None, node_type: str = "tensor(float)"):
        self.name = name
        self.shape = list(shape) if shape is not None else [1, 512]
        self.type = node_type


class MockHCSession:
    def __init__(self, shape=(2, 1, 64)):
        self.shape = shape
        self.calls = []

    def get_inputs(self):
        return [
            MockNode("input", [1, 512]),
            MockNode("sr", []),
            MockNode("h", self.shape),
            MockNode("c", self.shape),
        ]

    def get_outputs(self):
        return [
            MockNode("output", [1, 1]),
            MockNode("hn", self.shape),
            MockNode("cn", self.shape),
        ]

    def run(self, output_names, input_feed, run_options=None):
        self.calls.append(input_feed)
        # Return probability 0.8 (speech), updated h with +1, c with +2
        h_prev = input_feed["h"]
        c_prev = input_feed["c"]
        h_next = h_prev + 1.0
        c_next = c_prev + 2.0
        return [np.array([[0.8]], dtype=np.float32), h_next, c_next]


class MockStateSession:
    def __init__(self, shape=(2, 1, 128)):
        self.shape = shape
        self.calls = []

    def get_inputs(self):
        return [
            MockNode("input", [1, 512]),
            MockNode("state", self.shape),
            MockNode("sr", []),
        ]

    def get_outputs(self):
        return [
            MockNode("output", [1, 1]),
            MockNode("state", self.shape),
        ]

    def run(self, output_names, input_feed, run_options=None):
        self.calls.append(input_feed)
        state_prev = input_feed["state"]
        state_next = state_prev + 1.5
        return [np.array([[0.75]], dtype=np.float32), state_next]


def test_vad_silence_and_speech_detection():
    vad = VoiceActivityDetector(sample_rate=16000, trailing_silence_ms=600)

    # Silent 32ms frame (512 samples)
    silent_chunk = np.zeros(512, dtype=np.int16).tobytes()
    prob, state = vad.process_chunk(silent_chunk)
    assert prob < 0.2
    assert state == VADState.WAITING

    # Loud synthesized speech-like frame (sine wave)
    t = np.linspace(0, 0.032, 512, endpoint=False)
    loud_samples = (np.sin(2 * np.pi * 440 * t) * 15000).astype(np.int16)
    loud_chunk = loud_samples.tobytes()

    prob, state = vad.process_chunk(loud_chunk)
    assert prob > 0.5
    assert state == VADState.SPEAKING


def test_vad_state_trailing_silence_trigger():
    # 600 ms of 32 ms frames is ~19 consecutive silent frames
    vad = VoiceActivityDetector(sample_rate=16000, trailing_silence_ms=600)

    # 1. Trigger speech
    t = np.linspace(0, 0.032, 512, endpoint=False)
    loud_samples = (np.sin(2 * np.pi * 440 * t) * 15000).astype(np.int16).tobytes()
    vad.process_chunk(loud_samples)
    assert vad.state == VADState.SPEAKING

    # 2. Feed consecutive silent frames
    silent_chunk = np.zeros(512, dtype=np.int16).tobytes()
    state = vad.state
    for _ in range(18):
        _, state = vad.process_chunk(silent_chunk)
        assert state == VADState.SPEAKING  # Still within trailing window

    # 19th frame reaches >= 600 ms silence -> FINISHED
    _, state = vad.process_chunk(silent_chunk)
    assert state == VADState.FINISHED


def test_vad_recurrent_state_hc_lifecycle():
    """Verify h/c recurrent states are passed, maintained across chunks, and cleared on reset."""
    mock_sess = MockHCSession(shape=(2, 1, 64))
    vad = VoiceActivityDetector(sample_rate=16000, session=mock_sess)

    assert vad._has_hc is True
    assert vad._has_state is False
    assert np.all(vad._h == 0)
    assert np.all(vad._c == 0)

    chunk = np.zeros(512, dtype=np.int16).tobytes()

    # First chunk
    prob, state = vad.process_chunk(chunk)
    assert prob == pytest.approx(0.8)
    assert len(mock_sess.calls) == 1
    # Verify initial input had zeros
    assert np.all(mock_sess.calls[0]["h"] == 0)
    assert np.all(mock_sess.calls[0]["c"] == 0)
    # Verify vad stored the returned states
    assert np.all(vad._h == 1.0)
    assert np.all(vad._c == 2.0)

    # Second chunk should receive updated states
    vad.process_chunk(chunk)
    assert len(mock_sess.calls) == 2
    assert np.all(mock_sess.calls[1]["h"] == 1.0)
    assert np.all(mock_sess.calls[1]["c"] == 2.0)
    assert np.all(vad._h == 2.0)
    assert np.all(vad._c == 4.0)

    # Reset clears states back to zeros
    vad.reset()
    assert vad.state == VADState.WAITING
    assert np.all(vad._h == 0)
    assert np.all(vad._c == 0)

    # Subsequent chunk after reset starts from zeros again
    vad.process_chunk(chunk)
    assert len(mock_sess.calls) == 3
    assert np.all(mock_sess.calls[2]["h"] == 0)
    assert np.all(mock_sess.calls[2]["c"] == 0)


def test_vad_recurrent_state_single_state_lifecycle():
    """Verify single 'state' recurrent tensor is passed, maintained across chunks, and cleared on reset."""
    mock_sess = MockStateSession(shape=(2, 1, 128))
    vad = VoiceActivityDetector(sample_rate=16000, session=mock_sess)

    assert vad._has_state is True
    assert vad._has_hc is False
    assert np.all(vad._state == 0)

    chunk = np.zeros(512, dtype=np.int16).tobytes()

    # First chunk
    prob, state = vad.process_chunk(chunk)
    assert prob == pytest.approx(0.75)
    assert len(mock_sess.calls) == 1
    assert np.all(mock_sess.calls[0]["state"] == 0)
    assert np.all(vad._state == 1.5)

    # Second chunk
    vad.process_chunk(chunk)
    assert len(mock_sess.calls) == 2
    assert np.all(mock_sess.calls[1]["state"] == 1.5)
    assert np.all(vad._state == 3.0)

    # Reset clears state back to zeros
    vad.reset()
    assert vad.state == VADState.WAITING
    assert np.all(vad._state == 0)

    # Subsequent chunk after reset uses zeros
    vad.process_chunk(chunk)
    assert len(mock_sess.calls) == 3
    assert np.all(mock_sess.calls[2]["state"] == 0)


def test_vad_real_onnx_model_if_available():
    """Integration test with faster_whisper bundled Silero VAD onnx model if installed."""
    import faster_whisper
    asset_dir = Path(faster_whisper.__file__).parent / "assets"
    v6_path = asset_dir / "silero_vad_v6.onnx"
    if not v6_path.exists():
        pytest.skip("silero_vad_v6.onnx not present in environment")

    vad = VoiceActivityDetector(sample_rate=16000, model_path=str(v6_path))
    assert vad._has_hc is True
    assert vad._h is not None and vad._c is not None
    assert np.all(vad._h == 0)

    chunk = np.zeros(512, dtype=np.int16).tobytes()
    prob, state = vad.process_chunk(chunk)
    assert isinstance(prob, float)
    # Model should have updated recurrent states from 0
    assert not np.all(vad._h == 0)

    # Reset should clear states
    vad.reset()
    assert np.all(vad._h == 0)
    assert np.all(vad._c == 0)
