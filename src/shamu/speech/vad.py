"""Voice Activity Detection state machine with ONNX Silero VAD and energy fallback."""

import math
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np


class VADState(str, Enum):
    WAITING = "waiting"       # Waiting for initial speech to begin
    SPEAKING = "speaking"     # User is actively speaking
    FINISHED = "finished"     # Trailing silence detected after speech


class VoiceActivityDetector:
    """Detects speech and trailing silence to automatically bound Push-to-Talk utterances."""

    def __init__(
        self,
        sample_rate: int = 16000,
        trailing_silence_ms: int = 600,
        threshold: float = 0.5,
        model_path: Optional[str] = None,
        session: Optional[Any] = None,
    ):
        self.sample_rate = sample_rate
        self.trailing_silence_ms = trailing_silence_ms
        self.threshold = threshold
        self.state = VADState.WAITING
        self._silent_frames_count = 0
        self._frame_duration_ms = 32  # 512 samples at 16kHz
        self._required_silent_frames = math.ceil(trailing_silence_ms / self._frame_duration_ms)
        self._session = session

        if self._session is None and model_path and Path(model_path).exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                self._session = ort.InferenceSession(str(model_path), sess_options=opts)
            except Exception:
                self._session = None

        self._init_session_metadata()

    def _init_session_metadata(self) -> None:
        """Inspects model inputs and outputs to configure recurrent state handling."""
        self._has_state = False
        self._has_hc = False
        self._state_shape = (2, 1, 128)
        self._h_shape = (2, 1, 64)
        self._c_shape = (2, 1, 64)
        self._state: Optional[np.ndarray] = None
        self._h: Optional[np.ndarray] = None
        self._c: Optional[np.ndarray] = None

        if self._session is None:
            return

        try:
            inputs = self._session.get_inputs()
            input_names = [inp.name for inp in inputs]

            if "state" in input_names:
                self._has_state = True
                state_inp = next(inp for inp in inputs if inp.name == "state")
                if hasattr(state_inp, "shape") and state_inp.shape:
                    self._state_shape = tuple(
                        d if isinstance(d, int) and d > 0 else 1 for d in state_inp.shape
                    )
            elif "h" in input_names and "c" in input_names:
                self._has_hc = True
                h_inp = next(inp for inp in inputs if inp.name == "h")
                c_inp = next(inp for inp in inputs if inp.name == "c")
                if hasattr(h_inp, "shape") and h_inp.shape:
                    self._h_shape = tuple(
                        d if isinstance(d, int) and d > 0 else 1 for d in h_inp.shape
                    )
                if hasattr(c_inp, "shape") and c_inp.shape:
                    self._c_shape = tuple(
                        d if isinstance(d, int) and d > 0 else 1 for d in c_inp.shape
                    )
        except Exception:
            pass

        self._reset_states()

    def _reset_states(self) -> None:
        """Clears recurrent state tensors to zeros."""
        if self._has_state:
            self._state = np.zeros(self._state_shape, dtype=np.float32)
        elif self._has_hc:
            self._h = np.zeros(self._h_shape, dtype=np.float32)
            self._c = np.zeros(self._c_shape, dtype=np.float32)

    def reset(self) -> None:
        """Resets detector state to WAITING and clears recurrent states."""
        self.state = VADState.WAITING
        self._silent_frames_count = 0
        self._reset_states()

    def _calculate_energy_probability(self, samples: np.ndarray) -> float:
        """Fallback energy-based speech probability estimation."""
        if len(samples) == 0:
            return 0.0
        # Compute RMS (Root Mean Square) energy of 16-bit PCM samples
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        # RMS above ~350 indicates speech / voice activity in standard 16-bit audio
        if rms < 200:
            return 0.05
        if rms > 1000:
            return 0.95
        return (rms - 200) / 800.0

    def process_chunk(self, audio_bytes: bytes) -> Tuple[float, VADState]:
        """Processes a 32ms audio chunk (512 samples @ 16kHz int16).
        
        Returns:
            Tuple[float, VADState]: (speech_probability, current_vad_state)
        """
        samples = np.frombuffer(audio_bytes, dtype=np.int16)

        prob = 0.0
        if self._session is not None:
            if not self._has_state and not self._has_hc and self._state is None and self._h is None:
                self._init_session_metadata()

            try:
                # Prepare input tensor for Silero ONNX (1, N) float32 in [-1.0, 1.0]
                audio_float = samples.astype(np.float32) / 32768.0
                inputs_meta = self._session.get_inputs()
                input_names = [inp.name for inp in inputs_meta]

                # Match audio input name and check expected length
                input_name = "input" if "input" in input_names else input_names[0]
                audio_meta = next(inp for inp in inputs_meta if inp.name == input_name)
                if hasattr(audio_meta, "shape") and len(audio_meta.shape) >= 2:
                    expected_len = audio_meta.shape[1]
                    if isinstance(expected_len, int) and expected_len > 0:
                        if len(audio_float) < expected_len:
                            audio_float = np.pad(audio_float, (0, expected_len - len(audio_float)))
                        elif len(audio_float) > expected_len:
                            audio_float = audio_float[:expected_len]

                inputs = {
                    input_name: np.expand_dims(audio_float, axis=0),
                }

                if "sr" in input_names:
                    inputs["sr"] = np.array(self.sample_rate, dtype=np.int64)

                if self._has_state and "state" in input_names:
                    inputs["state"] = self._state
                elif self._has_hc:
                    if "h" in input_names:
                        inputs["h"] = self._h
                    if "c" in input_names:
                        inputs["c"] = self._c

                out = self._session.run(None, inputs)

                # Extract speech probability
                prob = float(np.asarray(out[0]).flat[-1])

                # Update recurrent states
                output_names = [out_meta.name for out_meta in self._session.get_outputs()]
                if self._has_state:
                    if "state" in output_names:
                        self._state = out[output_names.index("state")]
                    elif len(out) > 1:
                        self._state = out[1]
                elif self._has_hc:
                    if "hn" in output_names and "cn" in output_names:
                        self._h = out[output_names.index("hn")]
                        self._c = out[output_names.index("cn")]
                    elif "h" in output_names and "c" in output_names:
                        self._h = out[output_names.index("h")]
                        self._c = out[output_names.index("c")]
                    elif len(out) >= 3:
                        self._h = out[1]
                        self._c = out[2]

            except Exception:
                prob = self._calculate_energy_probability(samples)
        else:
            prob = self._calculate_energy_probability(samples)

        is_speech = prob >= self.threshold

        if self.state == VADState.WAITING:
            if is_speech:
                self.state = VADState.SPEAKING
                self._silent_frames_count = 0

        elif self.state == VADState.SPEAKING:
            if not is_speech:
                self._silent_frames_count += 1
                if self._silent_frames_count >= self._required_silent_frames:
                    self.state = VADState.FINISHED
            else:
                self._silent_frames_count = 0

        return prob, self.state
