"""Faster-Whisper Speech-to-Text running on CPU with INT8 quantization."""

import asyncio
import logging
from typing import List, Optional
import numpy as np
from shamu.protocols import SpeechToTextEngine, SpeechProcessingError

try:
    import psutil
except ImportError:
    psutil = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)


class FasterWhisperSTT(SpeechToTextEngine):
    """High-speed local speech transcription offloaded to CPU cores."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        p_cores: Optional[List[int]] = None,
        beam_size: int = 1,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.p_cores = p_cores if p_cores is not None else [0, 1, 2, 3]
        self.beam_size = beam_size
        self._model = None

    def _apply_cpu_affinity(self) -> None:
        """Deprecated/No-op: Whole-process CPU affinity was removed (ST-01) to avoid
        pinning the entire Python process (Qt GUI thread and asyncio event loop) to a subset
        of cores, which starved GUI responsiveness during Whisper inference.
        CTranslate2's cpu_threads parameter is used instead to control CPU parallelism.
        """
        pass

    def _get_model(self):
        """Lazy-loads WhisperModel into CPU memory."""
        if self._model is None:
            if WhisperModel is None:
                raise SpeechProcessingError(
                    "faster-whisper is not installed. Install via 'pip install faster-whisper'."
                )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        return self._model

    def _sync_transcribe(
        self,
        audio_float: np.ndarray,
        initial_prompt: Optional[str] = None,
        beam_size: Optional[int] = None,
    ) -> str:
        """Runs blocking inference in thread pool with dynamic prompt."""
        model = self._get_model()

        kwargs = {
            "beam_size": beam_size or self.beam_size,
            "language": "en",
            "vad_filter": True,
        }
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        segments, _ = model.transcribe(audio_float, **kwargs)
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        initial_prompt: Optional[str] = None,
        beam_size: Optional[int] = None,
    ) -> str:
        """Transcribes raw 16-bit PCM audio bytes to text asynchronously."""
        if not audio_data or len(audio_data) < 1024:
            return ""

        # Convert 16-bit PCM bytes to float32 in [-1.0, 1.0]
        samples = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = samples.astype(np.float32) / 32768.0

        try:
            return await asyncio.to_thread(
                self._sync_transcribe,
                audio_float,
                initial_prompt=initial_prompt,
                beam_size=beam_size,
            )
        except Exception as e:
            raise SpeechProcessingError(f"Transcription failed: {e}") from e
