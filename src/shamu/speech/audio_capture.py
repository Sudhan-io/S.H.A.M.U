"""Push-to-Talk 16kHz audio capture with thread-safe ring buffers and pre-speech buffering."""

import atexit
import asyncio
import collections
import logging
import math
import threading
import weakref
from typing import AsyncIterator, List, Optional
import numpy as np
from shamu.protocols import AudioCapture, AudioDeviceError
from shamu.speech.vad import VoiceActivityDetector, VADState

logger = logging.getLogger(__name__)

_active_captures = weakref.WeakSet()


def _cleanup_all_captures() -> None:
    """Closes all active audio capture streams deterministically upon process exit."""
    for cap in list(_active_captures):
        try:
            cap.close()
        except Exception:
            pass


atexit.register(_cleanup_all_captures)


class PushToTalkCapture(AudioCapture):
    """Captures 16kHz mono PCM audio using sounddevice with Push-to-Talk control and pre-buffering."""

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 512,
        pre_buffer_ms: int = 200,
        auto_start_stream: bool = True,
        vad: Optional[VoiceActivityDetector] = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.pre_buffer_ms = pre_buffer_ms
        self.pre_buffer_blocks = max(1, math.ceil((pre_buffer_ms / 1000.0) * sample_rate / block_size))
        self._ring_buffer: collections.deque = collections.deque(maxlen=self.pre_buffer_blocks)
        self._stream = None
        self._recording = False
        self._chunks: List[bytes] = []
        self._lock = threading.RLock()
        self.vad = vad
        self._chunk_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        _active_captures.add(self)

        if auto_start_stream:
            try:
                self._start_stream()
            except Exception as e:
                logger.debug("Initial background audio stream deferred: %s", e)

    def _start_stream(self) -> None:
        """Starts persistent audio stream for continuous pre-speech buffering."""
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.block_size,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise AudioDeviceError(f"Failed to start audio recording: {e}") from e

    def _audio_callback(self, indata, frames, time_info, status):
        """Low-level sounddevice C callback; buffers audio chunks and maintains pre-buffer."""
        data_bytes = indata.tobytes() if hasattr(indata, "tobytes") else bytes(indata)
        with self._lock:
            if self._recording:
                self._chunks.append(data_bytes)
                if self._chunk_queue is not None and self._loop is not None:
                    try:
                        if not self._loop.is_closed():
                            try:
                                running = asyncio.get_running_loop()
                            except RuntimeError:
                                running = None
                            if running is self._loop:
                                self._chunk_queue.put_nowait(data_bytes)
                            else:
                                self._loop.call_soon_threadsafe(
                                    self._chunk_queue.put_nowait, data_bytes
                                )
                    except (RuntimeError, asyncio.QueueFull):
                        pass
            else:
                self._ring_buffer.append(data_bytes)

    def start(self) -> None:
        """Opens microphone stream if needed and begins buffering, prepending pre-speech frames."""
        with self._lock:
            if self._recording:
                return
            # Prepend the pre-speech ring buffer so initial consonants are preserved
            self._chunks = list(self._ring_buffer)
            self._recording = True

            if self._stream is None:
                try:
                    self._start_stream()
                except Exception:
                    self._recording = False
                    raise

    def stop(self) -> bytes:
        """Stops recording mode and returns complete audio bytes while keeping stream open for pre-buffering."""
        with self._lock:
            self._recording = False
            q = self._chunk_queue
            loop = self._loop
            self._chunk_queue = None
            self._loop = None
            result = b"".join(self._chunks)
            self._chunks.clear()
            self._ring_buffer.clear()

        if q is not None and loop is not None:
            try:
                if not loop.is_closed():
                    try:
                        running = asyncio.get_running_loop()
                    except RuntimeError:
                        running = None
                    if running is loop:
                        q.put_nowait(None)
                    else:
                        loop.call_soon_threadsafe(q.put_nowait, None)
            except (RuntimeError, asyncio.QueueFull):
                pass

        return result

    def cancel(self) -> None:
        """Aborts recording and discards audio buffers without closing stream."""
        with self._lock:
            self._recording = False
            q = self._chunk_queue
            loop = self._loop
            self._chunk_queue = None
            self._loop = None
            self._chunks.clear()
            self._ring_buffer.clear()

        if q is not None and loop is not None:
            try:
                if not loop.is_closed():
                    try:
                        running = asyncio.get_running_loop()
                    except RuntimeError:
                        running = None
                    if running is loop:
                        q.put_nowait(None)
                    else:
                        loop.call_soon_threadsafe(q.put_nowait, None)
            except (RuntimeError, asyncio.QueueFull):
                pass

    def close(self) -> None:
        """Stops and closes the persistent microphone stream."""
        self.cancel()
        with self._lock:
            _active_captures.discard(self)
            stream = self._stream
            self._stream = None
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass

    def __enter__(self) -> "PushToTalkCapture":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    async def __aenter__(self) -> "PushToTalkCapture":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def is_recording(self) -> bool:
        """Returns True if audio capture is active."""
        return self._recording

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        """Streams audio chunks in real-time as an async generator."""
        loop = asyncio.get_running_loop()
        chunk_queue: asyncio.Queue = asyncio.Queue()

        with self._lock:
            self.start()
            for chunk in self._chunks:
                chunk_queue.put_nowait(chunk)
            self._chunk_queue = chunk_queue
            self._loop = loop

        try:
            while self._recording:
                chunk = await chunk_queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            with self._lock:
                if self._chunk_queue is chunk_queue:
                    self._chunk_queue = None
                    self._loop = None

    async def record_utterance(
        self,
        timeout_sec: float = 15.0,
        vad: Optional[VoiceActivityDetector] = None,
        max_initial_silence_sec: Optional[float] = 3.0,
    ) -> bytes:
        """Captures utterance until speech ends (trailing silence met) or timeout expires.

        Streams incoming audio chunks to VoiceActivityDetector to detect speech onset
        and trailing silence. Preserves pre-speech frames via ring buffer.
        """
        active_vad = vad or self.vad or VoiceActivityDetector(sample_rate=self.sample_rate)
        active_vad.reset()

        loop = asyncio.get_running_loop()
        chunk_queue: asyncio.Queue = asyncio.Queue()

        with self._lock:
            self.start()
            for chunk in self._chunks:
                chunk_queue.put_nowait(chunk)
            self._chunk_queue = chunk_queue
            self._loop = loop

        start_time = loop.time()
        speech_started = False

        try:
            while True:
                now = loop.time()
                elapsed = now - start_time

                if elapsed >= timeout_sec:
                    logger.debug("record_utterance: upper bound timeout (%.2fs) reached", timeout_sec)
                    break

                if not speech_started and max_initial_silence_sec is not None:
                    if elapsed >= max_initial_silence_sec:
                        logger.debug(
                            "record_utterance: initial silence timeout (%.2fs) reached",
                            max_initial_silence_sec,
                        )
                        break

                remaining = timeout_sec - elapsed
                if not speech_started and max_initial_silence_sec is not None:
                    remaining = min(remaining, max_initial_silence_sec - elapsed)
                remaining = max(0.005, remaining)

                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

                if chunk is None:
                    # Cancelled or stopped externally
                    break

                prob, state = active_vad.process_chunk(chunk)
                if state == VADState.SPEAKING:
                    speech_started = True
                elif state == VADState.FINISHED:
                    logger.debug("record_utterance: trailing silence detected, terminating capture")
                    break

            recorded = self.stop()
            if not speech_started and max_initial_silence_sec is not None:
                return b""
            return recorded

        except asyncio.CancelledError:
            self.cancel()
            raise
        except Exception:
            self.cancel()
            raise
        finally:
            with self._lock:
                if self._chunk_queue is chunk_queue:
                    self._chunk_queue = None
                    self._loop = None
