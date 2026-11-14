"""Unified Voice Pipeline coordinating Push-to-Talk, VAD, STT, and streaming TTS."""

import asyncio
from typing import AsyncIterator, Callable, List, Optional, Any
from shamu.types import ChatMessage, Role
from shamu.protocols import (
    SpeechToTextEngine,
    TextToSpeechEngine,
    AudioCapture,
    ModelProvider,
)
from shamu.speech.audio_capture import PushToTalkCapture
from shamu.speech.vad import VoiceActivityDetector, VADState
from shamu.speech.stt import FasterWhisperSTT
from shamu.speech.tts import TextToSpeech
from shamu.speech.sentence_splitter import SentenceSplitter
from shamu.speech.lexicon import DynamicLexicon
from shamu.gateway.router import ModelGateway


class VoicePipeline:
    """Orchestrates end-to-end speech interaction with sentence-boundary TTS streaming."""

    def __init__(
        self,
        stt: Optional[SpeechToTextEngine] = None,
        tts: Optional[TextToSpeechEngine] = None,
        gateway: Optional[ModelProvider] = None,
        capture: Optional[AudioCapture] = None,
        vad: Optional[VoiceActivityDetector] = None,
        lexicon: Optional[DynamicLexicon] = None,
        on_audio_chunk: Optional[Callable[[bytes], Any]] = None,
    ):
        self.stt = stt or FasterWhisperSTT()
        self.tts = tts or TextToSpeech()
        self.gateway = gateway or ModelGateway()
        self.vad = vad or VoiceActivityDetector()
        self.capture = capture or PushToTalkCapture(vad=self.vad)
        self.lexicon = lexicon
        self.on_audio_chunk = on_audio_chunk
        self._audio_queue: Optional[asyncio.Queue] = None
        self._playback_task: Optional[asyncio.Task] = None

    def _ensure_playback_worker(self) -> None:
        """Ensures the background playback task and queue are running."""
        if self._audio_queue is None:
            self._audio_queue = asyncio.Queue()
        if self._playback_task is None or self._playback_task.done():
            self._playback_task = asyncio.create_task(self._playback_worker())

    async def _playback_worker(self) -> None:
        """Background task reading audio chunks from queue and playing them sequentially."""
        try:
            while True:
                chunk = await self._audio_queue.get()
                if chunk is None:
                    self._audio_queue.task_done()
                    break
                try:
                    await self._play_audio_chunk(chunk)
                except Exception:
                    pass
                finally:
                    self._audio_queue.task_done()
        except asyncio.CancelledError:
            pass

    async def enqueue_audio(self, audio_data: bytes) -> None:
        """Enqueues audio data into the background playback queue."""
        if not audio_data:
            return
        self._ensure_playback_worker()
        await self._audio_queue.put(audio_data)

    async def enqueue_audio_chunk(self, audio_data: bytes) -> None:
        """Alias for enqueue_audio."""
        await self.enqueue_audio(audio_data)

    async def wait_for_playback(self) -> None:
        """Awaits completion of all currently enqueued playback items."""
        if self._audio_queue is not None:
            await self._audio_queue.join()

    async def stop_playback(self) -> None:
        """Immediately halts hardware playback and drains pending items from the queue."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        if self._audio_queue is not None:
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.task_done()
                except (asyncio.QueueEmpty, ValueError):
                    break

        if self._playback_task is not None and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except (asyncio.CancelledError, Exception):
                pass
            self._playback_task = None

    async def _play_audio_chunk(self, audio_data: bytes) -> None:
        """Dispatches audio bytes to the configured playback handler or audio device."""
        if not audio_data:
            return
        if self.on_audio_chunk:
            if asyncio.iscoroutinefunction(self.on_audio_chunk):
                await self.on_audio_chunk(audio_data)
            else:
                self.on_audio_chunk(audio_data)
            return

        try:
            import sounddevice as sd
            import numpy as np

            loop = asyncio.get_running_loop()

            def _play():
                data = audio_data
                if not data:
                    return
                # Check for RIFF WAV header
                if data.startswith(b"RIFF") and len(data) > 44:
                    try:
                        import io
                        import soundfile as sf
                        arr, sr = sf.read(io.BytesIO(data), dtype="int16")
                        sd.play(arr, samplerate=sr)
                        sd.wait()
                        return
                    except Exception:
                        pass

                if len(data) % 2 != 0:
                    data = data[:len(data) - (len(data) % 2)]
                if not data:
                    return
                arr = np.frombuffer(data, dtype=np.int16)
                sr = getattr(self.tts, "sample_rate", 24000)
                sd.play(arr, samplerate=sr)
                sd.wait()

            await loop.run_in_executor(None, _play)
        except Exception:
            pass

    async def listen_and_transcribe(
        self,
        timeout_sec: float = 10.0,
        initial_prompt: Optional[str] = None,
        max_initial_silence_sec: Optional[float] = 3.0,
    ) -> str:
        """Captures Push-to-Talk utterance with automatic VAD silence detection."""
        self.vad.reset()
        try:
            audio_data = await self.capture.record_utterance(
                timeout_sec=timeout_sec,
                vad=self.vad,
                max_initial_silence_sec=max_initial_silence_sec,
            )
        except TypeError:
            try:
                audio_data = await self.capture.record_utterance(
                    timeout_sec=timeout_sec,
                    vad=self.vad,
                )
            except TypeError:
                try:
                    audio_data = await self.capture.record_utterance(timeout_sec=timeout_sec)
                except TypeError:
                    audio_data = await self.capture.record_utterance()

        if not audio_data:
            return ""

        prompt = initial_prompt
        if prompt is None and self.lexicon is not None:
            prompt = self.lexicon.build_initial_prompt()

        return await self.stt.transcribe(audio_data, initial_prompt=prompt)

    async def run_voice_turn(
        self,
        prompt: Optional[str] = None,
        history: Optional[List[ChatMessage]] = None,
    ) -> AsyncIterator[str]:
        """Runs a complete conversational voice turn.
        
        Streams text tokens back to the caller while concurrently synthesizing
        sentences and clauses to the TTS audio output for sub-400ms TTFA.
        """
        user_text = prompt
        if not user_text:
            user_text = await self.listen_and_transcribe()

        if not user_text or not user_text.strip():
            return

        conv_history = history if history is not None else []
        conv_history.append(ChatMessage(role=Role.USER, content=user_text))

        splitter = SentenceSplitter(min_clause_words=4)
        full_reply_buffer = []

        try:
            async for chunk in self.gateway.stream_chat(conv_history):
                if chunk.text_delta:
                    yield chunk.text_delta
                    full_reply_buffer.append(chunk.text_delta)

                    # Check if a sentence or clause completed
                    sentences = splitter.feed(chunk.text_delta)
                    for sentence in sentences:
                        # Stream sentence or clause to TTS and queue audio in background
                        async for audio_chunk in self.tts.synthesize_stream(sentence):
                            if audio_chunk:
                                await self.enqueue_audio(audio_chunk)

            # Flush any remaining text at end of stream
            flushed = splitter.flush()
            for sentence in flushed:
                async for audio_chunk in self.tts.synthesize_stream(sentence):
                    if audio_chunk:
                        await self.enqueue_audio(audio_chunk)

            # Wait for queued audio playback to complete before finalizing turn
            await self.wait_for_playback()

            final_reply = "".join(full_reply_buffer)
            if final_reply:
                conv_history.append(ChatMessage(role=Role.ASSISTANT, content=final_reply))
        except GeneratorExit:
            # Caller closed generator early; halt playback
            await self.stop_playback()
            raise

    async def interrupt(self) -> None:
        """Barge-In: Immediately halts active speech output."""
        await self.tts.stop()
        self.capture.cancel()
        await self.stop_playback()
