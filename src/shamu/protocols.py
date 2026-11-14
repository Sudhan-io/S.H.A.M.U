"""Abstract interfaces and exception hierarchy for SHAMU."""

from typing import Protocol, AsyncIterator, List, Optional, runtime_checkable
from shamu.types import (
    ChatChunk,
    ChatMessage,
    ChatResponse,
    MemoryRecord,
    PermissionTier,
    PolicyOutcome,
    ToolDefinition,
)


# =====================================================================
# Exception Hierarchy
# =====================================================================

class ShamuError(Exception):
    """Base exception for all SHAMU errors."""
    pass


class ProviderUnavailableError(ShamuError):
    """Raised when an LLM provider (Ollama, OpenRouter) is unreachable or down."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message)
        self.provider = provider


class ProviderAuthError(ShamuError):
    """Raised when authentication credentials for a provider are missing or invalid."""
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message)
        self.provider = provider


class ContextWindowExceededError(ShamuError):
    """Raised when input tokens exceed model context window or trigger GPU OOM."""
    pass


class ToolExecutionError(ShamuError):
    """Raised when an automated tool fails during execution."""
    def __init__(self, tool_name: str, message: str):
        super().__init__(f"Tool '{tool_name}' failed: {message}")
        self.tool_name = tool_name


class PermissionDeniedError(ShamuError):
    """Raised when a tool action is rejected by the user or security policy."""
    pass


class AudioDeviceError(ShamuError):
    """Raised when an audio input or output device is unavailable or fails."""
    pass


class SpeechProcessingError(ShamuError):
    """Raised when speech transcription or synthesis encounters a failure."""
    pass


# =====================================================================
# Abstract Subsystem Protocols
# =====================================================================

@runtime_checkable
class ModelProvider(Protocol):
    """Abstract interface for LLM backends (Ollama, OpenRouter, etc.)."""

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatChunk]:
        """Streams response tokens and tool call deltas asynchronously."""
        ...

    async def complete_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Executes a single turn and returns the aggregated response."""
        ...


@runtime_checkable
class SpeechToTextEngine(Protocol):
    """Abstract interface for Speech-to-Text transcription (Faster-Whisper, etc.)."""

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Transcribes raw PCM audio bytes to text with optional context bias."""
        ...


@runtime_checkable
class TextToSpeechEngine(Protocol):
    """Abstract interface for streaming Text-to-Speech synthesis (Piper, SAPI5)."""

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Streams synthesized raw PCM/WAV audio bytes."""
        ...

    async def stop(self) -> None:
        """Stops active playback immediately."""
        ...


@runtime_checkable
class AudioCapture(Protocol):
    """Abstract interface for Push-to-Talk audio input capture."""

    async def record_utterance(self) -> bytes:
        """Records an utterance until silence is detected, returning raw PCM bytes."""
        ...

    def is_recording(self) -> bool:
        """Returns True if the microphone is currently active."""
        ...

    def cancel(self) -> None:
        """Cancels active recording and releases the microphone device."""
        ...


# =====================================================================
# Memory and Retrieval Protocols
# =====================================================================

@runtime_checkable
class MemoryStore(Protocol):
    """Abstract interface for zero-daemon persistent memory storage."""

    async def store(self, record: MemoryRecord, embedding: Optional[List[float]] = None) -> str:
        """Persists a memory record and optional embedding vector, returning its ID."""
        ...

    async def search_lexical(self, query: str, limit: int = 10) -> List[MemoryRecord]:
        """Performs full-text keyword search using BM25 ranking."""
        ...

    async def search_vector(self, query_vector: List[float], limit: int = 10) -> List[MemoryRecord]:
        """Performs semantic vector search using cosine similarity."""
        ...

    async def get(self, record_id: str) -> Optional[MemoryRecord]:
        """Retrieves a single memory record by its ID."""
        ...

    async def delete(self, record_id: str) -> bool:
        """Deletes a memory record and any associated embedding."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Abstract interface for hybrid retrieval combining lexical and semantic search."""

    async def retrieve(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        """Retrieves ranked memory records relevant to the query."""
        ...


