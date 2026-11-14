"""Vector embedding utilities and engine with graceful fallback."""

import asyncio
import hashlib
import logging
import math
import random
from typing import Any, List, Optional, Protocol

from shamu.config import MemoryConfig

logger = logging.getLogger(__name__)


def normalize_vector(vec: List[float]) -> List[float]:
    """Normalizes a float vector to unit length (L2 norm = 1.0)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return [0.0] * len(vec)
    return [x / norm for x in vec]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Dimension mismatch: len(v1)={len(v1)} != len(v2)={len(v2)}")

    norm1 = math.sqrt(sum(x * x for x in v1))
    norm2 = math.sqrt(sum(x * x for x in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2))
    return dot / (norm1 * norm2)


class EmbeddingBackend(Protocol):
    """Protocol for embedding model backends."""
    def embed(self, text: str) -> List[float]:
        ...


class MockEmbeddingBackend:
    """Deterministic mock embedding backend for testing without external models."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        """Produces a deterministic pseudo-random unit vector based on text hash."""
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        raw = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        return normalize_vector(raw)


class FastEmbedBackend:
    """FastEmbed ONNX CPU backend (active if fastembed is installed)."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> List[float]:
        generator = self._model.embed([text])
        for vec in generator:
            return list(vec)
        raise RuntimeError("FastEmbed returned empty embedding generator")


class EmbeddingEngine:
    """Manages text embedding generation with fallback when dependencies are missing."""

    def __init__(self, config: MemoryConfig, backend: Optional[Any] = None) -> None:
        self.config = config
        self._backend = backend

        if self._backend is None and self.config.embedding_enabled:
            try:
                self._backend = FastEmbedBackend(self.config.embedding_model)
                logger.info(f"Initialized FastEmbed backend with {self.config.embedding_model}")
            except Exception as e:
                logger.debug(f"FastEmbed backend unavailable: {e}. Falling back to lexical-only memory.")
                self._backend = None

    def is_available(self) -> bool:
        """Returns True if embedding backend is active and enabled."""
        return self.config.embedding_enabled and (self._backend is not None)

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generates a normalized embedding vector for the given text, or None if unavailable."""
        if not self.is_available() or self._backend is None:
            return None

        try:
            raw_vec = self._backend.embed(text)
            return normalize_vector(raw_vec)
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embeds a batch of texts, returning None for any failed entry."""
        return [self.embed_text(t) for t in texts]

    async def embed_text_async(self, text: str) -> Optional[List[float]]:
        """Asynchronously offloads ONNX/CPU embedding inference to a worker thread."""
        return await asyncio.to_thread(self.embed_text, text)

    async def embed_batch_async(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Asynchronously offloads batch embedding inference to a worker thread."""
        return await asyncio.to_thread(self.embed_batch, texts)


