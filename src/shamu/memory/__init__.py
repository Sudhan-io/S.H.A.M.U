"""SHAMU Memory Subsystem: Zero-daemon persistent storage and hybrid retrieval."""

from shamu.memory.sqlite_store import SQLiteMemoryStore
from shamu.memory.embedding import (
    EmbeddingEngine,
    MockEmbeddingBackend,
    normalize_vector,
    cosine_similarity,
)
from shamu.memory.hybrid import (
    HybridRetriever,
    reciprocal_rank_fusion,
    format_retrieved_context,
    assemble_user_message_content,
    extract_user_query,
)
from shamu.memory.context_compactor import (
    ContextCompactor,
    estimate_message_tokens,
    estimate_history_tokens,
)

__all__ = [
    "SQLiteMemoryStore",
    "EmbeddingEngine",
    "MockEmbeddingBackend",
    "normalize_vector",
    "cosine_similarity",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "format_retrieved_context",
    "assemble_user_message_content",
    "extract_user_query",
    "ContextCompactor",
    "estimate_message_tokens",
    "estimate_history_tokens",
]



