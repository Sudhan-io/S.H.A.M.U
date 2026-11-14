import math
import pytest
from shamu.config import MemoryConfig
from shamu.memory.embedding import EmbeddingEngine, MockEmbeddingBackend, normalize_vector, cosine_similarity


def test_embedding_engine_default_unavailable_without_dependencies():
    # fastembed is not installed in the current environment
    cfg = MemoryConfig(embedding_enabled=True)
    engine = EmbeddingEngine(cfg)
    # If no local model files / fastembed package exists, engine must report unavailable
    # and return None without crashing
    assert engine.is_available() is False
    assert engine.embed_text("test sentence") is None


def test_embedding_engine_disabled_in_config():
    cfg = MemoryConfig(embedding_enabled=False)
    engine = EmbeddingEngine(cfg)
    assert engine.is_available() is False
    assert engine.embed_text("test sentence") is None


def test_embedding_vector_normalization():
    raw_vec = [3.0, 4.0, 0.0]
    normed = normalize_vector(raw_vec)
    assert len(normed) == 3
    assert math.isclose(normed[0], 0.6, abs_tol=1e-5)
    assert math.isclose(normed[1], 0.8, abs_tol=1e-5)
    # Check L2 norm equals 1.0
    l2 = math.sqrt(sum(x * x for x in normed))
    assert math.isclose(l2, 1.0, abs_tol=1e-5)

    # Zero vector should return unchanged
    assert normalize_vector([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_embedding_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    v4 = [-1.0, 0.0, 0.0]

    assert math.isclose(cosine_similarity(v1, v2), 1.0, abs_tol=1e-5)
    assert math.isclose(cosine_similarity(v1, v3), 0.0, abs_tol=1e-5)
    assert math.isclose(cosine_similarity(v1, v4), -1.0, abs_tol=1e-5)
    # Dimension mismatch should raise ValueError
    with pytest.raises(ValueError, match="Dimension mismatch"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_embedding_engine_with_mock_backend():
    cfg = MemoryConfig(embedding_enabled=True)
    mock_backend = MockEmbeddingBackend(dim=384)
    engine = EmbeddingEngine(cfg, backend=mock_backend)

    assert engine.is_available() is True
    vec = engine.embed_text("The quick brown fox jumps over the lazy dog.")
    assert vec is not None
    assert len(vec) == 384
    # Check unit normalized
    l2 = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(l2, 1.0, abs_tol=1e-5)

    # Deterministic test
    vec2 = engine.embed_text("The quick brown fox jumps over the lazy dog.")
    assert vec == vec2


def test_embedding_failure_recovery():
    cfg = MemoryConfig(embedding_enabled=True)

    class FailingBackend:
        def embed(self, text: str):
            raise RuntimeError("Model execution crashed or out of memory")

    engine = EmbeddingEngine(cfg, backend=FailingBackend())
    assert engine.is_available() is True
    # Must catch and return None safely
    res = engine.embed_text("crash this")
    assert res is None

