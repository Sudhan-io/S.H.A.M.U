import math
import pytest
from pathlib import Path
from shamu.config import MemoryConfig
from shamu.types import MemoryRecord
from shamu.memory.sqlite_store import SQLiteMemoryStore
from shamu.memory.embedding import EmbeddingEngine, MockEmbeddingBackend
from shamu.memory.hybrid import HybridRetriever, reciprocal_rank_fusion, format_retrieved_context


def test_reciprocal_rank_fusion_math():
    # Candidates with ranks: 1-based index
    # Doc A: rank 1 in list 1, rank 2 in list 2
    # Doc B: rank 2 in list 1, not in list 2
    # Doc C: not in list 1, rank 1 in list 2
    lexical_ids = ["doc_a", "doc_b"]
    vector_ids = ["doc_c", "doc_a"]

    k = 60
    scores = reciprocal_rank_fusion([lexical_ids, vector_ids], k=k)

    expected_score_a = (1.0 / (k + 1)) + (1.0 / (k + 2))
    expected_score_b = 1.0 / (k + 2)
    expected_score_c = 1.0 / (k + 1)

    assert math.isclose(scores["doc_a"], expected_score_a, abs_tol=1e-6)
    assert math.isclose(scores["doc_b"], expected_score_b, abs_tol=1e-6)
    assert math.isclose(scores["doc_c"], expected_score_c, abs_tol=1e-6)

    # Doc A appears in both lists, so its score must strictly exceed B and C
    assert scores["doc_a"] > scores["doc_c"] > scores["doc_b"]


@pytest.mark.asyncio
async def test_hybrid_retriever_fuses_lexical_and_semantic(tmp_path: Path):
    cfg = MemoryConfig(
        db_path=str(tmp_path / "hybrid_test.db"),
        fts_enabled=True,
        hybrid_rrf_k=60,
        max_retrieved_memories=5,
    )
    store = SQLiteMemoryStore(cfg)
    mock_backend = MockEmbeddingBackend(dim=4)
    embedding_engine = EmbeddingEngine(cfg, backend=mock_backend)

    # Save 3 records with embeddings
    rec_a = MemoryRecord(id="rec-a", content="Python async programming with asyncio", category="code")
    rec_b = MemoryRecord(id="rec-b", content="Python data analysis with pandas", category="code")
    rec_c = MemoryRecord(id="rec-c", content="Rust concurrency models", category="code")

    v_a = embedding_engine.embed_text(rec_a.content)
    v_b = embedding_engine.embed_text(rec_b.content)
    v_c = embedding_engine.embed_text(rec_c.content)

    await store.store(rec_a, embedding=v_a)
    await store.store(rec_b, embedding=v_b)
    await store.store(rec_c, embedding=v_c)

    retriever = HybridRetriever(store, embedding_engine, cfg)
    results = await retriever.retrieve("Python asyncio", limit=3)

    assert len(results) >= 1
    # rec-a matches keyword 'asyncio' and has closest semantic embedding to 'Python asyncio'
    assert results[0].id == "rec-a"
    assert results[0].score > 0
    store.close()


@pytest.mark.asyncio
async def test_hybrid_retriever_lexical_only_when_embedding_unavailable(tmp_path: Path):
    cfg = MemoryConfig(
        db_path=str(tmp_path / "lexical_only.db"),
        fts_enabled=True,
        embedding_enabled=False,
    )
    store = SQLiteMemoryStore(cfg)
    embedding_engine = EmbeddingEngine(cfg)  # unavailable
    assert embedding_engine.is_available() is False

    await store.store(MemoryRecord(id="rec-lex-1", content="User prefers fast response times"))
    await store.store(MemoryRecord(id="rec-lex-2", content="Unrelated database record"))

    retriever = HybridRetriever(store, embedding_engine, cfg)
    results = await retriever.retrieve("response times", limit=3)

    assert len(results) == 1
    assert results[0].id == "rec-lex-1"
    store.close()


@pytest.mark.asyncio
async def test_hybrid_retriever_respects_max_limit_and_threshold(tmp_path: Path):
    cfg = MemoryConfig(
        db_path=str(tmp_path / "limits.db"),
        fts_enabled=True,
        max_retrieved_memories=2,
    )
    store = SQLiteMemoryStore(cfg)
    embedding_engine = EmbeddingEngine(cfg)

    for i in range(5):
        await store.store(MemoryRecord(id=f"rec-{i}", content=f"Common project note {i}"))

    retriever = HybridRetriever(store, embedding_engine, cfg)
    results = await retriever.retrieve("project note", limit=2)
    assert len(results) <= 2
    store.close()


@pytest.mark.asyncio
async def test_hybrid_retriever_preserves_metadata_and_source(tmp_path: Path):
    cfg = MemoryConfig(db_path=str(tmp_path / "meta.db"), fts_enabled=True)
    store = SQLiteMemoryStore(cfg)
    embedding_engine = EmbeddingEngine(cfg)

    rec = MemoryRecord(
        id="rec-meta",
        content="Deploy target is Windows 11 LOQ laptop",
        source="system_spec",
        category="hardware",
        metadata={"gpu": "RTX 3050", "vram_gb": 6},
    )
    await store.store(rec)

    retriever = HybridRetriever(store, embedding_engine, cfg)
    results = await retriever.retrieve("Windows 11 laptop")
    assert len(results) == 1
    matched = results[0]
    assert matched.id == "rec-meta"
    assert matched.source == "system_spec"
    assert matched.category == "hardware"
    assert matched.metadata == {"gpu": "RTX 3050", "vram_gb": 6}
    store.close()


def test_format_retrieved_context_empty_and_populated():
    # Empty list
    assert format_retrieved_context([]) == ""

    # Populated list
    records = [
        MemoryRecord(
            id="rec-1",
            content="User prefers concise answers",
            source="user_explicit",
            category="preference",
        ),
        MemoryRecord(
            id="rec-2",
            content="Project uses uv and Python 3.12",
            source="workspace",
            category="project",
        ),
    ]
    formatted = format_retrieved_context(records)
    assert "### Retrieved Relevant Knowledge & Context:" in formatted
    assert "[preference] User prefers concise answers" in formatted
    assert "[project] Project uses uv and Python 3.12" in formatted


def test_reciprocal_rank_fusion_normalized():
    """MEM-04: reciprocal_rank_fusion with normalize=True bounds scores in [0.0, 1.0]."""
    list1 = ["doc_top", "doc_mid"]
    list2 = ["doc_top", "doc_other"]
    k = 60

    scores = reciprocal_rank_fusion([list1, list2], k=k, normalize=True)

    # doc_top is rank 1 in both lists -> score must be 1.0
    assert pytest.approx(scores["doc_top"], 1e-5) == 1.0
    # doc_mid is rank 2 in list 1 only -> score < 0.5
    assert 0.0 < scores["doc_mid"] < 0.5
    # doc_other is rank 2 in list 2 only
    assert pytest.approx(scores["doc_other"], 1e-5) == scores["doc_mid"]


@pytest.mark.asyncio
async def test_hybrid_retriever_not_dropped_by_standard_threshold(tmp_path: Path):
    """MEM-04: Standard min_relevance_score thresholds (e.g. 0.3-0.5) must not drop relevant hybrid results."""
    cfg = MemoryConfig(
        db_path=str(tmp_path / "threshold_test.db"),
        fts_enabled=True,
        hybrid_rrf_k=60,
        min_relevance_score=0.4,  # Standard threshold; unnormalized RRF max score is ~0.033
        max_retrieved_memories=5,
    )
    store = SQLiteMemoryStore(cfg)
    mock_backend = MockEmbeddingBackend(dim=4)
    embedding_engine = EmbeddingEngine(cfg, backend=mock_backend)

    rec = MemoryRecord(id="rec-hit", content="Postgres connection pooling and query optimization", category="db")
    v = embedding_engine.embed_text(rec.content)
    await store.store(rec, embedding=v)

    retriever = HybridRetriever(store, embedding_engine, cfg)
    results = await retriever.retrieve("Postgres connection", limit=3)

    # Must NOT be dropped by min_relevance_score=0.4
    assert len(results) == 1
    assert results[0].id == "rec-hit"
    assert 0.4 <= results[0].score <= 1.0

    store.close()


