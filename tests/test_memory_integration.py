import time
import pytest
from pathlib import Path
from shamu.config import ShamuConfig, MemoryConfig, InferenceConfig
from shamu.types import ChatMessage, Role, MemoryRecord
from shamu.memory.sqlite_store import SQLiteMemoryStore
from shamu.memory.embedding import EmbeddingEngine, MockEmbeddingBackend
from shamu.memory.hybrid import HybridRetriever, format_retrieved_context
from shamu.memory.context_compactor import ContextCompactor, estimate_history_tokens


@pytest.mark.asyncio
async def test_full_memory_lifecycle_and_prompt_injection(tmp_path: Path):
    db_file = tmp_path / "integrated_memory.db"
    mem_cfg = MemoryConfig(
        db_path=str(db_file),
        fts_enabled=True,
        hybrid_rrf_k=60,
        context_token_budget=150,
        max_retrieved_memories=3,
    )
    store = SQLiteMemoryStore(mem_cfg)
    mock_emb = MockEmbeddingBackend(dim=64)
    embedding_engine = EmbeddingEngine(mem_cfg, backend=mock_emb)
    retriever = HybridRetriever(store, embedding_engine, mem_cfg)
    compactor = ContextCompactor(store, mem_cfg)

    # 1. Store persistent memory records
    records = [
        MemoryRecord(
            id="pref-gpu",
            content="Hardware configuration: Lenovo LOQ with RTX 3050 Laptop GPU (6GB VRAM) and i5-12450HX",
            source="user_explicit",
            category="hardware",
        ),
        MemoryRecord(
            id="pref-editor",
            content="User prefers Neovim and VS Code keybindings for code editing",
            source="conversation",
            category="preference",
        ),
        MemoryRecord(
            id="pref-speech",
            content="Speech pipeline uses Whisper large-v3-turbo with dynamic lexicon context biasing",
            source="design_doc",
            category="architecture",
        ),
    ]

    for r in records:
        emb = embedding_engine.embed_text(r.content)
        await store.store(r, embedding=emb)

    # 2. Retrieve memories relevant to user query
    user_query = "What GPU and hardware specs are we targeting?"
    t0 = time.perf_counter()
    retrieved = await retriever.retrieve(user_query, limit=2)
    query_latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n[BENCHMARK] Hybrid retrieval latency: {query_latency_ms:.2f} ms")
    assert query_latency_ms < 50.0  # Zero-daemon SQLite FTS5 + RRF should be < 50ms locally
    assert len(retrieved) >= 1
    assert retrieved[0].id == "pref-gpu"
    assert "RTX 3050" in retrieved[0].content

    # 3. Format context block for injection
    context_block = format_retrieved_context(retrieved)
    assert "### Retrieved Relevant Knowledge & Context:" in context_block
    assert "[hardware] Hardware configuration: Lenovo LOQ" in context_block

    # 4. Multi-turn conversation exceeding budget
    messages = [
        ChatMessage(role=Role.SYSTEM, content=f"You are S.H.A.M.U.\n\n{context_block}"),
    ]
    for i in range(1, 8):
        messages.append(ChatMessage(role=Role.USER, content=f"User request number {i} discussing intermediate topic."))
        messages.append(ChatMessage(role=Role.ASSISTANT, content=f"Assistant response number {i} confirming topic {i}."))
    messages.append(ChatMessage(role=Role.USER, content=user_query))

    assert estimate_history_tokens(messages) > mem_cfg.context_token_budget

    # 5. Context compaction
    t1 = time.perf_counter()
    compacted, pruned_count = await compactor.compact(messages)
    compaction_latency_ms = (time.perf_counter() - t1) * 1000.0

    print(f"[BENCHMARK] Context compaction latency: {compaction_latency_ms:.2f} ms")
    assert compaction_latency_ms < 20.0
    assert pruned_count > 0
    # System prompt with retrieved context must be preserved
    assert compacted[0].role == Role.SYSTEM
    assert "Lenovo LOQ" in compacted[0].content
    # Latest user query must be preserved
    assert compacted[-1].role == Role.USER
    assert compacted[-1].content == user_query

    # 6. Verify pruned messages were archived to SQLite
    archived_records = await store.search_lexical("intermediate topic")
    assert len(archived_records) >= 1
    assert archived_records[0].category == "conversation_history"

    store.close()

