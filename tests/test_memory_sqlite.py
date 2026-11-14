import pytest
from pathlib import Path
from shamu.config import MemoryConfig
from shamu.types import MemoryRecord
from shamu.memory.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_memory.db"


@pytest.fixture
def memory_config(temp_db_path: Path) -> MemoryConfig:
    return MemoryConfig(db_path=str(temp_db_path), fts_enabled=True)


@pytest.mark.asyncio
async def test_sqlite_store_initialization_and_wal(memory_config: MemoryConfig):
    store = SQLiteMemoryStore(memory_config)
    assert Path(memory_config.db_path).exists()
    assert store.is_fts_available() is True
    store.close()


@pytest.mark.asyncio
async def test_sqlite_store_and_get(memory_config: MemoryConfig):
    store = SQLiteMemoryStore(memory_config)
    rec = MemoryRecord(
        id="mem-1",
        content="User preferred VS Code dark theme with JetBrains Mono font.",
        source="conversation",
        category="preference",
        metadata={"editor": "vscode"},
        created_at=1700000000.0,
    )
    saved_id = await store.store(rec)
    assert saved_id == "mem-1"

    retrieved = await store.get("mem-1")
    assert retrieved is not None
    assert retrieved.id == "mem-1"
    assert retrieved.content == rec.content
    assert retrieved.source == "conversation"
    assert retrieved.category == "preference"
    assert retrieved.metadata == {"editor": "vscode"}
    assert retrieved.created_at == 1700000000.0
    store.close()


@pytest.mark.asyncio
async def test_sqlite_store_update_existing(memory_config: MemoryConfig):
    store = SQLiteMemoryStore(memory_config)
    rec1 = MemoryRecord(id="mem-update", content="Initial state", category="status")
    await store.store(rec1)

    rec2 = MemoryRecord(id="mem-update", content="Updated state", category="status")
    await store.store(rec2)

    retrieved = await store.get("mem-update")
    assert retrieved is not None
    assert retrieved.content == "Updated state"
    store.close()


@pytest.mark.asyncio
async def test_sqlite_store_delete(memory_config: MemoryConfig):
    store = SQLiteMemoryStore(memory_config)
    rec = MemoryRecord(id="mem-del", content="Temporary note")
    await store.store(rec)
    assert await store.get("mem-del") is not None

    deleted = await store.delete("mem-del")
    assert deleted is True
    assert await store.get("mem-del") is None
    deleted_again = await store.delete("mem-del")
    assert deleted_again is False
    store.close()


@pytest.mark.asyncio
async def test_sqlite_fts5_porter_stemming(memory_config: MemoryConfig):
    store = SQLiteMemoryStore(memory_config)
    await store.store(MemoryRecord(id="rec-1", content="User likes running every morning."))
    await store.store(MemoryRecord(id="rec-2", content="Database connection pooling is configured."))

    # Porter stemming: 'run' should match 'running'
    results = await store.search_lexical("run", limit=5)
    assert len(results) >= 1
    assert results[0].id == "rec-1"
    assert results[0].score > 0
    store.close()


@pytest.mark.asyncio
async def test_sqlite_fts5_unicode(memory_config: MemoryConfig):
    store = SQLiteMemoryStore(memory_config)
    await store.store(MemoryRecord(id="rec-cafe", content="We had coffee at the café near the station."))

    results = await store.search_lexical("café", limit=5)
    assert len(results) == 1
    assert results[0].id == "rec-cafe"
    store.close()


@pytest.mark.asyncio
async def test_sqlite_fts5_special_characters_sanitization(memory_config: MemoryConfig):
    store = SQLiteMemoryStore(memory_config)
    await store.store(MemoryRecord(id="rec-spec", content="Test with special syntax: NOT, AND, (OR), * wildcard."))

    # FTS5 characters should be safely sanitized without raising sqlite3.OperationalError
    queries = [
        "NOT",
        "AND",
        "(syntax OR wildcard)",
        "special * syntax",
        "\"unclosed quotes",
        "!!! ??? &&&",
    ]
    for q in queries:
        results = await store.search_lexical(q, limit=5)
        assert isinstance(results, list)
    store.close()


@pytest.mark.asyncio
async def test_sqlite_fts5_empty_query(memory_config: MemoryConfig):
    store = SQLiteMemoryStore(memory_config)
    await store.store(MemoryRecord(id="rec-1", content="Some valid content."))

    assert await store.search_lexical("", limit=5) == []
    assert await store.search_lexical("   ", limit=5) == []
    assert await store.search_lexical("   \t\n  ", limit=5) == []
    store.close()


@pytest.mark.asyncio
async def test_sqlite_vector_store_and_search(memory_config: MemoryConfig):
    store = SQLiteMemoryStore(memory_config)
    # Unit vectors for testing
    v_target = [1.0, 0.0, 0.0]
    v_close = [0.9, 0.1, 0.0]
    v_far = [0.0, 1.0, 0.0]

    await store.store(MemoryRecord(id="rec-close", content="Close semantic vector"), embedding=v_close)
    await store.store(MemoryRecord(id="rec-far", content="Perpendicular semantic vector"), embedding=v_far)

    results = await store.search_vector(v_target, limit=5)
    assert len(results) == 2
    assert results[0].id == "rec-close"
    assert results[1].id == "rec-far"
    assert results[0].score > results[1].score
    store.close()


@pytest.mark.asyncio
async def test_sqlite_reopen_database(memory_config: MemoryConfig):
    store1 = SQLiteMemoryStore(memory_config)
    await store1.store(MemoryRecord(id="rec-durable", content="Durable fact across restarts"))
    store1.close()

    # Reopen
    store2 = SQLiteMemoryStore(memory_config)
    rec = await store2.get("rec-durable")
    assert rec is not None
    assert rec.content == "Durable fact across restarts"
    search_res = await store2.search_lexical("restarts")
    assert len(search_res) == 1
    assert search_res[0].id == "rec-durable"
    store2.close()


@pytest.mark.asyncio
async def test_sqlite_lexical_fallback_when_fts_disabled(tmp_path: Path):
    cfg = MemoryConfig(db_path=str(tmp_path / "fallback.db"), fts_enabled=False)
    store = SQLiteMemoryStore(cfg)
    assert store.is_fts_available() is False

    await store.store(MemoryRecord(id="rec-fb", content="User prefers python 3.12 for async programming"))
    res = await store.search_lexical("python", limit=5)
    assert len(res) == 1
    assert res[0].id == "rec-fb"
    store.close()


@pytest.mark.asyncio
async def test_sqlite_store_batch_transactional(memory_config: MemoryConfig):
    """SPCH-04: Test batch storing multiple records in a single transaction."""
    store = SQLiteMemoryStore(memory_config)

    records = [
        MemoryRecord(id=f"batch-{i}", content=f"Batch record {i} discussing topic {i}", category="batch_test")
        for i in range(1, 6)
    ]

    await store.store_batch(records)

    # Verify all records stored and retrievable
    for i in range(1, 6):
        rec = await store.get(f"batch-{i}")
        assert rec is not None
        assert rec.id == f"batch-{i}"
        assert f"topic {i}" in rec.content

    # Verify searchable via FTS5
    search_res = await store.search_lexical("topic 3")
    assert len(search_res) >= 1
    assert search_res[0].id == "batch-3"

    store.close()


@pytest.mark.asyncio
async def test_sqlite_vectorized_cosine_similarity(memory_config: MemoryConfig):
    """SPCH-03: Vectorized cosine similarity correctly computes rankings."""
    store = SQLiteMemoryStore(memory_config)

    # Unit vector 1: [1.0, 0.0, 0.0]
    await store.store(MemoryRecord(id="v1", content="X-axis unit vector"), embedding=[1.0, 0.0, 0.0])
    # Unit vector 2: [0.707, 0.707, 0.0] (~45 degrees)
    await store.store(MemoryRecord(id="v2", content="XY diagonal vector"), embedding=[0.7071, 0.7071, 0.0])
    # Unit vector 3: [0.0, 1.0, 0.0] (90 degrees / orthogonal)
    await store.store(MemoryRecord(id="v3", content="Y-axis unit vector"), embedding=[0.0, 1.0, 0.0])

    # Query with [1.0, 0.0, 0.0]
    results = await store.search_vector([1.0, 0.0, 0.0], limit=3)
    assert len(results) == 3
    assert results[0].id == "v1"
    assert pytest.approx(results[0].score, 0.01) == 1.0
    assert results[1].id == "v2"
    assert pytest.approx(results[1].score, 0.01) == 0.7071
    assert results[2].id == "v3"
    assert pytest.approx(results[2].score, 0.01) == 0.0

    store.close()


@pytest.mark.asyncio
async def test_sqlite_bm25_weights_and_no_k1_b_sql_injection(tmp_path: Path):
    """MEM-01: Verify bm25(memories_fts, 0.0, 1.0) is called without formatting k1/b into SQL."""
    # Custom config with extreme bm25 parameters to ensure they are not string-formatted into SQL
    cfg = MemoryConfig(db_path=str(tmp_path / "bm25_weights.db"), bm25_k1=999.9, bm25_b=888.8)
    store = SQLiteMemoryStore(cfg)

    await store.store(MemoryRecord(id="bm25-1", content="Quantum computing using superconducting qubits"))
    await store.store(MemoryRecord(id="bm25-2", content="Classical algorithms for graph traversal"))

    results = await store.search_lexical("superconducting qubits")
    assert len(results) >= 1
    assert results[0].id == "bm25-1"
    assert results[0].score > 0.0

    store.close()


@pytest.mark.asyncio
async def test_sqlite_wal_checkpoint_explicit_and_on_close(tmp_path: Path):
    """MEM-02: SQLite store must support explicit and on-close PRAGMA wal_checkpoint(PASSIVE)."""
    cfg = MemoryConfig(db_path=str(tmp_path / "checkpoint.db"))
    store = SQLiteMemoryStore(cfg)

    for i in range(10):
        await store.store(MemoryRecord(id=f"chk-{i}", content=f"Checkpoint record {i}"))

    # Explicit checkpoint call should succeed without raising
    store.checkpoint()

    # Verify records remain intact after checkpoint
    rec = await store.get("chk-0")
    assert rec is not None

    # Closing runs checkpoint and closes connection safely
    store.close()
    assert store._conn is None


@pytest.mark.asyncio
async def test_sqlite_periodic_wal_checkpoint(tmp_path: Path):
    """MEM-02: Store must trigger periodic checkpoint when write count exceeds threshold."""
    cfg = MemoryConfig(db_path=str(tmp_path / "periodic_chk.db"))
    store = SQLiteMemoryStore(cfg)
    store._checkpoint_interval = 5  # Set small interval for test

    # Write 6 items to trigger periodic checkpoint
    for i in range(6):
        await store.store(MemoryRecord(id=f"per-{i}", content=f"Periodic record {i}"))

    # Write count should have reset after passing interval 5
    assert store._write_count < 5

    store.close()


