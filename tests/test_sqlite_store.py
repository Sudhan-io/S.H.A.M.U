"""Tests for SQLiteMemoryStore reliability, WAL mode, busy timeout, and multi-threaded concurrency."""

import asyncio
import concurrent.futures
import sqlite3
import threading
import time
from pathlib import Path
from typing import List

import pytest

from shamu.config import MemoryConfig
from shamu.memory.sqlite_store import SQLiteMemoryStore
from shamu.types import MemoryRecord


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_concurrency_memory.db"


@pytest.fixture
def memory_config(temp_db_path: Path) -> MemoryConfig:
    return MemoryConfig(db_path=str(temp_db_path), fts_enabled=True)


def test_sqlite_pragmas_busy_timeout_and_wal(memory_config: MemoryConfig):
    """CANON-P2-02: Verify PRAGMA busy_timeout = 30000 and PRAGMA journal_mode = WAL are enforced."""
    store = SQLiteMemoryStore(memory_config)
    assert store._conn is not None

    cur = store._conn.cursor()
    try:
        cur.execute("PRAGMA busy_timeout;")
        busy_timeout = cur.fetchone()[0]
        assert busy_timeout == 30000, f"Expected busy_timeout 30000, got {busy_timeout}"

        cur.execute("PRAGMA journal_mode;")
        journal_mode = cur.fetchone()[0].lower()
        assert journal_mode == "wal", f"Expected journal_mode wal, got {journal_mode}"

        cur.execute("PRAGMA synchronous;")
        synchronous = cur.fetchone()[0]
        assert synchronous == 1  # NORMAL = 1

        cur.execute("PRAGMA foreign_keys;")
        foreign_keys = cur.fetchone()[0]
        assert foreign_keys == 1  # ON = 1
    finally:
        cur.close()
        store.close()


def test_checkpoint_lifecycle_and_cleanups(memory_config: MemoryConfig):
    """CANON-P2-02: Verify checkpoint lifecycle, connection cleanup, and context manager."""
    with SQLiteMemoryStore(memory_config) as store:
        for i in range(10):
            store._sync_store(MemoryRecord(id=f"chk-{i}", content=f"Checkpoint record {i}"))

        # Explicit checkpoint calls (PASSIVE and TRUNCATE)
        store.checkpoint("PASSIVE")
        assert store._write_count == 0

        store.checkpoint("TRUNCATE")
        assert store._write_count == 0

        # Retrieve record to verify database integrity
        rec = store._sync_get("chk-0")
        assert rec is not None
        assert rec.id == "chk-0"

    # Context manager exit should close connection cleanly
    assert store._conn is None

    # Operations on closed store should raise RuntimeError
    with pytest.raises(RuntimeError, match="Database connection is closed"):
        store._sync_get("chk-0")

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        store._sync_store(MemoryRecord(id="chk-fail", content="fail"))


def test_concurrent_multithreaded_shared_store(memory_config: MemoryConfig):
    """CANON-P2-02: Multi-threaded concurrent reads and writes using a shared SQLiteMemoryStore instance."""
    store = SQLiteMemoryStore(memory_config)
    errors: List[Exception] = []

    def worker(worker_id: int):
        try:
            for i in range(20):
                rec_id = f"shared-{worker_id}-{i}"
                store._sync_store(
                    MemoryRecord(
                        id=rec_id,
                        content=f"Worker {worker_id} payload iteration {i} searchtoken",
                        category="concurrency",
                    )
                )
                retrieved = store._sync_get(rec_id)
                assert retrieved is not None
                assert retrieved.id == rec_id

                if i % 5 == 0:
                    results = store._sync_search_lexical("searchtoken", limit=5)
                    assert len(results) > 0
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    store.close()
    assert len(errors) == 0, f"Encountered concurrency errors: {errors}"


def test_concurrent_multithreaded_independent_connections_no_lock(memory_config: MemoryConfig):
    """CANON-P2-02: Multi-threaded concurrent writes across independent connections/instances to same DB.

    Without busy_timeout and WAL mode, SQLite raises sqlite3.OperationalError: database is locked.
    With busy_timeout = 30000 and WAL mode, all threads safely serialize their write transactions.
    """
    # Initialize DB schema first
    init_store = SQLiteMemoryStore(memory_config)
    init_store.close()

    errors: List[Exception] = []
    num_threads = 8
    writes_per_thread = 20

    def worker(worker_id: int):
        store = SQLiteMemoryStore(memory_config)
        try:
            for i in range(writes_per_thread):
                rec_id = f"indep-{worker_id}-{i}"
                store._sync_store(
                    MemoryRecord(
                        id=rec_id,
                        content=f"Independent worker {worker_id} writing token_{worker_id} fact {i}",
                        category="stress",
                    )
                )
                # Verify read-your-own-writes immediately
                rec = store._sync_get(rec_id)
                assert rec is not None
                assert rec.id == rec_id

                # Periodic checkpoint from concurrent threads
                if i % 7 == 0:
                    store.checkpoint("PASSIVE")
        except Exception as e:
            errors.append(e)
        finally:
            store.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, w) for w in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert len(errors) == 0, f"Independent connection concurrency errors: {errors}"

    # Verify total count of persisted records across all threads
    verify_store = SQLiteMemoryStore(memory_config)
    all_recs = verify_store._sync_get_all()
    verify_store.close()

    expected_total = num_threads * writes_per_thread
    assert len(all_recs) == expected_total, f"Expected {expected_total} records, got {len(all_recs)}"


def test_concurrent_readers_during_heavy_writes(memory_config: MemoryConfig):
    """CANON-P2-02: Continuous readers never suffer 'database is locked' during heavy writes in WAL mode."""
    store = SQLiteMemoryStore(memory_config)
    stop_event = threading.Event()
    read_errors: List[Exception] = []
    write_errors: List[Exception] = []
    reads_completed = [0]

    def writer():
        writer_store = SQLiteMemoryStore(memory_config)
        try:
            for i in range(50):
                writer_store._sync_store(
                    MemoryRecord(
                        id=f"writer-{i}",
                        content=f"Continuous stream of data entry {i} keyword",
                    )
                )
                time.sleep(0.002)
        except Exception as e:
            write_errors.append(e)
        finally:
            writer_store.close()
            stop_event.set()

    def reader(reader_id: int):
        reader_store = SQLiteMemoryStore(memory_config)
        try:
            while not stop_event.is_set():
                reader_store._sync_search_lexical("keyword", limit=10)
                reads_completed[0] += 1
                time.sleep(0.001)
        except Exception as e:
            read_errors.append(e)
        finally:
            reader_store.close()

    writer_thread = threading.Thread(target=writer)
    reader_threads = [threading.Thread(target=reader, args=(r,)) for r in range(4)]

    for rt in reader_threads:
        rt.start()
    writer_thread.start()

    writer_thread.join(timeout=10.0)
    stop_event.set()
    for rt in reader_threads:
        rt.join(timeout=5.0)

    store.close()

    assert len(write_errors) == 0, f"Write errors: {write_errors}"
    assert len(read_errors) == 0, f"Read errors: {read_errors}"
    assert reads_completed[0] > 0, "Expected readers to execute queries concurrently"


@pytest.mark.asyncio
async def test_async_concurrent_store_and_search(memory_config: MemoryConfig):
    """CANON-P2-02: Async concurrency via asyncio.gather on SQLiteMemoryStore."""
    store = SQLiteMemoryStore(memory_config)

    async def store_task(task_id: int):
        for i in range(10):
            await store.store(
                MemoryRecord(id=f"async-{task_id}-{i}", content=f"Async content {task_id}-{i} testasync")
            )

    async def read_task():
        for _ in range(10):
            await store.search_lexical("testasync", limit=5)
            await asyncio.sleep(0.001)

    tasks = [store_task(t) for t in range(5)] + [read_task() for _ in range(3)]
    await asyncio.gather(*tasks)

    all_records = await store.get_all()
    assert len(all_records) == 50
    store.close()


@pytest.mark.asyncio
async def test_batch_get_records_by_ids_correctness(memory_config: MemoryConfig):
    """CANON-P2-04: Batch fetching get_records_by_ids correctness, ordering, and chunking."""
    store = SQLiteMemoryStore(memory_config)

    # 1. Empty list
    assert await store.get_records_by_ids([]) == []

    # 2. Store multiple records (> 500 to test chunking across SQLite parameter limit)
    total_records = 600
    batch_records = [
        MemoryRecord(
            id=f"batch-rec-{i}",
            content=f"Knowledge record payload {i}",
            category="knowledge",
            metadata={"index": i},
        )
        for i in range(total_records)
    ]
    await store.store_batch(batch_records)

    # 3. Retrieve specific subset with arbitrary order and duplicates
    target_ids = ["batch-rec-42", "batch-rec-100", "batch-rec-42", "batch-rec-999", "batch-rec-5"]
    fetched = await store.get_records_by_ids(target_ids)

    # Verify deduplication, ordering, and skipping non-existent IDs
    assert [r.id for r in fetched] == ["batch-rec-42", "batch-rec-100", "batch-rec-5"]
    assert fetched[0].metadata == {"index": 42}
    assert fetched[1].metadata == {"index": 100}
    assert fetched[2].metadata == {"index": 5}

    # 4. Fetch all 600 records across chunk boundaries in a single call
    all_ids = [f"batch-rec-{i}" for i in range(total_records)]
    fetched_all = await store.get_records_by_ids(all_ids)
    assert len(fetched_all) == total_records
    assert fetched_all[0].id == "batch-rec-0"
    assert fetched_all[-1].id == f"batch-rec-{total_records - 1}"

    store.close()


@pytest.mark.asyncio
async def test_search_vector_eliminates_n_plus_one_query(memory_config: MemoryConfig, monkeypatch: pytest.MonkeyPatch):
    """CANON-P2-04: Verify search_vector uses batch fetching and eliminates N+1 queries."""
    store = SQLiteMemoryStore(memory_config)

    # Store 20 records with embeddings
    records = []
    for i in range(20):
        rec = MemoryRecord(id=f"vec-item-{i}", content=f"Vector item content {i}")
        vec = [float(i + j) for j in range(4)]
        records.append((rec, vec))
    await store.store_batch(records)

    # Patch _sync_get to ensure it is NOT called during search_vector
    def forbidden_sync_get(*args, **kwargs):
        pytest.fail("N+1 query detected: _sync_get was called during search_vector!")

    monkeypatch.setattr(store, "_sync_get", forbidden_sync_get)

    # Run vector search
    query_vec = [1.0, 2.0, 3.0, 4.0]
    results = await store.search_vector(query_vec, limit=5)

    assert len(results) == 5
    assert all(r.score is not None for r in results)
    # Cosine similarity is monotonically sorted
    assert results[0].score >= results[1].score >= results[2].score

    store.close()


@pytest.mark.asyncio
async def test_memory_retrieval_performance_under_1000_records(memory_config: MemoryConfig):
    """CANON-P2-04: Verify search_vector and batch retrieval under 1,000+ records remain sub-10ms."""
    store = SQLiteMemoryStore(memory_config)

    total_records = 1200
    dim = 64
    import numpy as np

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((total_records, dim)).astype(np.float32)

    records = [
        (
            MemoryRecord(
                id=f"perf-{i}",
                content=f"Sub-10ms performance verification record number {i}",
                category="perf_test",
                metadata={"seq": i},
            ),
            embeddings[i].tolist(),
        )
        for i in range(total_records)
    ]
    await store.store_batch(records)

    query_vec = rng.standard_normal(dim).astype(np.float32).tolist()

    # Warmup
    await store.search_vector(query_vec, limit=10)

    # Benchmark search_vector over 1,200 records
    iterations = 20
    latencies: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        results = await store.search_vector(query_vec, limit=10)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        assert len(results) == 10

    avg_latency = sum(latencies) / len(latencies)
    sorted_latencies = sorted(latencies)
    median_latency = sorted_latencies[len(sorted_latencies) // 2]

    # CANON-P2-04 requirement: sub-10ms retrieval under 1000+ records
    assert avg_latency < 10.0, f"Average vector search latency {avg_latency:.2f}ms exceeded 10ms threshold"
    assert median_latency < 10.0, f"Median vector search latency {median_latency:.2f}ms exceeded 10ms threshold"

    # Also benchmark batch get_records_by_ids for 50 records
    sample_ids = [f"perf-{i * 20}" for i in range(50)]
    batch_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        batch_results = await store.get_records_by_ids(sample_ids)
        t1 = time.perf_counter()
        batch_latencies.append((t1 - t0) * 1000.0)
        assert len(batch_results) == 50

    avg_batch_latency = sum(batch_latencies) / len(batch_latencies)
    assert avg_batch_latency < 10.0, f"Average batch retrieval latency {avg_batch_latency:.2f}ms exceeded 10ms"

    store.close()
