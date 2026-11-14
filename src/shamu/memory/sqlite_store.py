"""Zero-daemon SQLite memory store with FTS5 lexical search and vector storage."""

import asyncio
import json
import math
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from shamu.config import MemoryConfig
from shamu.types import MemoryRecord, ChatMessage, Role, ToolCall


class SQLiteMemoryStore:
    """Thread-safe, zero-daemon SQLite storage engine with FTS5 BM25 and vector search."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        db_p = config.resolved_db_path if hasattr(config, "resolved_db_path") else Path(config.db_path)
        self.db_path = Path(db_p)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._fts_available = False
        self._write_count = 0
        self._checkpoint_interval = 50
        self._conn: Optional[sqlite3.Connection] = self._open_connection()
        self._init_db()

    def _open_connection(self) -> sqlite3.Connection:
        """Opens a SQLite connection configured with busy_timeout=30000 and WAL journal mode."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode with explicit transactions
        )
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            cur.execute("PRAGMA busy_timeout = 30000;")
            cur.execute("PRAGMA journal_mode = WAL;")
            cur.execute("PRAGMA synchronous = NORMAL;")
            cur.execute("PRAGMA foreign_keys = ON;")
        finally:
            cur.close()
        return conn

    def _init_db(self) -> None:
        """Initializes tables, PRAGMAs, and FTS5 virtual tables."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("PRAGMA busy_timeout = 30000;")
                cur.execute("PRAGMA journal_mode = WAL;")
                cur.execute("PRAGMA synchronous = NORMAL;")
                cur.execute("PRAGMA foreign_keys = ON;")

                # Base memories table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        category TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                """)

                # Embeddings storage table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                        vector BLOB NOT NULL,
                        dims INTEGER NOT NULL
                    );
                """)

                # Performance indexes for fast filtering and batch lookups
                cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_dims ON embeddings(dims);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);")

                # Active conversation session persistence table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS active_sessions (
                        session_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT,
                        name TEXT,
                        tool_calls TEXT,
                        tool_call_id TEXT,
                        created_at REAL NOT NULL,
                        PRIMARY KEY (session_id, sequence)
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_active_sessions_id ON active_sessions(session_id);")

                # Check FTS5 availability
                if self.config.fts_enabled:
                    try:
                        cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _test_fts USING fts5(txt);")
                        cur.execute("DROP TABLE _test_fts;")
                        # FTS5 works; create memories_fts virtual table
                        cur.execute("""
                            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                                id UNINDEXED,
                                content,
                                tokenize='porter unicode61'
                            );
                        """)
                        self._fts_available = True
                    except sqlite3.OperationalError:
                        self._fts_available = False
                else:
                    self._fts_available = False
            finally:
                cur.close()

    def is_fts_available(self) -> bool:
        """Returns True if FTS5 is active and available."""
        return self._fts_available

    def _sanitize_fts_query(self, query: str) -> Optional[str]:
        """Sanitizes user input to prevent FTS5 syntax errors with special chars and booleans."""
        if not query or not query.strip():
            return None

        # Extract alphanumeric terms preserving unicode
        tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
        # Exclude bare FTS5 reserved keywords if isolated
        reserved = {"NOT", "AND", "OR", "NEAR"}
        valid_terms = [t for t in tokens if t.upper() not in reserved]

        if not valid_terms:
            return None

        # Format as prefix tokens joined with OR so multi-word conversational queries rank by term overlap
        formatted = " OR ".join(f'"{term}"*' for term in valid_terms)
        return formatted

    async def store(self, record: MemoryRecord, embedding: Optional[List[float]] = None) -> str:
        """Stores or updates a memory record and its optional embedding vector."""
        return await asyncio.to_thread(self._sync_store, record, embedding)

    def _sync_store(self, record: MemoryRecord, embedding: Optional[List[float]] = None) -> str:
        metadata_str = json.dumps(record.metadata, ensure_ascii=False)
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(
                    """
                    INSERT INTO memories (id, content, source, category, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content=excluded.content,
                        source=excluded.source,
                        category=excluded.category,
                        metadata=excluded.metadata,
                        created_at=excluded.created_at;
                    """,
                    (record.id, record.content, record.source, record.category, metadata_str, record.created_at),
                )

                if self._fts_available:
                    # Update FTS5 index: delete existing then insert
                    cur.execute("DELETE FROM memories_fts WHERE id = ?;", (record.id,))
                    cur.execute(
                        "INSERT INTO memories_fts (id, content) VALUES (?, ?);",
                        (record.id, record.content),
                    )

                if embedding is not None and len(embedding) > 0:
                    blob = struct.pack(f"{len(embedding)}f", *embedding)
                    cur.execute(
                        """
                        INSERT INTO embeddings (memory_id, vector, dims)
                        VALUES (?, ?, ?)
                        ON CONFLICT(memory_id) DO UPDATE SET
                            vector=excluded.vector,
                            dims=excluded.dims;
                        """,
                        (record.id, blob, len(embedding)),
                    )
                cur.execute("COMMIT;")
                self._write_count += 1
                if self._write_count >= self._checkpoint_interval:
                    self._sync_checkpoint()
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                cur.close()
        return record.id

    async def store_batch(self, records: List[Any]) -> None:
        """Stores multiple memory records in a single database transaction."""
        await asyncio.to_thread(self._sync_store_batch, records)

    def _sync_store_batch(self, records: List[Any]) -> None:
        if not records:
            return
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                for item in records:
                    if isinstance(item, tuple):
                        record, embedding = item
                    else:
                        record, embedding = item, None

                    metadata_str = json.dumps(record.metadata or {})
                    cur.execute(
                        """
                        INSERT INTO memories (id, content, source, category, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            content=excluded.content,
                            source=excluded.source,
                            category=excluded.category,
                            metadata=excluded.metadata,
                            created_at=excluded.created_at;
                        """,
                        (record.id, record.content, record.source, record.category, metadata_str, record.created_at),
                    )

                    if self._fts_available:
                        cur.execute("DELETE FROM memories_fts WHERE id = ?;", (record.id,))
                        cur.execute(
                            "INSERT INTO memories_fts (id, content) VALUES (?, ?);",
                            (record.id, record.content),
                        )

                    if embedding is not None and len(embedding) > 0:
                        blob = struct.pack(f"{len(embedding)}f", *embedding)
                        cur.execute(
                            """
                            INSERT INTO embeddings (memory_id, vector, dims)
                            VALUES (?, ?, ?)
                            ON CONFLICT(memory_id) DO UPDATE SET
                                vector=excluded.vector,
                                dims=excluded.dims;
                            """,
                            (record.id, blob, len(embedding)),
                        )
                cur.execute("COMMIT;")
                self._write_count += 1
                if self._write_count >= self._checkpoint_interval:
                    self._sync_checkpoint()
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                cur.close()

    async def get(self, record_id: str) -> Optional[MemoryRecord]:
        """Retrieves a memory record by ID."""
        return await asyncio.to_thread(self._sync_get, record_id)

    def _sync_get(self, record_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute(
                    "SELECT id, content, source, category, metadata, created_at FROM memories WHERE id = ?;",
                    (record_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return MemoryRecord(
                    id=row["id"],
                    content=row["content"],
                    source=row["source"],
                    category=row["category"],
                    metadata=json.loads(row["metadata"]),
                    created_at=row["created_at"],
                )
            finally:
                cur.close()

    async def get_records_by_ids(self, record_ids: List[str]) -> List[MemoryRecord]:
        """Retrieves multiple memory records by their IDs in a single batch query (eliminating N+1 queries)."""
        return await asyncio.to_thread(self._sync_get_records_by_ids, record_ids)

    def _sync_get_records_by_ids(self, record_ids: List[str]) -> List[MemoryRecord]:
        """Synchronously retrieves multiple memory records using a chunked IN clause batch query."""
        if not record_ids:
            return []

        # Deduplicate while preserving order of first appearance
        unique_ids = list(dict.fromkeys(record_ids))
        records_by_id: Dict[str, MemoryRecord] = {}

        chunk_size = 500
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                for i in range(0, len(unique_ids), chunk_size):
                    chunk = unique_ids[i : i + chunk_size]
                    placeholders = ",".join("?" for _ in chunk)
                    query = f"""
                        SELECT id, content, source, category, metadata, created_at
                        FROM memories
                        WHERE id IN ({placeholders});
                    """
                    cur.execute(query, chunk)
                    rows = cur.fetchall()
                    for row in rows:
                        records_by_id[row["id"]] = MemoryRecord(
                            id=row["id"],
                            content=row["content"],
                            source=row["source"],
                            category=row["category"],
                            metadata=json.loads(row["metadata"]),
                            created_at=row["created_at"],
                        )
            finally:
                cur.close()

        results: List[MemoryRecord] = []
        for rid in unique_ids:
            if rid in records_by_id:
                results.append(records_by_id[rid])
        return results

    async def delete(self, record_id: str) -> bool:
        """Deletes a memory record and any associated embedding."""
        return await asyncio.to_thread(self._sync_delete, record_id)

    def _sync_delete(self, record_id: str) -> bool:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute("DELETE FROM memories WHERE id = ?;", (record_id,))
                changes = cur.rowcount
                if self._fts_available:
                    cur.execute("DELETE FROM memories_fts WHERE id = ?;", (record_id,))
                cur.execute("DELETE FROM embeddings WHERE memory_id = ?;", (record_id,))
                cur.execute("COMMIT;")
                if changes > 0:
                    self._write_count += 1
                    if self._write_count >= self._checkpoint_interval:
                        self._sync_checkpoint()
                return changes > 0
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                cur.close()

    async def search_lexical(self, query: str, limit: int = 10) -> List[MemoryRecord]:
        """Searches memories using BM25 ranking via FTS5, falling back to LIKE if unavailable."""
        return await asyncio.to_thread(self._sync_search_lexical, query, limit)

    def _sync_search_lexical(self, query: str, limit: int = 10) -> List[MemoryRecord]:
        if not query or not query.strip():
            return []

        if not self._fts_available:
            return self._sync_fallback_search(query, limit)

        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []

        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            # SQLite bm25() returns negative values where lower is better match.
            # We negate it so higher positive score is better match.
            # Column weights: 0.0 for unindexed id, 1.0 for content.
            query_sql = """
                SELECT
                    m.id, m.content, m.source, m.category, m.metadata, m.created_at,
                    -bm25(memories_fts, 0.0, 1.0) AS score
                FROM memories_fts f
                JOIN memories m ON f.id = m.id
                WHERE memories_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?;
            """
            try:
                cur.execute(query_sql, (sanitized, limit))
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                # Fallback to LIKE if FTS query syntax still fails
                return self._sync_fallback_search(query, limit)
            finally:
                cur.close()

            results: List[MemoryRecord] = []
            for row in rows:
                results.append(
                    MemoryRecord(
                        id=row["id"],
                        content=row["content"],
                        source=row["source"],
                        category=row["category"],
                        metadata=json.loads(row["metadata"]),
                        created_at=row["created_at"],
                        score=float(row["score"]),
                    )
                )
            return results

    def _sync_fallback_search(self, query: str, limit: int = 10) -> List[MemoryRecord]:
        tokens = [t for t in re.findall(r"[\w]+", query, flags=re.UNICODE) if len(t) > 1]
        if not tokens:
            return []

        pattern = f"%{tokens[0]}%"
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT id, content, source, category, metadata, created_at
                    FROM memories
                    WHERE content LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?;
                    """,
                    (pattern, limit),
                )
                rows = cur.fetchall()
            finally:
                cur.close()

            results = []
            for row in rows:
                results.append(
                    MemoryRecord(
                        id=row["id"],
                        content=row["content"],
                        source=row["source"],
                        category=row["category"],
                        metadata=json.loads(row["metadata"]),
                        created_at=row["created_at"],
                        score=1.0,
                    )
                )
            return results

    async def search_vector(self, query_vector: List[float], limit: int = 10) -> List[MemoryRecord]:
        """Performs brute-force in-memory cosine similarity search over stored vectors."""
        return await asyncio.to_thread(self._sync_search_vector, query_vector, limit)

    def _sync_search_vector(self, query_vector: List[float], limit: int = 10) -> List[MemoryRecord]:
        if not query_vector or limit <= 0:
            return []

        q_norm = math.sqrt(sum(x * x for x in query_vector))
        if q_norm == 0.0:
            return []

        dim = len(query_vector)

        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                # Efficient SQLite query filtering by dimension directly via index
                cur.execute("SELECT memory_id, vector FROM embeddings WHERE dims = ?;", (dim,))
                rows = cur.fetchall()
            finally:
                cur.close()

            if not rows:
                return []

            top_matches = []
            try:
                import numpy as np

                q_arr = np.array(query_vector, dtype=np.float32)
                q_norm_val = float(np.linalg.norm(q_arr))
                if q_norm_val == 0.0:
                    return []

                # Batched numpy matrix construction and BLAS-accelerated dot product
                raw_bytes = b"".join(r["vector"] for r in rows)
                matrix = np.frombuffer(raw_bytes, dtype=np.float32).reshape((len(rows), dim))
                dots = np.dot(matrix, q_arr)
                row_norms = np.linalg.norm(matrix, axis=1)
                denom = row_norms * q_norm_val
                denom = np.where(denom == 0.0, 1e-10, denom)
                sims = dots / denom

                # Fast top-k selection using argpartition to avoid full O(N log N) sort
                if len(rows) <= limit:
                    top_indices = np.argsort(-sims)
                else:
                    top_partition = np.argpartition(sims, -limit)[-limit:]
                    top_indices = top_partition[np.argsort(-sims[top_partition])]

                top_matches = [(rows[int(idx)]["memory_id"], float(sims[idx])) for idx in top_indices]
            except Exception:
                # Fallback to pure Python unpack loop if numpy unavailable or fails
                scored_ids = []
                for row in rows:
                    v = struct.unpack(f"{dim}f", row["vector"])
                    dot = sum(a * b for a, b in zip(query_vector, v))
                    v_norm = math.sqrt(sum(x * x for x in v))
                    sim = (dot / (q_norm * v_norm)) if (q_norm > 0 and v_norm > 0) else 0.0
                    scored_ids.append((row["memory_id"], sim))

                scored_ids.sort(key=lambda item: item[1], reverse=True)
                top_matches = scored_ids[:limit]

            if not top_matches:
                return []

            # Batch retrieve full records in 1 query instead of N individual queries (N+1 elimination)
            top_ids = [m[0] for m in top_matches]
            records = self._sync_get_records_by_ids(top_ids)
            records_map = {r.id: r for r in records}

            results: List[MemoryRecord] = []
            for mem_id, sim in top_matches:
                if mem_id in records_map:
                    rec = records_map[mem_id]
                    rec.score = sim
                    results.append(rec)
            return results

    async def get_all(self) -> List[MemoryRecord]:
        """Returns all stored memories."""
        return await asyncio.to_thread(self._sync_get_all)

    def _sync_get_all(self) -> List[MemoryRecord]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("SELECT id, content, source, category, metadata, created_at FROM memories ORDER BY created_at ASC;")
                rows = cur.fetchall()
            finally:
                cur.close()
            return [
                MemoryRecord(
                    id=row["id"],
                    content=row["content"],
                    source=row["source"],
                    category=row["category"],
                    metadata=json.loads(row["metadata"]),
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    async def save_session(self, session_id: str, messages: List[ChatMessage]) -> None:
        """Asynchronously persists active conversation messages for the given session_id."""
        await asyncio.to_thread(self._sync_save_session, session_id, messages)

    def _sync_save_session(self, session_id: str, messages: List[ChatMessage]) -> None:
        """Synchronously persists active conversation messages within an atomic SQLite transaction."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute("DELETE FROM active_sessions WHERE session_id = ?;", (session_id,))
                now = time.time()
                for seq, msg in enumerate(messages):
                    role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                    tool_calls_json = json.dumps([tc.model_dump() for tc in msg.tool_calls]) if msg.tool_calls else None
                    cur.execute(
                        """
                        INSERT INTO active_sessions (session_id, sequence, role, content, name, tool_calls, tool_call_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (session_id, seq, role_str, msg.content, msg.name, tool_calls_json, msg.tool_call_id, now),
                    )
                cur.execute("COMMIT;")
                self._write_count += 1
                if self._write_count >= self._checkpoint_interval:
                    self._sync_checkpoint()
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                cur.close()

    async def load_session(self, session_id: str) -> List[ChatMessage]:
        """Asynchronously retrieves active conversation messages for the given session_id ordered by sequence."""
        return await asyncio.to_thread(self._sync_load_session, session_id)

    def _sync_load_session(self, session_id: str) -> List[ChatMessage]:
        """Synchronously retrieves active conversation messages for the given session_id."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT sequence, role, content, name, tool_calls, tool_call_id
                    FROM active_sessions
                    WHERE session_id = ?
                    ORDER BY sequence ASC;
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
            finally:
                cur.close()

            messages: List[ChatMessage] = []
            for r in rows:
                tool_calls = None
                if r["tool_calls"]:
                    try:
                        raw_tc = json.loads(r["tool_calls"])
                        tool_calls = [ToolCall(**item) for item in raw_tc]
                    except Exception:
                        tool_calls = None

                role_val = r["role"]
                try:
                    role = Role(role_val)
                except ValueError:
                    role = Role.USER

                messages.append(
                    ChatMessage(
                        role=role,
                        content=r["content"],
                        name=r["name"],
                        tool_calls=tool_calls,
                        tool_call_id=r["tool_call_id"],
                    )
                )
            return messages

    async def clear_session(self, session_id: str) -> None:
        """Asynchronously clears persisted messages for the given session_id."""
        await asyncio.to_thread(self._sync_clear_session, session_id)

    def _sync_clear_session(self, session_id: str) -> None:
        """Synchronously clears persisted messages for the given session_id."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute("DELETE FROM active_sessions WHERE session_id = ?;", (session_id,))
                cur.execute("COMMIT;")
                self._write_count += 1
                if self._write_count >= self._checkpoint_interval:
                    self._sync_checkpoint()
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                cur.close()

    def checkpoint(self, mode: str = "PASSIVE") -> None:
        """Executes a WAL checkpoint (PRAGMA wal_checkpoint(PASSIVE))."""
        self._sync_checkpoint(mode=mode)

    def _sync_checkpoint(self, mode: str = "PASSIVE") -> None:
        with self._lock:
            if self._conn is not None:
                cur = None
                try:
                    cur = self._conn.cursor()
                    cur.execute(f"PRAGMA wal_checkpoint({mode});")
                    cur.fetchall()
                    self._write_count = 0
                except Exception:
                    pass
                finally:
                    if cur is not None:
                        try:
                            cur.close()
                        except Exception:
                            pass

    def close(self) -> None:
        """Closes the underlying SQLite connection after running a WAL checkpoint."""
        with self._lock:
            if self._conn is not None:
                try:
                    try:
                        self._sync_checkpoint(mode="TRUNCATE")
                    except Exception:
                        self._sync_checkpoint(mode="PASSIVE")
                except Exception:
                    pass
                try:
                    self._conn.close()
                except Exception:
                    pass
                finally:
                    self._conn = None

    def __enter__(self) -> "SQLiteMemoryStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            if hasattr(self, "_conn") and self._conn is not None:
                self.close()
        except Exception:
            pass
