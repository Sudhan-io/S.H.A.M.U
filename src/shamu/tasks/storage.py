"""SQLite persistence, task history, and crash recovery for S.H.A.M.U. Milestone 3."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Union

from shamu.tasks.models import (
    StepVerification,
    TaskPlan,
    TaskStatus,
    TaskStep,
    VerificationType,
)

logger = logging.getLogger(__name__)


class TaskStorage:
    """Thread-safe SQLite storage engine for TaskPlans and TaskSteps with crash recovery.

    Uses SQLite with Write-Ahead Logging (WAL) and PRAGMA busy_timeout = 30000 for concurrent
    access across GUI, CLI, and worker execution threads.
    """

    def __init__(self, db_path: Optional[Union[Path, str]] = None) -> None:
        """Initializes TaskStorage with a database file path or in-memory instance."""
        if db_path is None:
            self.db_path = Path.home() / ".shamu" / "tasks.db"
        elif str(db_path) == ":memory:":
            self.db_path = Path(":memory:")
        else:
            self.db_path = Path(db_path)

        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = self._open_connection()
        self._init_db()

    def _open_connection(self) -> sqlite3.Connection:
        """Opens SQLite connection configured with WAL mode and busy_timeout=30000."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit with explicit transactions
        )
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            cur.execute("PRAGMA busy_timeout = 30000;")
            if str(self.db_path) != ":memory:":
                cur.execute("PRAGMA journal_mode = WAL;")
            cur.execute("PRAGMA synchronous = NORMAL;")
            cur.execute("PRAGMA foreign_keys = ON;")
        finally:
            cur.close()
        return conn

    def _init_db(self) -> None:
        """Initializes database schema and indices if not present."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        objective TEXT NOT NULL,
                        status TEXT NOT NULL,
                        approved INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        summary TEXT DEFAULT '',
                        final_summary TEXT DEFAULT '',
                        execution_log TEXT DEFAULT '[]'
                    );
                    """
                )

                # Ensure backwards-compatible columns if table already existed
                cur.execute("PRAGMA table_info(tasks);")
                task_cols = [row["name"] for row in cur.fetchall()]
                if "summary" not in task_cols:
                    try:
                        cur.execute("ALTER TABLE tasks ADD COLUMN summary TEXT DEFAULT '';")
                    except Exception:
                        pass
                if "final_summary" not in task_cols:
                    try:
                        cur.execute("ALTER TABLE tasks ADD COLUMN final_summary TEXT DEFAULT '';")
                    except Exception:
                        pass
                if "execution_log" not in task_cols:
                    try:
                        cur.execute("ALTER TABLE tasks ADD COLUMN execution_log TEXT DEFAULT '[]';")
                    except Exception:
                        pass

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_steps (
                        id TEXT PRIMARY KEY,
                        plan_id TEXT NOT NULL,
                        step_index INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT DEFAULT '',
                        tool_name TEXT,
                        arguments_json TEXT DEFAULT '{}',
                        dependencies_json TEXT DEFAULT '[]',
                        status TEXT NOT NULL,
                        result TEXT DEFAULT '',
                        error TEXT DEFAULT '',
                        verification_json TEXT,
                        started_at REAL,
                        completed_at REAL,
                        FOREIGN KEY (plan_id) REFERENCES tasks (id) ON DELETE CASCADE
                    );
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);")
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_steps_plan_id ON task_steps(plan_id, step_index);"
                )
                cur.execute("COMMIT;")
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                cur.close()

    def _row_to_step(self, s_row: sqlite3.Row) -> TaskStep:
        """Reconstructs a TaskStep from a database row."""
        ver: Optional[StepVerification] = None
        if s_row["verification_json"]:
            try:
                v_data = json.loads(s_row["verification_json"])
                v_type_raw = v_data.get("type", "file_exists")
                try:
                    v_type = VerificationType(v_type_raw)
                except ValueError:
                    v_type = VerificationType(str(v_type_raw).lower())

                ver = StepVerification(
                    type=v_type,
                    target=v_data.get("target", ""),
                    expected=v_data.get("expected"),
                    passed=v_data.get("passed"),
                    message=v_data.get("message"),
                )
            except Exception as e:
                logger.warning("Failed to deserialize verification for step %s: %s", s_row["id"], e)
                ver = None

        status_val = s_row["status"]
        try:
            status = TaskStatus(status_val)
        except ValueError:
            try:
                status = TaskStatus(status_val.lower())
            except ValueError:
                status = TaskStatus.PENDING

        args = {}
        if s_row["arguments_json"]:
            try:
                args = json.loads(s_row["arguments_json"])
            except Exception:
                args = {}

        deps = []
        if s_row["dependencies_json"]:
            try:
                deps = json.loads(s_row["dependencies_json"])
            except Exception:
                deps = []

        return TaskStep(
            id=s_row["id"],
            title=s_row["title"],
            description=s_row["description"] or "",
            tool_name=s_row["tool_name"],
            arguments=args,
            dependencies=deps,
            status=status,
            result=s_row["result"] if s_row["result"] != "" else None,
            error=s_row["error"] if s_row["error"] != "" else None,
            verification=ver,
            started_at=s_row["started_at"],
            completed_at=s_row["completed_at"],
        )

    def _row_to_plan(self, row: sqlite3.Row, steps: List[TaskStep]) -> TaskPlan:
        """Reconstructs a TaskPlan from a tasks table row and step list."""
        exec_log: List[str] = []
        if "execution_log" in row.keys() and row["execution_log"]:
            try:
                exec_log = json.loads(row["execution_log"])
            except Exception:
                exec_log = []

        status_val = row["status"]
        try:
            status = TaskStatus(status_val)
        except ValueError:
            try:
                status = TaskStatus(status_val.lower())
            except ValueError:
                status = TaskStatus.PENDING

        summary_val = ""
        if "final_summary" in row.keys() and row["final_summary"]:
            summary_val = row["final_summary"]
        elif "summary" in row.keys() and row["summary"]:
            summary_val = row["summary"]

        return TaskPlan(
            id=row["id"],
            objective=row["objective"],
            steps=steps,
            status=status,
            approved=bool(row["approved"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            execution_log=exec_log,
            final_summary=summary_val or None,
        )

    def save_plan(self, plan: TaskPlan) -> None:
        """Persists or updates an entire TaskPlan and all its steps."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                if isinstance(plan.status, TaskStatus):
                    status_str = plan.status.value
                else:
                    try:
                        status_str = TaskStatus(str(plan.status).lower()).value
                    except ValueError:
                        status_str = str(plan.status).lower()
                summary_str = plan.final_summary or ""

                # Upsert task plan
                cur.execute(
                    """
                    INSERT INTO tasks (
                        id, objective, status, approved, created_at, updated_at, execution_log, summary, final_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        objective = excluded.objective,
                        status = excluded.status,
                        approved = excluded.approved,
                        updated_at = excluded.updated_at,
                        execution_log = excluded.execution_log,
                        summary = excluded.summary,
                        final_summary = excluded.final_summary;
                    """,
                    (
                        plan.id,
                        plan.objective,
                        status_str,
                        1 if plan.approved else 0,
                        plan.created_at,
                        plan.updated_at,
                        json.dumps(plan.execution_log or []),
                        summary_str,
                        summary_str,
                    ),
                )

                # Delete steps that might no longer exist
                current_step_ids = [s.id for s in plan.steps]
                if current_step_ids:
                    placeholders = ",".join("?" for _ in current_step_ids)
                    cur.execute(
                        f"DELETE FROM task_steps WHERE plan_id = ? AND id NOT IN ({placeholders});",
                        [plan.id] + current_step_ids,
                    )
                else:
                    cur.execute("DELETE FROM task_steps WHERE plan_id = ?;", (plan.id,))

                # Upsert each step
                for idx, step in enumerate(plan.steps):
                    ver_json = None
                    if step.verification:
                        ver_json = json.dumps(
                            {
                                "type": (
                                    step.verification.type.value
                                    if isinstance(step.verification.type, VerificationType)
                                    else str(step.verification.type)
                                ),
                                "target": step.verification.target,
                                "expected": step.verification.expected,
                                "passed": step.verification.passed,
                                "message": step.verification.message,
                            }
                        )

                    if isinstance(step.status, TaskStatus):
                        step_status_str = step.status.value
                    else:
                        try:
                            step_status_str = TaskStatus(str(step.status).lower()).value
                        except ValueError:
                            step_status_str = str(step.status).lower()

                    cur.execute(
                        """
                        INSERT INTO task_steps (
                            id, plan_id, step_index, title, description, tool_name,
                            arguments_json, dependencies_json, status, result, error,
                            verification_json, started_at, completed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            plan_id = excluded.plan_id,
                            step_index = excluded.step_index,
                            title = excluded.title,
                            description = excluded.description,
                            tool_name = excluded.tool_name,
                            arguments_json = excluded.arguments_json,
                            dependencies_json = excluded.dependencies_json,
                            status = excluded.status,
                            result = excluded.result,
                            error = excluded.error,
                            verification_json = excluded.verification_json,
                            started_at = excluded.started_at,
                            completed_at = excluded.completed_at;
                        """,
                        (
                            step.id,
                            plan.id,
                            idx,
                            step.title,
                            step.description or "",
                            step.tool_name,
                            json.dumps(step.arguments or {}),
                            json.dumps(step.dependencies or []),
                            step_status_str,
                            step.result if step.result is not None else "",
                            step.error if step.error is not None else "",
                            ver_json,
                            step.started_at,
                            step.completed_at,
                        ),
                    )

                cur.execute("COMMIT;")
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                cur.close()

    def get_plan(self, plan_id: str) -> Optional[TaskPlan]:
        """Retrieves a full TaskPlan and its reconstructed steps by ID."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("SELECT * FROM tasks WHERE id = ?;", (plan_id,))
                row = cur.fetchone()
                if not row:
                    return None

                cur.execute(
                    "SELECT * FROM task_steps WHERE plan_id = ? ORDER BY step_index ASC;",
                    (plan_id,),
                )
                step_rows = cur.fetchall()
                steps = [self._row_to_step(s_row) for s_row in step_rows]
                return self._row_to_plan(row, steps)
            finally:
                cur.close()

    def list_plans(
        self,
        status: Optional[Union[TaskStatus, str]] = None,
        limit: int = 50,
    ) -> List[TaskPlan]:
        """Lists plans matching status filter, ordered newest first."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                if status is not None:
                    if isinstance(status, TaskStatus):
                        status_str = status.value
                    else:
                        try:
                            status_str = TaskStatus(str(status).lower()).value
                        except ValueError:
                            status_str = str(status).lower()
                    cur.execute(
                        "SELECT id FROM tasks WHERE LOWER(status) = LOWER(?) ORDER BY created_at DESC LIMIT ?;",
                        (status_str, limit),
                    )
                else:
                    cur.execute(
                        "SELECT id FROM tasks ORDER BY created_at DESC LIMIT ?;",
                        (limit,),
                    )
                ids = [r["id"] for r in cur.fetchall()]
            finally:
                cur.close()

        plans: List[TaskPlan] = []
        for pid in ids:
            p = self.get_plan(pid)
            if p:
                plans.append(p)
        return plans

    def update_step(self, step: TaskStep, plan_id: Optional[str] = None) -> bool:
        """Updates status, result, error, and verification outcome of a step."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                ver_json = None
                if step.verification:
                    ver_json = json.dumps(
                        {
                            "type": (
                                step.verification.type.value
                                if isinstance(step.verification.type, VerificationType)
                                else str(step.verification.type)
                            ),
                            "target": step.verification.target,
                            "expected": step.verification.expected,
                            "passed": step.verification.passed,
                            "message": step.verification.message,
                        }
                    )

                if isinstance(step.status, TaskStatus):
                    step_status_str = step.status.value
                else:
                    try:
                        step_status_str = TaskStatus(str(step.status).lower()).value
                    except ValueError:
                        step_status_str = str(step.status).lower()

                cur.execute(
                    """
                    UPDATE task_steps SET
                        status = ?,
                        result = ?,
                        error = ?,
                        verification_json = ?,
                        started_at = ?,
                        completed_at = ?
                    WHERE id = ?;
                    """,
                    (
                        step_status_str,
                        step.result if step.result is not None else "",
                        step.error if step.error is not None else "",
                        ver_json,
                        step.started_at,
                        step.completed_at,
                        step.id,
                    ),
                )
                updated = cur.rowcount > 0

                # Update updated_at on the parent task
                now = time.time()
                if plan_id:
                    cur.execute("UPDATE tasks SET updated_at = ? WHERE id = ?;", (now, plan_id))
                else:
                    cur.execute(
                        """
                        UPDATE tasks SET updated_at = ?
                        WHERE id = (SELECT plan_id FROM task_steps WHERE id = ?);
                        """,
                        (now, step.id),
                    )

                cur.execute("COMMIT;")
                return updated
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                cur.close()

    def update_plan_status(
        self,
        plan_id: str,
        status: Union[TaskStatus, str],
        summary: Optional[str] = None,
    ) -> bool:
        """Updates plan status and summary."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                if isinstance(status, TaskStatus):
                    status_str = status.value
                else:
                    try:
                        status_str = TaskStatus(str(status).lower()).value
                    except ValueError:
                        status_str = str(status).lower()
                now = time.time()
                cur.execute("BEGIN IMMEDIATE;")
                if summary is not None:
                    cur.execute(
                        """
                        UPDATE tasks SET status = ?, summary = ?, final_summary = ?, updated_at = ? WHERE id = ?;
                        """,
                        (status_str, summary, summary, now, plan_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?;
                        """,
                        (status_str, now, plan_id),
                    )
                updated = cur.rowcount > 0
                cur.execute("COMMIT;")
                return updated
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                cur.close()

    def get_step(self, step_id: str) -> Optional[TaskStep]:
        """Retrieves a single step by ID."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("SELECT * FROM task_steps WHERE id = ?;", (step_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_step(row)
            finally:
                cur.close()

    def delete_plan(self, plan_id: str) -> bool:
        """Deletes a plan and cascades to its steps."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute("DELETE FROM task_steps WHERE plan_id = ?;", (plan_id,))
                cur.execute("DELETE FROM tasks WHERE id = ?;", (plan_id,))
                deleted = cur.rowcount > 0
                cur.execute("COMMIT;")
                return deleted
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                cur.close()

    def recover_interrupted_plans(self) -> List[TaskPlan]:
        """Queries for plans left in IN_PROGRESS or WAITING_APPROVAL after an abnormal termination.

        For each interrupted plan, resets steps stuck in IN_PROGRESS or WAITING_APPROVAL back to PENDING
        so that execution can be cleanly resumed from where it stopped without repeating completed steps.
        If all steps in a plan were already completed, updates the plan status to COMPLETED.
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cur = self._conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT id FROM tasks
                    WHERE LOWER(status) IN ('in_progress', 'waiting_approval')
                    ORDER BY updated_at DESC;
                    """
                )
                plan_ids = [r["id"] for r in cur.fetchall()]
            finally:
                cur.close()

        recovered_plans: List[TaskPlan] = []
        for pid in plan_ids:
            plan = self.get_plan(pid)
            if not plan:
                continue

            # If all steps were already completed prior to interruption, mark plan COMPLETED
            if plan.is_successful():
                plan.status = TaskStatus.COMPLETED
                plan.updated_at = time.time()
                self.save_plan(plan)
                continue

            # If all steps have already reached terminal state, finalize plan status
            if plan.is_complete():
                plan.status = TaskStatus.COMPLETED if plan.is_successful() else TaskStatus.FAILED
                plan.updated_at = time.time()
                self.save_plan(plan)
                continue

            needs_save = False
            for step in plan.steps:
                # Reset steps that were interrupted mid-execution or while awaiting approval
                if step.status == TaskStatus.IN_PROGRESS:
                    step.status = TaskStatus.PENDING
                    step.error = "Execution interrupted by system exit. Ready to resume."
                    needs_save = True
                elif step.status == TaskStatus.WAITING_APPROVAL:
                    step.status = TaskStatus.PENDING
                    step.error = "Execution interrupted while waiting for approval. Ready to resume."
                    needs_save = True

            if needs_save:
                plan.updated_at = time.time()
                self.save_plan(plan)

            recovered_plans.append(plan)

        return recovered_plans

    def close(self) -> None:
        """Closes the underlying SQLite connection."""
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def __enter__(self) -> TaskStorage:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
