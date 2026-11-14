"""Unit and integration tests for TaskStorage, SQLite schema, CRUD operations, and crash recovery."""

import sqlite3
import threading
from pathlib import Path
import pytest

from shamu.tasks.models import (
    StepVerification,
    TaskPlan,
    TaskStatus,
    TaskStep,
    VerificationType,
)
from shamu.tasks.storage import TaskStorage


@pytest.fixture
def storage(tmp_path: Path):
    """Provides a fresh isolated TaskStorage instance."""
    db_file = tmp_path / "tasks_test.db"
    store = TaskStorage(db_path=db_file)
    yield store
    store.close()


def test_sqlite_schema_and_pragmas(storage: TaskStorage):
    """TaskStorage configures WAL mode, busy_timeout, and required schema."""
    conn = storage._conn
    assert conn is not None

    cur = conn.cursor()
    # Check journal mode
    cur.execute("PRAGMA journal_mode;")
    mode = cur.fetchone()[0]
    assert mode.lower() == "wal"

    # Check busy timeout
    cur.execute("PRAGMA busy_timeout;")
    timeout = cur.fetchone()[0]
    assert timeout == 30000

    # Check tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    assert "tasks" in tables
    assert "task_steps" in tables


def test_save_and_get_plan_roundtrip(storage: TaskStorage):
    """TaskPlan and TaskSteps serialize and deserialize faithfully."""
    v = StepVerification(
        type=VerificationType.FILE_CONTAINS,
        target="data/test.txt",
        expected="HELLO",
        passed=True,
        message="Verification ok",
    )
    s1 = TaskStep(
        id="step-1",
        title="Create initial file",
        description="Write text",
        tool_name="write_file",
        arguments={"path": "data/test.txt", "content": "HELLO WORLD"},
        dependencies=[],
        status=TaskStatus.COMPLETED,
        result="Success",
        error=None,
        verification=v,
    )
    s2 = TaskStep(
        id="step-2",
        title="Check status",
        description="Verify output",
        tool_name="run_command",
        arguments={"command": "dir"},
        dependencies=["step-1"],
        status=TaskStatus.PENDING,
    )
    plan = TaskPlan(
        id="plan-roundtrip-1",
        objective="Verify roundtrip serialization",
        steps=[s1, s2],
        status=TaskStatus.IN_PROGRESS,
        approved=True,
        execution_log=["Started step 1", "Completed step 1"],
    )

    storage.save_plan(plan)
    loaded = storage.get_plan("plan-roundtrip-1")

    assert loaded is not None
    assert loaded.id == "plan-roundtrip-1"
    assert loaded.objective == "Verify roundtrip serialization"
    assert loaded.status == TaskStatus.IN_PROGRESS
    assert loaded.approved is True
    assert loaded.execution_log == ["Started step 1", "Completed step 1"]

    assert len(loaded.steps) == 2
    step1 = loaded.steps[0]
    assert step1.id == "step-1"
    assert step1.tool_name == "write_file"
    assert step1.arguments == {"path": "data/test.txt", "content": "HELLO WORLD"}
    assert step1.status == TaskStatus.COMPLETED
    assert step1.result == "Success"
    assert step1.verification is not None
    assert step1.verification.type == VerificationType.FILE_CONTAINS
    assert step1.verification.passed is True

    step2 = loaded.steps[1]
    assert step2.id == "step-2"
    assert step2.dependencies == ["step-1"]
    assert step2.status == TaskStatus.PENDING


def test_get_nonexistent_plan(storage: TaskStorage):
    """Retrieving a non-existent plan ID returns None."""
    assert storage.get_plan("does-not-exist") is None


def test_update_plan_and_steps(storage: TaskStorage):
    """Updating plan status and individual steps."""
    step = TaskStep(id="s1", title="Step 1", status=TaskStatus.PENDING)
    plan = TaskPlan(id="p1", objective="Update test", steps=[step])
    storage.save_plan(plan)

    # Update step directly
    step.status = TaskStatus.COMPLETED
    step.result = "Done!"
    storage.update_step(step, plan_id="p1")

    reloaded = storage.get_plan("p1")
    assert reloaded.steps[0].status == TaskStatus.COMPLETED
    assert reloaded.steps[0].result == "Done!"

    # Update plan status directly
    storage.update_plan_status("p1", TaskStatus.COMPLETED, summary="All steps finished.")
    reloaded2 = storage.get_plan("p1")
    assert reloaded2.status == TaskStatus.COMPLETED
    assert reloaded2.final_summary == "All steps finished."


def test_list_plans_and_filtering(storage: TaskStorage):
    """Listing plans with status filtering and limit."""
    p_completed = TaskPlan(id="p-c", objective="Completed plan", status=TaskStatus.COMPLETED)
    p_progress = TaskPlan(id="p-p", objective="In progress plan", status=TaskStatus.IN_PROGRESS)
    p_failed = TaskPlan(id="p-f", objective="Failed plan", status=TaskStatus.FAILED)

    storage.save_plan(p_completed)
    storage.save_plan(p_progress)
    storage.save_plan(p_failed)

    all_plans = storage.list_plans()
    assert len(all_plans) == 3

    in_progress_only = storage.list_plans(status=TaskStatus.IN_PROGRESS)
    assert len(in_progress_only) == 1
    assert in_progress_only[0].id == "p-p"

    limited = storage.list_plans(limit=2)
    assert len(limited) == 2


def test_delete_plan(storage: TaskStorage):
    """Deleting a plan removes it and cascades to task_steps."""
    step = TaskStep(id="s-del", title="Delete step")
    plan = TaskPlan(id="p-del", objective="To delete", steps=[step])
    storage.save_plan(plan)

    assert storage.get_plan("p-del") is not None
    assert storage.delete_plan("p-del") is True
    assert storage.get_plan("p-del") is None
    assert storage.delete_plan("p-del") is False


def test_interrupted_execution_recovery(storage: TaskStorage):
    """Crash recovery identifies unfinished plans and resets interrupted steps to PENDING."""
    # Step 1 was completed before crash
    s1 = TaskStep(id="s1", title="Step 1", status=TaskStatus.COMPLETED, result="Success 1")
    # Step 2 was in progress when crash occurred
    s2 = TaskStep(id="s2", title="Step 2", status=TaskStatus.IN_PROGRESS, dependencies=["s1"])
    # Step 3 was waiting for Step 2
    s3 = TaskStep(id="s3", title="Step 3", status=TaskStatus.PENDING, dependencies=["s2"])

    interrupted_plan = TaskPlan(
        id="crashed-plan-1",
        objective="Recover from crash",
        steps=[s1, s2, s3],
        status=TaskStatus.IN_PROGRESS,
        approved=True,
    )
    storage.save_plan(interrupted_plan)

    # Another plan that completed normally
    completed_plan = TaskPlan(
        id="normal-plan",
        objective="Already done",
        status=TaskStatus.COMPLETED,
    )
    storage.save_plan(completed_plan)

    # Recover interrupted plans
    recovered = storage.recover_interrupted_plans()
    assert len(recovered) == 1
    rec_plan = recovered[0]
    assert rec_plan.id == "crashed-plan-1"

    # Step 1 stays COMPLETED
    assert rec_plan.steps[0].status == TaskStatus.COMPLETED
    assert rec_plan.steps[0].result == "Success 1"

    # Step 2 was reset to PENDING with recovery note
    assert rec_plan.steps[1].status == TaskStatus.PENDING
    assert "interrupted" in rec_plan.steps[1].error.lower()

    # Step 3 stays PENDING
    assert rec_plan.steps[2].status == TaskStatus.PENDING

    # Verify persisted in SQLite
    persisted = storage.get_plan("crashed-plan-1")
    assert persisted.steps[1].status == TaskStatus.PENDING


def test_concurrent_access_threads(storage: TaskStorage):
    """TaskStorage is thread-safe across concurrent threads."""
    errors = []

    def worker(worker_id: int):
        try:
            for i in range(10):
                pid = f"thread-{worker_id}-plan-{i}"
                plan = TaskPlan(
                    id=pid,
                    objective=f"Worker {worker_id} plan {i}",
                    steps=[TaskStep(id=f"{pid}-s1", title="Step 1")],
                )
                storage.save_plan(plan)
                p = storage.get_plan(pid)
                assert p is not None
                assert p.id == pid
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    all_plans = storage.list_plans(limit=100)
    assert len(all_plans) == 40


def test_task_storage_context_manager_and_memory():
    """TaskStorage functions properly as a context manager and with :memory: db."""
    with TaskStorage(":memory:") as store:
        plan = TaskPlan(id="mem-p1", objective="In memory test", steps=[TaskStep(id="s1", title="Step 1")])
        store.save_plan(plan)
        loaded = store.get_plan("mem-p1")
        assert loaded is not None
        assert loaded.objective == "In memory test"
        assert store.get_step("s1") is not None
        assert store.get_step("nonexistent") is None


def test_task_storage_schema_columns(storage: TaskStorage):
    """Tasks table has both summary and final_summary columns."""
    cur = storage._conn.cursor()
    cur.execute("PRAGMA table_info(tasks);")
    cols = {row["name"] for row in cur.fetchall()}
    assert "summary" in cols
    assert "final_summary" in cols
    assert "execution_log" in cols

    cur.execute("PRAGMA table_info(task_steps);")
    step_cols = {row["name"] for row in cur.fetchall()}
    for required in [
        "id", "plan_id", "step_index", "title", "description",
        "tool_name", "arguments_json", "dependencies_json", "status",
        "result", "error", "verification_json", "started_at", "completed_at"
    ]:
        assert required in step_cols


def test_string_status_handling(storage: TaskStorage):
    """String values for TaskStatus are accepted in updates and list filtering."""
    plan = TaskPlan(id="str-p1", objective="String status test", steps=[TaskStep(id="s1", title="Step 1")])
    storage.save_plan(plan)

    assert storage.update_plan_status("str-p1", "in_progress", summary="Work started")
    reloaded = storage.get_plan("str-p1")
    assert reloaded.status == TaskStatus.IN_PROGRESS
    assert reloaded.final_summary == "Work started"

    # List with string status filter
    in_prog = storage.list_plans(status="in_progress")
    assert any(p.id == "str-p1" for p in in_prog)


def test_recovery_waiting_approval(storage: TaskStorage):
    """Interrupted plans with WAITING_APPROVAL status are recovered."""
    plan = TaskPlan(
        id="p-wait",
        objective="Waiting approval plan",
        steps=[TaskStep(id="s-wait", title="Sensitive step", status=TaskStatus.PENDING)],
        status=TaskStatus.WAITING_APPROVAL,
    )
    storage.save_plan(plan)

    recovered = storage.recover_interrupted_plans()
    assert any(p.id == "p-wait" for p in recovered)


def test_application_restart_survival(tmp_path: Path):
    """Task plans survive complete application shutdown and restart."""
    db_file = tmp_path / "restart_test.db"

    # 1. First application session
    store1 = TaskStorage(db_path=db_file)
    v1 = StepVerification(
        type=VerificationType.FILE_EXISTS,
        target="data/out.txt",
        passed=True,
        message="File exists on disk",
    )
    s1 = TaskStep(
        id="step-restart-1",
        title="Init step",
        description="Initialize workspace",
        tool_name="write_file",
        arguments={"path": "data/out.txt", "content": "INITIALIZED"},
        dependencies=[],
        status=TaskStatus.COMPLETED,
        result="File created",
        verification=v1,
        started_at=100.0,
        completed_at=105.0,
    )
    s2 = TaskStep(
        id="step-restart-2",
        title="Process step",
        description="Run batch command",
        tool_name="powershell_exec",
        arguments={"command": "dir"},
        dependencies=["step-restart-1"],
        status=TaskStatus.IN_PROGRESS,
        started_at=106.0,
    )
    plan = TaskPlan(
        id="plan-restart-1",
        objective="Verify app restart survival",
        steps=[s1, s2],
        status=TaskStatus.IN_PROGRESS,
        approved=True,
        created_at=99.0,
        updated_at=106.0,
        execution_log=["Step 1 complete", "Step 2 started"],
        final_summary="Halfway done",
    )
    store1.save_plan(plan)
    # Simulate shutdown
    store1.close()

    # 2. Second application session (after restart)
    store2 = TaskStorage(db_path=db_file)
    try:
        reloaded = store2.get_plan("plan-restart-1")
        assert reloaded is not None
        assert reloaded.id == "plan-restart-1"
        assert reloaded.objective == "Verify app restart survival"
        assert reloaded.status == TaskStatus.IN_PROGRESS
        assert reloaded.approved is True
        assert reloaded.created_at == 99.0
        assert reloaded.updated_at == 106.0
        assert reloaded.execution_log == ["Step 1 complete", "Step 2 started"]
        assert reloaded.final_summary == "Halfway done"

        # Verify steps survived with full fidelity
        assert len(reloaded.steps) == 2
        r_s1 = reloaded.steps[0]
        assert r_s1.id == "step-restart-1"
        assert r_s1.title == "Init step"
        assert r_s1.description == "Initialize workspace"
        assert r_s1.tool_name == "write_file"
        assert r_s1.arguments == {"path": "data/out.txt", "content": "INITIALIZED"}
        assert r_s1.dependencies == []
        assert r_s1.status == TaskStatus.COMPLETED
        assert r_s1.result == "File created"
        assert r_s1.started_at == 100.0
        assert r_s1.completed_at == 105.0
        assert r_s1.verification is not None
        assert r_s1.verification.type == VerificationType.FILE_EXISTS
        assert r_s1.verification.target == "data/out.txt"
        assert r_s1.verification.passed is True

        r_s2 = reloaded.steps[1]
        assert r_s2.id == "step-restart-2"
        assert r_s2.dependencies == ["step-restart-1"]
        assert r_s2.status == TaskStatus.IN_PROGRESS
        assert r_s2.started_at == 106.0
    finally:
        store2.close()


def test_all_task_states_persist_accurately(storage: TaskStorage):
    """All TaskStatus states (COMPLETED, FAILED, IN_PROGRESS, WAITING_APPROVAL, CANCELLED) persist accurately."""
    states = [
        TaskStatus.PENDING,
        TaskStatus.WAITING_APPROVAL,
        TaskStatus.IN_PROGRESS,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    ]

    for st in states:
        p = TaskPlan(
            id=f"plan-state-{st.value}",
            objective=f"Objective for {st.value}",
            steps=[
                TaskStep(
                    id=f"step-{st.value}",
                    title=f"Step in {st.value}",
                    status=st,
                    result=f"Result {st.value}" if st == TaskStatus.COMPLETED else None,
                    error=f"Error {st.value}" if st == TaskStatus.FAILED else None,
                )
            ],
            status=st,
            approved=True,
            final_summary=f"Summary for {st.value}",
        )
        storage.save_plan(p)

    for st in states:
        loaded = storage.get_plan(f"plan-state-{st.value}")
        assert loaded is not None
        assert loaded.status == st
        assert loaded.final_summary == f"Summary for {st.value}"
        assert loaded.steps[0].status == st
        if st == TaskStatus.COMPLETED:
            assert loaded.steps[0].result == f"Result {st.value}"
        if st == TaskStatus.FAILED:
            assert loaded.steps[0].error == f"Error {st.value}"


def test_recovery_resets_in_progress_and_waiting_approval_safely(storage: TaskStorage):
    """Recovery resets IN_PROGRESS and WAITING_APPROVAL steps to PENDING without repeating COMPLETED steps."""
    s1 = TaskStep(
        id="s-done",
        title="Finished step",
        status=TaskStatus.COMPLETED,
        result="Already finished successfully",
        verification=StepVerification(type=VerificationType.FILE_EXISTS, target="test.txt", passed=True),
    )
    s2 = TaskStep(
        id="s-prog",
        title="Interrupted mid-flight",
        status=TaskStatus.IN_PROGRESS,
        dependencies=["s-done"],
    )
    s3 = TaskStep(
        id="s-wait",
        title="Interrupted awaiting user approval",
        status=TaskStatus.WAITING_APPROVAL,
        dependencies=["s-prog"],
    )
    s4 = TaskStep(
        id="s-pend",
        title="Downstream pending",
        status=TaskStatus.PENDING,
        dependencies=["s-wait"],
    )

    plan = TaskPlan(
        id="plan-complex-recovery",
        objective="Recovery reset safety check",
        steps=[s1, s2, s3, s4],
        status=TaskStatus.IN_PROGRESS,
        approved=True,
    )
    storage.save_plan(plan)

    recovered = storage.recover_interrupted_plans()
    assert len(recovered) == 1
    rec = recovered[0]
    assert rec.id == "plan-complex-recovery"

    # Step 1: strictly COMPLETED, result & verification preserved intact
    assert rec.steps[0].status == TaskStatus.COMPLETED
    assert rec.steps[0].result == "Already finished successfully"
    assert rec.steps[0].verification.passed is True

    # Step 2: reset to PENDING with diagnostic error note
    assert rec.steps[1].status == TaskStatus.PENDING
    assert "interrupted" in rec.steps[1].error.lower()

    # Step 3: reset to PENDING with diagnostic error note
    assert rec.steps[2].status == TaskStatus.PENDING
    assert "interrupted" in rec.steps[2].error.lower()

    # Step 4: remains PENDING
    assert rec.steps[3].status == TaskStatus.PENDING

    # Verify persisted in SQLite
    persisted = storage.get_plan("plan-complex-recovery")
    assert persisted.steps[0].status == TaskStatus.COMPLETED
    assert persisted.steps[1].status == TaskStatus.PENDING
    assert persisted.steps[2].status == TaskStatus.PENDING
    assert persisted.steps[3].status == TaskStatus.PENDING


def test_recovery_auto_completes_when_all_steps_already_done(storage: TaskStorage):
    """If all steps in an interrupted plan were already COMPLETED, recovery finalizes the plan as COMPLETED."""
    s1 = TaskStep(id="s1", title="Step 1", status=TaskStatus.COMPLETED, result="Done 1")
    s2 = TaskStep(id="s2", title="Step 2", status=TaskStatus.COMPLETED, result="Done 2")

    plan = TaskPlan(
        id="plan-all-done",
        objective="All steps completed before crash",
        steps=[s1, s2],
        status=TaskStatus.IN_PROGRESS,
        approved=True,
    )
    storage.save_plan(plan)

    # Calling recovery finalizes it as COMPLETED
    recovered = storage.recover_interrupted_plans()
    assert len(recovered) == 0

    reloaded = storage.get_plan("plan-all-done")
    assert reloaded.status == TaskStatus.COMPLETED
    assert reloaded.is_successful() is True


def test_case_insensitive_list_plans_filtering(storage: TaskStorage):
    """list_plans filtering is case-insensitive for both TaskStatus enums and raw strings."""
    plan = TaskPlan(id="case-p1", objective="Case sensitivity test", status=TaskStatus.COMPLETED)
    storage.save_plan(plan)

    assert any(p.id == "case-p1" for p in storage.list_plans(status="COMPLETED"))
    assert any(p.id == "case-p1" for p in storage.list_plans(status="completed"))
    assert any(p.id == "case-p1" for p in storage.list_plans(status=TaskStatus.COMPLETED))


