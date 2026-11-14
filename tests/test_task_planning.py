"""Unit and integration tests for Task Planning, DAG validation, and topological sorting."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from shamu.tasks.models import (
    StepVerification,
    TaskPlan,
    TaskStatus,
    TaskStep,
    VerificationType,
)
from shamu.tasks.planner import (
    PlanValidationError,
    TaskPlanner,
    get_execution_order,
    validate_plan_dag,
)
from shamu.types import ChatChunk


def test_validate_plan_dag_empty():
    """Empty plan should fail validation."""
    plan = TaskPlan(objective="Test empty plan", steps=[])
    is_valid, err = validate_plan_dag(plan)
    assert not is_valid
    assert "no steps" in err.lower()


def test_validate_plan_dag_missing_or_duplicate_id():
    """Duplicate step IDs or empty IDs should fail validation."""
    # Duplicate ID
    plan = TaskPlan(
        objective="Test duplicate ID",
        steps=[
            TaskStep(id="step-1", title="First"),
            TaskStep(id="step-1", title="Duplicate"),
        ],
    )
    is_valid, err = validate_plan_dag(plan)
    assert not is_valid
    assert "duplicate" in err.lower()

    # Empty ID
    plan2 = TaskPlan(
        objective="Test empty ID",
        steps=[
            TaskStep(id="", title="Empty ID"),
        ],
    )
    is_valid2, err2 = validate_plan_dag(plan2)
    assert not is_valid2
    assert "missing or empty" in err2.lower()


def test_validate_plan_dag_self_dependency():
    """Self-referencing step dependency should fail validation."""
    step = TaskStep(id="step-1", title="Self loop", dependencies=["step-1"])
    plan = TaskPlan(objective="Self loop", steps=[step])
    is_valid, err = validate_plan_dag(plan)
    assert not is_valid
    assert "self-dependency" in err.lower()


def test_validate_plan_dag_nonexistent_dependency():
    """Dependency pointing to a missing step ID should fail validation."""
    step = TaskStep(id="step-1", title="Missing dep", dependencies=["step-ghost"])
    plan = TaskPlan(objective="Missing dep", steps=[step])
    is_valid, err = validate_plan_dag(plan)
    assert not is_valid
    assert "non-existent" in err.lower()


def test_validate_plan_dag_cycles():
    """Circular dependencies must be caught properly."""
    # 2-node cycle: A -> B -> A
    s1 = TaskStep(id="A", title="A", dependencies=["B"])
    s2 = TaskStep(id="B", title="B", dependencies=["A"])
    plan_2node = TaskPlan(objective="2-node cycle", steps=[s1, s2])
    is_valid, err = validate_plan_dag(plan_2node)
    assert not is_valid
    assert "circular dependency" in err.lower()

    # 3-node cycle: A -> B -> C -> A
    # Here A depends on C, C depends on B, B depends on A
    step_a = TaskStep(id="A", title="A", dependencies=["C"])
    step_b = TaskStep(id="B", title="B", dependencies=["A"])
    step_c = TaskStep(id="C", title="C", dependencies=["B"])
    plan_3node = TaskPlan(objective="3-node cycle", steps=[step_a, step_b, step_c])
    is_valid3, err3 = validate_plan_dag(plan_3node)
    assert not is_valid3
    assert "circular dependency" in err3.lower()


def test_validate_plan_dag_valid_diamond():
    """Valid DAG with diamond dependency structure."""
    # A (root), B depends on A, C depends on A, D depends on B and C
    step_a = TaskStep(id="A", title="A", dependencies=[])
    step_b = TaskStep(id="B", title="B", dependencies=["A"])
    step_c = TaskStep(id="C", title="C", dependencies=["A"])
    step_d = TaskStep(id="D", title="D", dependencies=["B", "C"])

    plan = TaskPlan(objective="Diamond plan", steps=[step_a, step_b, step_c, step_d])
    is_valid, err = validate_plan_dag(plan)
    assert is_valid
    assert err is None


def test_get_execution_order_linear_and_diamond():
    """Topological sorting should return steps in valid dependency execution order."""
    # Linear: 1 -> 2 -> 3
    s1 = TaskStep(id="step-1", title="1")
    s2 = TaskStep(id="step-2", title="2", dependencies=["step-1"])
    s3 = TaskStep(id="step-3", title="3", dependencies=["step-2"])
    linear_plan = TaskPlan(objective="Linear", steps=[s3, s1, s2])  # passed in random order

    order = get_execution_order(linear_plan)
    order_ids = [s.id for s in order]
    assert order_ids == ["step-1", "step-2", "step-3"]

    # Diamond: A -> B, A -> C, (B, C) -> D
    step_a = TaskStep(id="A", title="A")
    step_b = TaskStep(id="B", title="B", dependencies=["A"])
    step_c = TaskStep(id="C", title="C", dependencies=["A"])
    step_d = TaskStep(id="D", title="D", dependencies=["B", "C"])
    diamond_plan = TaskPlan(objective="Diamond", steps=[step_d, step_c, step_b, step_a])

    d_order = get_execution_order(diamond_plan)
    d_order_ids = [s.id for s in d_order]
    assert d_order_ids.index("A") < d_order_ids.index("B")
    assert d_order_ids.index("A") < d_order_ids.index("C")
    assert d_order_ids.index("B") < d_order_ids.index("D")
    assert d_order_ids.index("C") < d_order_ids.index("D")


def test_get_execution_order_invalid_dag_raises():
    """Calling get_execution_order on an invalid plan raises PlanValidationError."""
    s1 = TaskStep(id="A", title="A", dependencies=["B"])
    s2 = TaskStep(id="B", title="B", dependencies=["A"])
    plan = TaskPlan(objective="Cycle", steps=[s1, s2])

    with pytest.raises(PlanValidationError) as exc_info:
        get_execution_order(plan)
    assert "Circular dependency" in str(exc_info.value)


@pytest.mark.asyncio
async def test_task_planner_json_extraction_code_fence():
    """TaskPlanner correctly parses JSON formatted within markdown code blocks."""
    mock_gateway = MagicMock()
    mock_firewall = MagicMock()
    mock_firewall.get_tool_definitions.return_value = [
        {"name": "read_file", "description": "Reads a file"}
    ]

    llm_output = """Here is the structured execution plan:
```json
{
  "objective": "Inspect system logs",
  "steps": [
    {
      "id": "s1",
      "title": "Read log",
      "description": "Read the main log file",
      "tool_name": "read_file",
      "arguments": {"path": "logs/app.log"},
      "dependencies": [],
      "verification": {
        "type": "file_exists",
        "target": "logs/app.log"
      }
    }
  ]
}
```
Hope this helps!"""

    async def mock_stream(*args, **kwargs):
        yield ChatChunk(text_delta=llm_output)

    mock_gateway.stream_chat = mock_stream

    planner = TaskPlanner(gateway=mock_gateway, firewall=mock_firewall)
    plan = await planner.create_plan("Inspect system logs")

    assert plan.objective == "Inspect system logs"
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.id == "s1"
    assert step.title == "Read log"
    assert step.tool_name == "read_file"
    assert step.arguments == {"path": "logs/app.log"}
    assert step.verification is not None
    assert step.verification.type == VerificationType.FILE_EXISTS
    assert step.verification.target == "logs/app.log"


@pytest.mark.asyncio
async def test_task_planner_json_extraction_raw():
    """TaskPlanner extracts raw outer JSON without markdown fences."""
    mock_gateway = MagicMock()

    llm_output = """{
      "objective": "Build project",
      "steps": [
        {
          "id": "step-1",
          "title": "Compile",
          "description": "Run compiler",
          "tool_name": "run_command",
          "arguments": {"command": "make build"},
          "dependencies": []
        },
        {
          "id": "step-2",
          "title": "Verify binary",
          "description": "Check binary exists",
          "tool_name": null,
          "arguments": {},
          "dependencies": ["step-1"],
          "verification": {
            "type": "file_exists",
            "target": "bin/app.exe"
          }
        }
      ]
    }"""

    async def mock_stream(*args, **kwargs):
        yield ChatChunk(text_delta=llm_output)

    mock_gateway.stream_chat = mock_stream
    planner = TaskPlanner(gateway=mock_gateway)
    plan = await planner.create_plan("Build project")

    assert len(plan.steps) == 2
    assert plan.steps[0].id == "step-1"
    assert plan.steps[1].dependencies == ["step-1"]


@pytest.mark.asyncio
async def test_task_planner_malformed_json_raises():
    """TaskPlanner raises PlanValidationError if response has invalid JSON."""
    mock_gateway = MagicMock()

    async def mock_stream(*args, **kwargs):
        yield ChatChunk(text_delta="This is not JSON at all.")

    mock_gateway.stream_chat = mock_stream
    planner = TaskPlanner(gateway=mock_gateway)

    with pytest.raises(PlanValidationError) as exc:
        await planner.create_plan("Test malformed")
    assert "Could not parse valid JSON" in str(exc.value)


@pytest.mark.asyncio
async def test_task_planner_empty_steps_raises():
    """TaskPlanner raises PlanValidationError if steps list is empty."""
    mock_gateway = MagicMock()

    async def mock_stream(*args, **kwargs):
        yield ChatChunk(text_delta='{"objective": "Empty", "steps": []}')

    mock_gateway.stream_chat = mock_stream
    planner = TaskPlanner(gateway=mock_gateway)

    with pytest.raises(PlanValidationError) as exc:
        await planner.create_plan("Test empty steps")
    assert "empty steps list" in str(exc.value)
