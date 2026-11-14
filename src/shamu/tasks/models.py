"""Shared domain models and contracts for S.H.A.M.U. Milestone 3: Task Planning & Execution."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Lifecycle state of an individual step or full task plan."""
    PENDING = "pending"
    WAITING_APPROVAL = "waiting_approval"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class VerificationType(str, Enum):
    """Types of programmatic post-condition state verification."""
    FILE_EXISTS = "file_exists"
    FILE_CONTAINS = "file_contains"
    FILE_NOT_EXISTS = "file_not_exists"
    COMMAND_SUCCESS = "command_success"
    CUSTOM = "custom"


class StepVerification(BaseModel):
    """Specification for independent, deterministic post-condition state verification."""
    type: VerificationType
    target: str = Field(description="Target path or subject of assertion")
    expected: Optional[str] = Field(default=None, description="Expected content substring or value")
    passed: Optional[bool] = Field(default=None, description="Evaluation outcome")
    message: Optional[str] = Field(default=None, description="Detailed diagnostic or error message")


class TaskStep(BaseModel):
    """Atomic step in a multi-step task plan forming a DAG node."""
    id: str = Field(default_factory=lambda: f"step-{uuid.uuid4().hex[:8]}")
    title: str = Field(description="Short human-readable summary of step")
    description: str = Field(default="", description="Detailed instruction of what this step accomplishes")
    tool_name: Optional[str] = Field(default=None, description="Name of the tool to invoke, if any")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    dependencies: List[str] = Field(default_factory=list, description="Step IDs that must complete before this step can run")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    result: Optional[str] = Field(default=None, description="Output from tool execution or step completion")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    verification: Optional[StepVerification] = Field(default=None, description="Post-condition verification check")
    started_at: Optional[float] = Field(default=None)
    completed_at: Optional[float] = Field(default=None)


class TaskPlan(BaseModel):
    """Structured plan composed of interdependent task steps forming a DAG."""
    id: str = Field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}")
    objective: str = Field(description="User objective that originated this plan")
    steps: List[TaskStep] = Field(default_factory=list, description="Ordered or interdependent steps")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    approved: bool = Field(default=False, description="Whether user explicitly approved execution of this plan")
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    execution_log: List[str] = Field(default_factory=list)
    final_summary: Optional[str] = Field(default=None)

    def get_step(self, step_id: str) -> Optional[TaskStep]:
        """Finds a step by its unique identifier."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def is_complete(self) -> bool:
        """Returns True if all steps have reached terminal status."""
        return all(s.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.CANCELLED) for s in self.steps)

    def is_successful(self) -> bool:
        """Returns True if all steps completed successfully."""
        return len(self.steps) > 0 and all(s.status == TaskStatus.COMPLETED for s in self.steps)


class TaskEvent(BaseModel):
    """Base event emitted during plan execution."""
    plan_id: str
    timestamp: float = Field(default_factory=time.time)


class PlanStartedEvent(TaskEvent):
    objective: str
    total_steps: int


class StepStartedEvent(TaskEvent):
    step_id: str
    step_title: str
    tool_name: Optional[str] = None


class StepWaitingApprovalEvent(TaskEvent):
    step_id: str
    step_title: str
    tool_name: str
    arguments: Dict[str, Any]


class StepCompletedEvent(TaskEvent):
    step_id: str
    step_title: str
    result: str
    verified: bool


class StepFailedEvent(TaskEvent):
    step_id: str
    step_title: str
    error: str


class PlanCompletedEvent(TaskEvent):
    success: bool
    completed_steps: int
    total_steps: int
    summary: str


class TaskExecutionReport(BaseModel):
    """Final summary report of a completed or failed task plan execution."""
    plan_id: str
    objective: str
    status: TaskStatus
    total_steps: int
    completed_steps: int
    failed_steps: int
    duration_seconds: float
    step_summaries: List[Dict[str, Any]]
    final_output: str
