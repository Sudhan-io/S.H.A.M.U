"""Task planning, DAG dependency resolution, and plan validation for S.H.A.M.U."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from shamu.gateway.router import ModelGateway
from shamu.security.firewall import ExecutionFirewall
from shamu.tasks.models import StepVerification, TaskPlan, TaskStatus, TaskStep, VerificationType
from shamu.types import ChatMessage, Role

logger = logging.getLogger(__name__)


class PlanValidationError(Exception):
    """Raised when a task plan fails structural or dependency validation."""
    pass


def validate_plan_dag(plan: TaskPlan) -> Tuple[bool, Optional[str]]:
    """Validates that a TaskPlan forms a valid Directed Acyclic Graph (DAG) with no cycles.

    Returns:
        (True, None) if the plan is a valid DAG.
        (False, error_message) if invalid, containing cycle or missing dependency details.
    """
    if plan is None or not getattr(plan, "steps", None):
        return False, "Plan contains no steps."

    step_ids: Set[str] = set()
    for i, step in enumerate(plan.steps):
        if not step.id or not str(step.id).strip():
            return False, f"Step at index {i} has missing or empty id."
        if step.id in step_ids:
            return False, f"Duplicate step ID detected: '{step.id}'"
        step_ids.add(step.id)

    # Validate that all dependencies reference valid steps and not self
    for step in plan.steps:
        for dep_id in step.dependencies:
            if not isinstance(dep_id, str) or not dep_id.strip():
                return False, f"Step '{step.id}' has invalid dependency: '{dep_id}'"
            dep_clean = dep_id.strip()
            if dep_clean == step.id:
                return False, f"Self-dependency detected in step '{step.id}'"
            if dep_clean not in step_ids:
                return False, f"Step '{step.id}' depends on non-existent step '{dep_clean}'"

    # Cycle detection using 3-color DFS
    # 0 = unvisited, 1 = visiting (in current recursion stack), 2 = visited (completed)
    state: Dict[str, int] = {s.id: 0 for s in plan.steps}
    adj: Dict[str, List[str]] = {
        s.id: list(dict.fromkeys(d.strip() for d in s.dependencies if isinstance(d, str) and d.strip()))
        for s in plan.steps
    }

    def has_cycle(u: str, path: List[str]) -> Tuple[bool, Optional[List[str]]]:
        state[u] = 1  # visiting
        path.append(u)
        for v in adj.get(u, []):
            if state[v] == 1:
                # Cycle found! Extract the exact cycle subpath starting and ending at v
                cycle_start = path.index(v) if v in path else 0
                return True, path[cycle_start:] + [v]
            if state[v] == 0:
                cycle_found, cpath = has_cycle(v, path)
                if cycle_found:
                    return True, cpath
        state[u] = 2  # visited
        path.pop()
        return False, None

    for step in plan.steps:
        if state[step.id] == 0:
            cycle_found, cycle_path = has_cycle(step.id, [])
            if cycle_found and cycle_path:
                cycle_str = " -> ".join(cycle_path)
                return False, f"Circular dependency detected in plan: {cycle_str}"

    return True, None


def get_execution_order(plan: TaskPlan) -> List[TaskStep]:
    """Computes a valid topological execution order for the task steps using Kahn's algorithm.

    Raises PlanValidationError if the plan contains cycles, self-dependencies,
    or missing dependencies.
    """
    is_valid, err = validate_plan_dag(plan)
    if not is_valid:
        raise PlanValidationError(f"Cannot compute execution order: {err}")

    # Build adjacency and in-degree counts (dep -> dependent)
    step_map: Dict[str, TaskStep] = {s.id: s for s in plan.steps}
    in_degree: Dict[str, int] = {
        s.id: len(set(d.strip() for d in s.dependencies if isinstance(d, str) and d.strip()))
        for s in plan.steps
    }
    dependents: Dict[str, List[str]] = {s.id: [] for s in plan.steps}

    for s in plan.steps:
        clean_deps = set(d.strip() for d in s.dependencies if isinstance(d, str) and d.strip())
        for dep in clean_deps:
            dependents[dep].append(s.id)

    # Queue of nodes with 0 prerequisites (preserving definition order)
    queue: List[str] = [s.id for s in plan.steps if in_degree[s.id] == 0]
    order: List[TaskStep] = []

    while queue:
        curr_id = queue.pop(0)
        order.append(step_map[curr_id])
        for dep_on_curr in dependents.get(curr_id, []):
            in_degree[dep_on_curr] -= 1
            if in_degree[dep_on_curr] == 0:
                queue.append(dep_on_curr)

    if len(order) != len(plan.steps):
        raise PlanValidationError("Topological ordering resolution failed: cycle detected.")

    return order


def _clean_trailing_commas(text: str) -> str:
    """Removes trailing commas before closing braces or brackets."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _find_balanced_json(text: str) -> Optional[Dict[str, Any]]:
    """Scans for balanced curly brace substrings and parses the first valid JSON object."""
    start_idx = None
    depth = 0
    in_string = False
    escape = False

    fallback_dict: Optional[Dict[str, Any]] = None

    for i, ch in enumerate(text):
        if ch == '"' and not escape:
            in_string = not in_string
        elif ch == '\\' and in_string:
            escape = not escape
            continue
        elif not in_string:
            if ch == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        candidate_str = text[start_idx : i + 1]
                        for c_str in (candidate_str, _clean_trailing_commas(candidate_str)):
                            try:
                                obj = json.loads(c_str)
                                if isinstance(obj, dict):
                                    if "steps" in obj or "objective" in obj:
                                        return obj
                                    if fallback_dict is None:
                                        fallback_dict = obj
                            except Exception:
                                pass
        escape = False

    return fallback_dict


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Extracts a valid JSON object from Markdown code fences, raw strings, or mixed text."""
    s = text.strip()
    if not s:
        raise PlanValidationError("Could not parse valid JSON from empty response.")

    # 1. Direct JSON parse
    for raw in (s, _clean_trailing_commas(s)):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 2. Markdown code fences: ```(?:json)? ... ```
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    for match in fence_pattern.finditer(s):
        block = match.group(1).strip()
        for raw in (block, _clean_trailing_commas(block)):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        # Scan for balanced JSON inside the code fence block
        candidate = _find_balanced_json(block)
        if candidate is not None:
            return candidate

    # 3. Balanced braces scan across entire text
    candidate = _find_balanced_json(s)
    if candidate is not None:
        return candidate

    # 4. Fallback: regex search for outer braces
    outer_match = re.search(r"(\{.*\})", s, re.DOTALL)
    if outer_match:
        for raw in (outer_match.group(1), _clean_trailing_commas(outer_match.group(1))):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    raise PlanValidationError(f"Could not parse valid JSON from planner response:\n{text[:200]}")


def parse_plan_dict(data: Dict[str, Any], default_objective: str = "") -> TaskPlan:
    """Parses a dictionary into a validated TaskPlan."""
    raw_steps = data.get("steps")
    if not raw_steps or not isinstance(raw_steps, list):
        raise PlanValidationError("Planner response contains empty steps list.")

    steps: List[TaskStep] = []
    for i, s_data in enumerate(raw_steps):
        if not isinstance(s_data, dict):
            raise PlanValidationError(f"Step at index {i} is invalid: expected dictionary, got {type(s_data).__name__}")

        step_id = str(s_data.get("id") or f"step-{i+1}").strip()
        title = str(s_data.get("title") or f"Step {i+1}").strip()
        description = str(s_data.get("description") or "").strip()

        tool_name = s_data.get("tool_name")
        if tool_name is not None:
            tool_name = str(tool_name).strip()
            if not tool_name or tool_name.lower() in ("null", "none"):
                tool_name = None

        arguments = s_data.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}

        dependencies = s_data.get("dependencies")
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        elif not isinstance(dependencies, list):
            dependencies = []
        cleaned_deps = [str(d).strip() for d in dependencies if d is not None and str(d).strip()]

        ver_data = s_data.get("verification")
        verification = None
        if ver_data and isinstance(ver_data, dict):
            v_type_str = str(ver_data.get("type", "file_exists")).strip().lower()
            try:
                verification_type = VerificationType(v_type_str)
            except ValueError:
                verification_type = VerificationType.CUSTOM
            target = str(ver_data.get("target", "") or "").strip()
            expected = ver_data.get("expected")
            if expected is not None:
                expected = str(expected)
            verification = StepVerification(
                type=verification_type,
                target=target,
                expected=expected,
            )

        step = TaskStep(
            id=step_id,
            title=title,
            description=description,
            tool_name=tool_name,
            arguments=arguments,
            dependencies=cleaned_deps,
            verification=verification,
        )
        steps.append(step)

    objective_text = str(data.get("objective") or default_objective or "").strip()
    if not objective_text:
        objective_text = "Task execution plan"

    plan = TaskPlan(
        objective=objective_text,
        steps=steps,
    )

    is_valid, error_msg = validate_plan_dag(plan)
    if not is_valid:
        raise PlanValidationError(f"Generated plan is invalid: {error_msg}")

    return plan


def parse_plan(data_or_text: Union[str, Dict[str, Any]], default_objective: str = "") -> TaskPlan:
    """Parses a JSON string, Markdown response, or dictionary into a validated TaskPlan."""
    if isinstance(data_or_text, str):
        data = extract_json_from_response(data_or_text)
    elif isinstance(data_or_text, dict):
        data = data_or_text
    else:
        raise PlanValidationError(f"Expected str or dict for plan parsing, got {type(data_or_text).__name__}")
    return parse_plan_dict(data, default_objective=default_objective)


class TaskPlanner:
    """Generates and validates structured, multi-step task execution plans."""

    PLANNING_SYSTEM_PROMPT = (
        "You are S.H.A.M.U. Task Planner, an expert system for decomposing complex user objectives "
        "into discrete, deterministic, verifiable task execution steps forming a Directed Acyclic Graph (DAG).\n\n"
        "Guidelines:\n"
        "1. Break the user objective into discrete sequential or parallel steps.\n"
        "2. For each step that interacts with the operating system or tools, specify the exact tool name and arguments.\n"
        "3. Explicitly list dependencies by ID if a step depends on the output or side-effects of another step.\n"
        "4. Include programmatic verification criteria whenever possible (e.g. file_exists, file_contains, file_not_exists, command_success).\n"
        "5. Respond STRICTLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "objective": "<original objective>",\n'
        '  "steps": [\n'
        "    {\n"
        '      "id": "step-1",\n'
        '      "title": "<short title>",\n'
        '      "description": "<detailed instruction>",\n'
        '      "tool_name": "<tool_name or null>",\n'
        '      "arguments": { "<arg_name>": "<arg_value>" },\n'
        '      "dependencies": [],\n'
        '      "verification": {\n'
        '        "type": "file_exists | file_contains | file_not_exists | command_success",\n'
        '        "target": "<filepath or command>",\n'
        '        "expected": "<expected content or null>"\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    def __init__(
        self,
        gateway: ModelGateway,
        firewall: Optional[ExecutionFirewall] = None,
    ) -> None:
        self.gateway = gateway
        self.firewall = firewall

    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """Extracts JSON object from Markdown code fence or raw string."""
        return extract_json_from_response(text)

    def parse_plan(
        self,
        data_or_text: Union[str, Dict[str, Any]],
        default_objective: str = "",
    ) -> TaskPlan:
        """Parses and validates a plan from response text or a dictionary."""
        return parse_plan(data_or_text, default_objective=default_objective)

    async def create_plan(
        self,
        objective: str,
        tools: Optional[Sequence[Any]] = None,
        context: Optional[str] = None,
    ) -> TaskPlan:
        """Prompts the LLM gateway to decompose an objective into a verified TaskPlan."""
        available_tools = tools
        if available_tools is None and self.firewall is not None:
            available_tools = self.firewall.get_tool_definitions()

        tools_desc = ""
        if available_tools:
            formatted_tools: List[str] = []
            for t in available_tools:
                if isinstance(t, dict):
                    t_name = t.get("name", "")
                    t_desc = t.get("description", "")
                else:
                    t_name = getattr(t, "name", "")
                    t_desc = getattr(t, "description", "")
                if t_name:
                    formatted_tools.append(f"- {t_name}: {t_desc}" if t_desc else f"- {t_name}")
            if formatted_tools:
                tools_desc = "\n\nAvailable tools:\n" + "\n".join(formatted_tools)

        user_content = f"Objective: {objective}"
        if context:
            user_content += f"\n\nContext / Constraints:\n{context}"

        messages = [
            ChatMessage(role=Role.SYSTEM, content=self.PLANNING_SYSTEM_PROMPT + tools_desc),
            ChatMessage(role=Role.USER, content=user_content),
        ]

        full_response: List[str] = []
        async for chunk in self.gateway.stream_chat(messages, tools=None):
            if chunk.text_delta:
                full_response.append(chunk.text_delta)

        response_text = "".join(full_response)
        return self.parse_plan(response_text, default_objective=objective)
