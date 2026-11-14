"""Declarative capability registry and JSON schema argument validation for tools."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from shamu.types import PermissionTier, PolicyOutcome


class ToolCapability(BaseModel):
    """Specification of an authorized capability and its execution constraints."""
    name: str = Field(description="Unique tool identifier")
    tier: PermissionTier = Field(description="Assigned permission tier")
    description: str = Field(description="Human-readable description of tool action")
    parameters_schema: Dict[str, Any] = Field(description="JSON schema describing accepted parameters")
    path_arg_names: List[str] = Field(default_factory=list, description="Argument names representing filesystem paths")
    command_arg_name: Optional[str] = Field(default=None, description="Argument name representing a shell command")
    requires_double_confirmation: bool = Field(default=False, description="Mandates explicit double confirmation")
    is_prohibited: bool = Field(default=False, description="Whether this action is permanently banned")

    def to_tool_definition(self):
        """Convert capability to OpenAI-compatible ToolDefinition."""
        from shamu.types import ToolDefinition
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )


class ToolRegistry:
    """Catalog of declared capabilities with strict argument schema validation."""

    def __init__(self) -> None:
        self._capabilities: Dict[str, ToolCapability] = {}
        self._register_builtins()

    @classmethod
    def create_default(cls) -> "ToolRegistry":
        """Factory method to construct a ToolRegistry populated with default capabilities."""
        return cls()

    def register(self, capability: ToolCapability) -> None:
        """Registers a capability in the security registry."""
        self._capabilities[capability.name] = capability

    def get(self, tool_name: str) -> Optional[ToolCapability]:
        """Retrieves capability descriptor by tool name."""
        return self._capabilities.get(tool_name)

    def list_all(self) -> List[ToolCapability]:
        """Returns all registered capabilities."""
        return list(self._capabilities.values())

    def get_tool_definitions(self):
        """Returns ToolDefinitions only for authorized, non-prohibited capabilities."""
        return [
            cap.to_tool_definition()
            for cap in self._capabilities.values()
            if not cap.is_prohibited and cap.tier != PermissionTier.PROHIBITED
        ]

    def _register_builtins(self) -> None:
        """Initializes default built-in capabilities."""
        # Tier 1: Read-Only
        self.register(ToolCapability(
            name="read_file",
            tier=PermissionTier.TIER_1_READ_ONLY,
            description="Reads file contents from within approved workspace",
            parameters_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            path_arg_names=["path"],
        ))

        self.register(ToolCapability(
            name="list_directory",
            tier=PermissionTier.TIER_1_READ_ONLY,
            description="Lists directory entries within approved workspace",
            parameters_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            path_arg_names=["path"],
        ))

        self.register(ToolCapability(
            name="get_system_metrics",
            tier=PermissionTier.TIER_1_READ_ONLY,
            description="Queries CPU, GPU, and RAM telemetry",
            parameters_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ))

        self.register(ToolCapability(
            name="search_local_history",
            tier=PermissionTier.TIER_1_READ_ONLY,
            description="Searches persistent local memory",
            parameters_schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
                "additionalProperties": False,
            },
        ))

        # Tier 2: Safe Workspace Mutations (Requires Interactive Approval)
        self.register(ToolCapability(
            name="write_file",
            tier=PermissionTier.TIER_2_MUTATION,
            description="Creates or overwrites a file within approved workspace",
            parameters_schema={
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "additionalProperties": False,
            },
            path_arg_names=["path"],
        ))

        self.register(ToolCapability(
            name="append_file",
            tier=PermissionTier.TIER_2_MUTATION,
            description="Appends text to a file within approved workspace",
            parameters_schema={
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "additionalProperties": False,
            },
            path_arg_names=["path"],
        ))

        self.register(ToolCapability(
            name="focus_application",
            tier=PermissionTier.TIER_2_MUTATION,
            description="Brings application window to foreground",
            parameters_schema={
                "type": "object",
                "required": ["window_title"],
                "properties": {"window_title": {"type": "string"}},
                "additionalProperties": False,
            },
        ))

        # Tier 3: High-Impact Operations (Requires Double Confirmation)
        self.register(ToolCapability(
            name="powershell_exec",
            tier=PermissionTier.TIER_3_HIGH_IMPACT,
            description="Executes PowerShell script in sandboxed Job Object",
            parameters_schema={
                "type": "object",
                "required": ["command"],
                "properties": {"command": {"type": "string"}},
                "additionalProperties": False,
            },
            command_arg_name="command",
            requires_double_confirmation=True,
        ))

        self.register(ToolCapability(
            name="delete_file",
            tier=PermissionTier.TIER_3_HIGH_IMPACT,
            description="Permanently deletes a file within approved workspace",
            parameters_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            path_arg_names=["path"],
            requires_double_confirmation=True,
        ))

        self.register(ToolCapability(
            name="kill_process",
            tier=PermissionTier.TIER_3_HIGH_IMPACT,
            description="Terminates a process by PID",
            parameters_schema={
                "type": "object",
                "required": ["pid"],
                "properties": {"pid": {"type": "integer"}},
                "additionalProperties": False,
            },
            requires_double_confirmation=True,
        ))

        # Prohibited Tools (Hard Deny)
        for prohibited_name in ["format_disk", "reg_delete", "disable_firewall", "elevate_uac"]:
            self.register(ToolCapability(
                name=prohibited_name,
                tier=PermissionTier.PROHIBITED,
                description="Catastrophic operation permanently prohibited by security policy",
                parameters_schema={"type": "object", "properties": {}, "additionalProperties": True},
                is_prohibited=True,
            ))

    def validate_call(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates tool invocation against registered capability and JSON parameter schema."""
        if not isinstance(arguments, dict):
            return False, f"Arguments must be a dictionary, got {type(arguments).__name__}."

        cap = self.get(tool_name)
        if cap is None:
            return False, f"Unregistered tool: '{tool_name}' is not recognized in security registry."

        if cap.is_prohibited or cap.tier == PermissionTier.PROHIBITED:
            return False, f"Tool '{tool_name}' is permanently prohibited by security policy."

        schema = cap.parameters_schema
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})
        additional_properties = schema.get("additionalProperties", True)

        # 1. Check required parameters
        for req in required_fields:
            if req not in arguments:
                return False, f"Missing required parameter '{req}' for tool '{tool_name}'."

        # 2. Check undeclared additional parameters
        if not additional_properties:
            for arg_key in arguments:
                if arg_key not in properties:
                    return False, f"Unexpected additional argument '{arg_key}' for tool '{tool_name}'."

        # 3. Check types
        for prop_name, prop_spec in properties.items():
            if prop_name not in arguments:
                continue
            val = arguments[prop_name]
            expected_type = prop_spec.get("type")

            if expected_type == "string":
                if not isinstance(val, str):
                    return False, f"Parameter '{prop_name}' must be of type string, got {type(val).__name__}."
            elif expected_type == "integer":
                if not isinstance(val, int) or isinstance(val, bool):
                    return False, f"Parameter '{prop_name}' must be of type integer, got {type(val).__name__}."
            elif expected_type == "boolean":
                if not isinstance(val, bool):
                    return False, f"Parameter '{prop_name}' must be of type boolean, got {type(val).__name__}."
            elif expected_type == "number":
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    return False, f"Parameter '{prop_name}' must be of type number, got {type(val).__name__}."
            elif expected_type == "array":
                if not isinstance(val, list):
                    return False, f"Parameter '{prop_name}' must be of type array, got {type(val).__name__}."
            elif expected_type == "object":
                if not isinstance(val, dict):
                    return False, f"Parameter '{prop_name}' must be of type object, got {type(val).__name__}."

        return True, None
