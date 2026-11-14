"""Core Pydantic data models for SHAMU."""

import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PermissionTier(str, Enum):
    """Authoritative permission tier classification for tools and security policies."""
    TIER_1_READ_ONLY = "TIER_1_READ_ONLY"      # Read-only / autonomous within bounds
    TIER_2_MUTATION = "TIER_2_MUTATION"        # Ordinary reversible mutation -> Interactive approval required
    TIER_3_HIGH_IMPACT = "TIER_3_HIGH_IMPACT"  # Destructive / system / shell -> Double confirmation required
    PROHIBITED = "PROHIBITED"                  # Permanently prohibited -> Hard deny

    # Backward-compatible legacy aliases
    TIER_1_AUTO = "TIER_1_AUTO"                # Equivalent to TIER_1_READ_ONLY
    TIER_2_AUDITED = "TIER_2_AUDITED"          # Equivalent to TIER_2_MUTATION
    TIER_3_CONFIRMATION = "TIER_3_CONFIRMATION"# Equivalent to TIER_3_HIGH_IMPACT

    def canonical(self) -> "PermissionTier":
        """Return canonical permission tier (mapping legacy aliases to authoritative tiers)."""
        mapping = {
            PermissionTier.TIER_1_AUTO: PermissionTier.TIER_1_READ_ONLY,
            PermissionTier.TIER_2_AUDITED: PermissionTier.TIER_2_MUTATION,
            PermissionTier.TIER_3_CONFIRMATION: PermissionTier.TIER_3_HIGH_IMPACT,
        }
        return mapping.get(self, self)

    def __eq__(self, other: object) -> bool:
        res = super().__eq__(other)
        if res is True:
            return True
        val1 = self.value if isinstance(self, Enum) else str(self)
        val2 = other.value if isinstance(other, Enum) else str(other) if isinstance(other, str) else None
        if val2 is None:
            return NotImplemented
        aliases = {
            "TIER_1_AUTO": "TIER_1_READ_ONLY",
            "TIER_2_AUDITED": "TIER_2_MUTATION",
            "TIER_3_CONFIRMATION": "TIER_3_HIGH_IMPACT",
        }
        return aliases.get(val1, val1) == aliases.get(val2, val2)

    def __hash__(self) -> int:
        aliases = {
            "TIER_1_AUTO": "TIER_1_READ_ONLY",
            "TIER_2_AUDITED": "TIER_2_MUTATION",
            "TIER_3_CONFIRMATION": "TIER_3_HIGH_IMPACT",
        }
        return hash(aliases.get(self.value, self.value))


class PolicyOutcome(str, Enum):
    """Authorization policy evaluation outcomes."""
    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"


class ToolCall(BaseModel):
    """Represents a function/tool call requested by a model."""
    id: str = Field(default="", description="Unique identifier for the tool call")
    type: str = Field(default="function", description="Type of tool invocation")
    name: str = Field(description="Name of the function to call")
    arguments: str = Field(default="{}", description="JSON string of arguments")

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


class ChatMessage(BaseModel):
    """Structured message in a conversation turn."""
    role: Role
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts message to standard OpenAI-compatible dictionary."""
        data: Dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            data["content"] = self.content
        if self.name is not None:
            data["name"] = self.name
        if self.tool_calls is not None:
            data["tool_calls"] = [tc.to_openai_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data


class ToolDefinition(BaseModel):
    """Definition and JSON Schema for an executable tool."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class SpeechConfigModel(BaseModel):
    """Schema model for speech configuration settings."""
    stt_model_size: str = "large-v3-turbo"
    stt_cpu_threads: int = 4
    stt_p_cores: List[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    vad_trailing_silence_ms: int = 400
    pre_speech_buffer_ms: int = 200
    custom_lexicon_path: str = "data/lexicon.json"


class ChatChunk(BaseModel):
    """Incremental chunk streamed from an LLM."""
    text_delta: str = ""
    tool_call_delta: Optional[ToolCall] = None
    finish_reason: Optional[str] = None
    reset_turn: bool = False

    @property
    def is_tool_call(self) -> bool:
        return self.tool_call_delta is not None

    @property
    def is_done(self) -> bool:
        return self.finish_reason is not None


class ChatResponse(BaseModel):
    """Final aggregated response for a single conversational turn."""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class MemoryRecord(BaseModel):
    """Durable memory record for facts, preferences, and interactions."""
    id: str = Field(description="Unique record identifier")
    content: str = Field(description="Text content of the memory or interaction")
    source: str = Field(default="conversation", description="Origin of record (e.g. conversation, user_explicit, document)")
    category: str = Field(default="general", description="Categorization (e.g. preference, project, identity, fact)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary JSON metadata")
    created_at: float = Field(default_factory=time.time, description="Unix timestamp of record creation")
    score: float = Field(default=0.0, description="Relevance score assigned during retrieval")


