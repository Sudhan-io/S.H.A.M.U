import pytest
from shamu.types import (
    Role,
    ChatMessage,
    ToolDefinition,
    ToolCall,
    ChatChunk,
    ChatResponse,
    PermissionTier,
)
from shamu.protocols import (
    ShamuError,
    ProviderUnavailableError,
    ProviderAuthError,
    ContextWindowExceededError,
)

def test_chat_message_creation():
    msg = ChatMessage(role=Role.USER, content="Hello SHAMU")
    assert msg.role == Role.USER
    assert msg.content == "Hello SHAMU"
    assert msg.tool_calls is None

    dumped = msg.to_dict()
    assert dumped == {"role": "user", "content": "Hello SHAMU"}

def test_chat_message_with_tool_calls():
    tool_call = ToolCall(
        id="call_123",
        type="function",
        name="get_weather",
        arguments='{"location": "Seattle"}',
    )
    msg = ChatMessage(role=Role.ASSISTANT, content=None, tool_calls=[tool_call])
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "get_weather"

    dumped = msg.to_dict()
    assert dumped["role"] == "assistant"
    assert len(dumped["tool_calls"]) == 1
    assert dumped["tool_calls"][0]["function"]["name"] == "get_weather"

def test_tool_definition_to_openai_schema():
    tool = ToolDefinition(
        name="read_file",
        description="Reads a file from the workspace",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    schema = tool.to_openai_dict()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "read_file"
    assert "parameters" in schema["function"]

def test_chat_chunk_properties():
    chunk = ChatChunk(text_delta="Hello", finish_reason=None)
    assert chunk.text_delta == "Hello"
    assert not chunk.is_tool_call
    assert not chunk.is_done

    finish_chunk = ChatChunk(text_delta="", finish_reason="stop")
    assert finish_chunk.is_done

def test_exception_hierarchy():
    err = ProviderUnavailableError("Ollama is offline", provider="ollama")
    assert isinstance(err, ShamuError)
    assert err.provider == "ollama"
    assert "Ollama is offline" in str(err)


def test_permission_tier_equality_and_aliases():
    """Verify PermissionTier equality, legacy aliases, unrelated types, and no deprecation warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")

        t = PermissionTier.TIER_1_READ_ONLY

        # 1. Identical enum comparison
        assert t == t
        assert t == PermissionTier.TIER_1_READ_ONLY
        assert PermissionTier.TIER_2_MUTATION == PermissionTier.TIER_2_MUTATION
        assert PermissionTier.TIER_3_HIGH_IMPACT == PermissionTier.TIER_3_HIGH_IMPACT
        assert PermissionTier.PROHIBITED == PermissionTier.PROHIBITED

        # 2. Alias string comparison
        assert t == "TIER_1_AUTO"
        assert t == "TIER_1_READ_ONLY"
        assert PermissionTier.TIER_1_AUTO == "TIER_1_READ_ONLY"
        assert PermissionTier.TIER_2_MUTATION == "TIER_2_AUDITED"
        assert PermissionTier.TIER_2_AUDITED == "TIER_2_MUTATION"
        assert PermissionTier.TIER_3_HIGH_IMPACT == "TIER_3_CONFIRMATION"
        assert PermissionTier.TIER_3_CONFIRMATION == "TIER_3_HIGH_IMPACT"

        # 3. Alias enum comparison
        assert PermissionTier.TIER_1_READ_ONLY == PermissionTier.TIER_1_AUTO
        assert PermissionTier.TIER_1_AUTO == PermissionTier.TIER_1_READ_ONLY
        assert PermissionTier.TIER_2_MUTATION == PermissionTier.TIER_2_AUDITED
        assert PermissionTier.TIER_2_AUDITED == PermissionTier.TIER_2_MUTATION
        assert PermissionTier.TIER_3_HIGH_IMPACT == PermissionTier.TIER_3_CONFIRMATION
        assert PermissionTier.TIER_3_CONFIRMATION == PermissionTier.TIER_3_HIGH_IMPACT

        # Non-matching tiers
        assert PermissionTier.TIER_1_READ_ONLY != PermissionTier.TIER_2_MUTATION
        assert PermissionTier.TIER_1_READ_ONLY != "TIER_2_MUTATION"

        # 4. Unrelated types
        assert not (t == 42)
        assert t != 42
        assert not (t == None)
        assert t != None
        assert not (t == 3.14)
        assert t != 3.14
        assert not (t == ["TIER_1_READ_ONLY"])
        assert t != ["TIER_1_READ_ONLY"]

        # 5. Zero DeprecationWarning for NotImplemented in boolean context
        dep_warnings = [
            w for w in recorded_warnings
            if issubclass(w.category, DeprecationWarning) and "NotImplemented" in str(w.message)
        ]
        assert len(dep_warnings) == 0, f"Expected 0 DeprecationWarnings, got: {[str(w.message) for w in dep_warnings]}"
