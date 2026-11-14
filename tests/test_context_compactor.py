import pytest
from pathlib import Path
from shamu.config import MemoryConfig
from shamu.types import ChatMessage, Role, ToolCall
from shamu.memory.sqlite_store import SQLiteMemoryStore
from shamu.memory.context_compactor import (
    ContextCompactor,
    estimate_message_tokens,
    estimate_history_tokens,
)
from PySide6.QtWidgets import QApplication


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app



def test_token_estimator_basic():
    msg = ChatMessage(role=Role.USER, content="Hello, SHAMU! How are you doing today?")
    tokens = estimate_message_tokens(msg)
    # 41 chars / ~3.8 = ~11 + 4 framing = ~15 tokens
    assert 10 <= tokens <= 25

    system_msg = ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant.")
    assert 8 <= estimate_message_tokens(system_msg) <= 20


def test_token_estimator_with_tool_calls():
    msg = ChatMessage(
        role=Role.ASSISTANT,
        content=None,
        tool_calls=[
            ToolCall(id="tc-1", name="read_file", arguments='{"path": "D:/workspace/main.py"}')
        ],
    )
    tokens = estimate_message_tokens(msg)
    assert tokens >= 10


@pytest.mark.asyncio
async def test_context_compaction_under_budget_leaves_messages_intact(tmp_path: Path):
    cfg = MemoryConfig(db_path=str(tmp_path / "compact_test.db"), context_token_budget=4000)
    store = SQLiteMemoryStore(cfg)
    compactor = ContextCompactor(store, cfg)

    messages = [
        ChatMessage(role=Role.SYSTEM, content="System prompt"),
        ChatMessage(role=Role.USER, content="Turn 1 user"),
        ChatMessage(role=Role.ASSISTANT, content="Turn 1 assistant"),
        ChatMessage(role=Role.USER, content="Turn 2 user"),
    ]

    compacted, archived_count = await compactor.compact(messages)
    assert len(compacted) == 4
    assert archived_count == 0
    assert compacted == messages
    store.close()


@pytest.mark.asyncio
async def test_context_compactor_prunes_oldest_turns_when_over_budget(tmp_path: Path):
    # Set a tiny budget to trigger pruning
    cfg = MemoryConfig(db_path=str(tmp_path / "compact_prune.db"), context_token_budget=120)
    store = SQLiteMemoryStore(cfg)
    compactor = ContextCompactor(store, cfg)

    # Build 10 turns of messages
    messages = [ChatMessage(role=Role.SYSTEM, content="You are SHAMU.")]
    for i in range(1, 10):
        messages.append(ChatMessage(role=Role.USER, content=f"Long turn {i} user message explaining details about task {i}."))
        messages.append(ChatMessage(role=Role.ASSISTANT, content=f"Long turn {i} assistant reply acknowledging details for task {i}."))
    messages.append(ChatMessage(role=Role.USER, content="Latest critical request from the user."))

    assert estimate_history_tokens(messages) > 120

    compacted, archived_count = await compactor.compact(messages)

    # Must be within budget (or minimal essential)
    assert archived_count > 0
    # System prompt MUST be first
    assert compacted[0].role == Role.SYSTEM
    assert compacted[0].content == "You are SHAMU."
    # Latest user message MUST be last
    assert compacted[-1].role == Role.USER
    assert compacted[-1].content == "Latest critical request from the user."
    store.close()


@pytest.mark.asyncio
async def test_archiving_pruned_messages_to_sqlite(tmp_path: Path):
    cfg = MemoryConfig(db_path=str(tmp_path / "compact_archive.db"), context_token_budget=35)
    store = SQLiteMemoryStore(cfg)
    compactor = ContextCompactor(store, cfg)

    messages = [
        ChatMessage(role=Role.SYSTEM, content="System instructions"),
        ChatMessage(role=Role.USER, content="Old unique user turn talking about quantum gravity"),
        ChatMessage(role=Role.ASSISTANT, content="Old unique assistant reply discussing string theory"),
        ChatMessage(role=Role.USER, content="Current user turn"),
    ]

    compacted, archived_count = await compactor.compact(messages)
    assert archived_count >= 1

    # Pruned messages must be stored in SQLite and searchable via lexical search
    archived = await store.search_lexical("quantum gravity")
    assert len(archived) >= 1
    assert "quantum gravity" in archived[0].content
    assert archived[0].category == "conversation_history"
    store.close()


@pytest.mark.asyncio
async def test_agent_worker_history_compaction_integration(qapp, tmp_path: Path):
    """ARCH-04: AgentWorker must auto-compact history to list type and archive pruned turns to SQLite."""
    from shamu.app import AgentWorker, UIBridge
    from shamu.types import ChatChunk

    class MockStreamingGateway:
        def __init__(self, turns_chunks):
            self.turns_chunks = turns_chunks
            self.call_count = 0

        async def stream_chat(self, history, tools=None):
            chunks = self.turns_chunks[self.call_count % len(self.turns_chunks)]
            self.call_count += 1
            for c in chunks:
                yield c

    db_path = tmp_path / "agent_memory.db"
    cfg = MemoryConfig(db_path=str(db_path), context_token_budget=100)
    store = SQLiteMemoryStore(cfg)
    compactor = ContextCompactor(store, cfg)
    bridge = UIBridge()

    # Create dummy gateway returning short replies
    chunks = [ChatChunk(text_delta="Reply turn"), ChatChunk(finish_reason="stop")]
    gateway = MockStreamingGateway([chunks for _ in range(10)])

    worker = AgentWorker(
        gateway=gateway,
        bridge=bridge,
        compactor=compactor,
    )

    # Populate worker with long conversation turns
    worker.history.append(ChatMessage(role=Role.SYSTEM, content="System Prompt Instructions"))
    for i in range(1, 10):
        worker.history.append(ChatMessage(role=Role.USER, content=f"Historical User Turn {i} with extensive detail"))
        worker.history.append(ChatMessage(role=Role.ASSISTANT, content=f"Historical Assistant Turn {i} with extensive detail"))

    # Now trigger _process_chat
    await worker._process_chat("Final trigger message")

    # Verify history is still a valid Python list of ChatMessage, NOT a tuple
    assert isinstance(worker.history, list)
    for m in worker.history:
        assert isinstance(m, ChatMessage)

    # System prompt preserved at index 0
    assert worker.history[0].role == Role.SYSTEM

    # Pruned messages must be present in SQLite archive
    archived = await store.search_lexical("Historical User Turn 1")
    assert len(archived) >= 1
    assert "Historical User Turn 1" in archived[0].content

    store.close()


@pytest.mark.asyncio
async def test_compact_single_message_no_duplication(tmp_path: Path):
    """CC-02: Compacting a single message must never duplicate it even if over budget."""
    cfg = MemoryConfig(db_path=str(tmp_path / "single_msg.db"), context_token_budget=10)
    store = SQLiteMemoryStore(cfg)
    compactor = ContextCompactor(store, cfg)

    # 1. Single user message over budget
    user_msg = ChatMessage(role=Role.USER, content="This is a long user message that exceeds the tiny 10 token budget.")
    compacted, pruned = await compactor.compact([user_msg])
    assert len(compacted) == 1
    assert compacted[0] == user_msg
    assert pruned == 0

    # 2. Single system message over budget
    sys_msg = ChatMessage(role=Role.SYSTEM, content="This is a long system prompt exceeding budget.")
    compacted_sys, pruned_sys = await compactor.compact([sys_msg])
    assert len(compacted_sys) == 1
    assert compacted_sys[0] == sys_msg
    assert pruned_sys == 0

    store.close()


@pytest.mark.asyncio
async def test_compact_atomic_tool_call_turn_pruning(tmp_path: Path):
    """CC-01: Assistant tool call and corresponding tool responses must be grouped and pruned atomically.
    Never leave an orphaned Role.TOOL message."""
    cfg = MemoryConfig(db_path=str(tmp_path / "atomic_tool.db"), context_token_budget=60)
    store = SQLiteMemoryStore(cfg)
    compactor = ContextCompactor(store, cfg)

    tc1 = ToolCall(id="tc-1", name="fetch_file", arguments='{"file": "data.txt"}')
    tc2 = ToolCall(id="tc-2", name="parse_file", arguments='{"format": "json"}')

    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=Role.USER, content="Please fetch and parse data.txt for the report."),
        # Tool call turn with 2 tool calls and 2 tool results
        ChatMessage(role=Role.ASSISTANT, content=None, tool_calls=[tc1, tc2]),
        ChatMessage(role=Role.TOOL, content="data.txt contents here", tool_call_id="tc-1"),
        ChatMessage(role=Role.TOOL, content="parsed JSON object", tool_call_id="tc-2"),
        ChatMessage(role=Role.ASSISTANT, content="I have fetched and parsed data.txt successfully."),
        ChatMessage(role=Role.USER, content="Now please summarize the parsed JSON results."),
    ]

    compacted, pruned_count = await compactor.compact(messages, budget_override=50)

    # Verify no orphaned tool message exists in compacted
    for idx, msg in enumerate(compacted):
        if msg.role == Role.TOOL:
            assert idx > 0, "Tool message cannot be first in history"
            prev = compacted[idx - 1]
            assert (prev.role == Role.ASSISTANT and prev.tool_calls) or prev.role == Role.TOOL, \
                f"Tool message at index {idx} was orphaned after {prev.role}"

    # If the tool turn was pruned, both assistant and both tool messages were pruned
    roles_in_compacted = [m.role for m in compacted]
    if Role.TOOL not in roles_in_compacted:
        # Tool call turn was pruned
        assert not any(m.role == Role.ASSISTANT and m.tool_calls for m in compacted)

    store.close()


@pytest.mark.asyncio
async def test_compact_archives_tool_calls_with_empty_or_none_content(tmp_path: Path):
    """CC-01: Archiving must archive tool calls to SQLite even when content is empty/None."""
    cfg = MemoryConfig(db_path=str(tmp_path / "archive_empty_content.db"), context_token_budget=30)
    store = SQLiteMemoryStore(cfg)
    compactor = ContextCompactor(store, cfg)

    tc = ToolCall(id="tc-search", name="web_search", arguments='{"query": "antigravity deepmind"}')
    messages = [
        ChatMessage(role=Role.SYSTEM, content="System instructions"),
        ChatMessage(role=Role.USER, content="Search for antigravity deepmind"),
        ChatMessage(role=Role.ASSISTANT, content=None, tool_calls=[tc]),
        ChatMessage(role=Role.TOOL, content="Found antigravity project", tool_call_id="tc-search"),
        ChatMessage(role=Role.ASSISTANT, content="Search finished."),
        ChatMessage(role=Role.USER, content="Latest user query to retain."),
    ]

    compacted, pruned_count = await compactor.compact(messages, budget_override=30)
    assert pruned_count >= 1

    # Verify tool call was archived despite content=None
    results = await store.search_lexical("web_search")
    assert len(results) >= 1
    assert "web_search" in results[0].content
    assert "antigravity deepmind" in results[0].content
    assert results[0].metadata.get("role") == "assistant"
    assert "tool_calls" in results[0].metadata

    store.close()


@pytest.mark.asyncio
async def test_compact_preserves_latest_tool_turn_without_orphaning(tmp_path: Path):
    """CC-01: When conversation ends with tool call turn, it is preserved atomically as latest turn."""
    cfg = MemoryConfig(db_path=str(tmp_path / "latest_tool.db"), context_token_budget=70)
    store = SQLiteMemoryStore(cfg)
    compactor = ContextCompactor(store, cfg)

    tc = ToolCall(id="tc-latest", name="run_calc", arguments='{"x": 42}')
    messages = [
        ChatMessage(role=Role.SYSTEM, content="System prompt"),
        ChatMessage(role=Role.USER, content="Old question about biology with long description."),
        ChatMessage(role=Role.ASSISTANT, content="Old answer about biology with extensive explanation."),
        # Latest turn is tool call + tool response
        ChatMessage(role=Role.ASSISTANT, content=None, tool_calls=[tc]),
        ChatMessage(role=Role.TOOL, content="result: 42", tool_call_id="tc-latest"),
    ]

    compacted, pruned = await compactor.compact(messages, budget_override=60)
    assert pruned > 0

    # Compacted should keep system and latest tool turn (assistant + tool)
    assert compacted[0].role == Role.SYSTEM
    assert compacted[-2].role == Role.ASSISTANT
    assert compacted[-2].tool_calls is not None
    assert compacted[-1].role == Role.TOOL
    assert compacted[-1].content == "result: 42"

    store.close()


