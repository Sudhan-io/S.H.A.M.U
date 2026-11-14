import asyncio
import time
import pytest
from PySide6.QtWidgets import QApplication
from shamu.app import ShamuApp
from shamu.config import ShamuConfig, MemoryConfig
from shamu.types import ChatMessage, Role, ChatChunk, MemoryRecord
from shamu.memory.sqlite_store import SQLiteMemoryStore
from shamu.memory.hybrid import HybridRetriever
from shamu.memory.context_compactor import ContextCompactor, estimate_history_tokens
from shamu.security.approval_manager import ApprovalRequest
from shamu.security.tool_registry import PermissionTier


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_shamu_app_initialization(qapp, monkeypatch):
    # Prevent starting real native Windows hotkey hook during CI unit test
    monkeypatch.setattr("shamu.ui.hotkey.GlobalHotkeyListener.start", lambda self: None)
    monkeypatch.setattr("shamu.app.AgentWorker.start", lambda self: None)

    app = ShamuApp()
    assert app.window is not None
    assert app.tray is not None
    assert app.gateway is not None
    assert app.firewall is not None
    assert app.worker is not None
    assert app.memory_store is not None
    assert app.retriever is not None
    assert app.compactor is not None
    assert app.memory_init_error is None

    # Test toggling window
    app.window.hide()
    assert not app.window.isVisible()
    app.toggle_window()
    assert app.window.isVisible()

    app.quit()


def test_voice_transcription_displays_in_window(qapp, monkeypatch):
    monkeypatch.setattr("shamu.ui.hotkey.GlobalHotkeyListener.start", lambda self: None)
    monkeypatch.setattr("shamu.app.AgentWorker.start", lambda self: None)

    app = ShamuApp()
    initial_count = app.window.chat_layout.count()

    # Emitting user_message_sent (as voice pipeline does) should add user card to window
    app.bridge.user_message_sent.emit("Hello from voice pipeline")

    assert app.window.chat_layout.count() == initial_count + 1
    card = app.window.chat_layout.itemAt(initial_count - 1).widget()
    assert card._raw_text == "Hello from voice pipeline"

    # Typing a message and sending does not duplicate
    app.window.input_field.setText("Typed input")
    app.window._on_send()
    assert app.window.chat_layout.count() == initial_count + 2
    typed_card = app.window.chat_layout.itemAt(initial_count).widget()
    assert typed_card._raw_text == "Typed input"

    app.quit()


def test_approval_modal_slot_approves(qapp, monkeypatch):
    monkeypatch.setattr("shamu.ui.hotkey.GlobalHotkeyListener.start", lambda self: None)
    monkeypatch.setattr("shamu.app.AgentWorker.start", lambda self: None)

    app = ShamuApp()

    # Mock ApprovalModal to simulate user approval
    def fake_exec(modal_self):
        modal_self.approved = True
        modal_self.double_confirmed = True
        return 1

    monkeypatch.setattr("shamu.app.ApprovalModal.exec", fake_exec)

    req = ApprovalRequest(
        request_id="test-req-1",
        tool_name="shell_execute",
        arguments={"cmd": "dir"},
        canonical_argument_hash="",
        tier=PermissionTier.TIER_3_HIGH_IMPACT,
        reason="Requires double confirmation",
        requires_double_confirmation=True,
    )

    loop = asyncio.new_event_loop()
    fut = loop.create_future()

    app.bridge.approval_requested.emit(req, fut)
    qapp.processEvents()

    assert fut.done()
    resp = fut.result()
    assert resp.approved is True
    assert resp.double_confirmed is True
    assert resp.request_id == "test-req-1"

    loop.close()
    app.quit()





def test_approval_modal_slot_handles_exception(qapp, monkeypatch):
    monkeypatch.setattr("shamu.ui.hotkey.GlobalHotkeyListener.start", lambda self: None)
    monkeypatch.setattr("shamu.app.AgentWorker.start", lambda self: None)

    app = ShamuApp()

    def fake_init(modal_self, *args, **kwargs):
        raise RuntimeError("Failed to create modal window")

    monkeypatch.setattr("shamu.app.ApprovalModal.__init__", fake_init)

    req = ApprovalRequest(
        request_id="test-req-err",
        tool_name="delete_database",
        arguments={},
        canonical_argument_hash="",
        tier=PermissionTier.TIER_3_HIGH_IMPACT,
    )

    loop = asyncio.new_event_loop()
    fut = loop.create_future()

    app.bridge.approval_requested.emit(req, fut)
    qapp.processEvents()

    assert fut.done()
    resp = fut.result()
    assert resp.approved is False
    assert "Modal display error" in resp.reason

    loop.close()
    app.quit()


def test_cross_thread_approval_deadlock_prevention(qapp, monkeypatch):
    import threading
    import time

    monkeypatch.setattr("shamu.ui.hotkey.GlobalHotkeyListener.start", lambda self: None)
    monkeypatch.setattr("shamu.app.AgentWorker.start", lambda self: None)

    app = ShamuApp()

    # Mock ApprovalModal exec
    def fake_exec(modal_self):
        modal_self.approved = True
        return 1

    monkeypatch.setattr("shamu.app.ApprovalModal.exec", fake_exec)

    result_container = []
    exc_container = []

    def background_worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            token = loop.run_until_complete(
                app.firewall.approval_manager.request_approval(
                    tool_name="shell_execute",
                    arguments={"cmd": "whoami"},
                    tier=PermissionTier.TIER_3_HIGH_IMPACT,
                    timeout_sec=5.0,
                )
            )
            result_container.append(token)
        except Exception as exc:
            exc_container.append(exc)
        finally:
            loop.close()

    t = threading.Thread(target=background_worker)
    t.start()

    # Main GUI thread pumps Qt events while background worker awaits
    start_time = time.time()
    while t.is_alive() and (time.time() - start_time) < 5.0:
        qapp.processEvents()
        time.sleep(0.01)

    t.join(timeout=2.0)
    assert not t.is_alive(), "Worker deadlocked on cross-thread approval modal!"
    assert not exc_container, f"Exception in worker: {exc_container}"
    assert len(result_container) == 1
    assert result_container[0].tool_name == "shell_execute"

    app.quit()


def test_gui_memory_initialization_when_enabled(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr("shamu.ui.hotkey.GlobalHotkeyListener.start", lambda self: None)
    monkeypatch.setattr("shamu.app.AgentWorker.start", lambda self: None)

    db_path = str(tmp_path / "gui_mem_test.db")
    cfg = ShamuConfig(memory=MemoryConfig(db_path=db_path, fts_enabled=True, embedding_enabled=False))

    app = ShamuApp(cfg=cfg)
    assert app.memory_store is not None
    assert isinstance(app.memory_store, SQLiteMemoryStore)
    assert app.retriever is not None
    assert isinstance(app.retriever, HybridRetriever)
    assert app.compactor is not None
    assert isinstance(app.compactor, ContextCompactor)
    assert app.memory_init_error is None

    assert app.worker.retriever is app.retriever
    assert app.worker.compactor is app.compactor

    app.quit()


@pytest.mark.asyncio
async def test_gui_memory_retrieval_chat_execution_path(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr("shamu.ui.hotkey.GlobalHotkeyListener.start", lambda self: None)
    monkeypatch.setattr("shamu.app.AgentWorker.start", lambda self: None)

    db_path = str(tmp_path / "retrieval_chat_test.db")
    cfg = ShamuConfig(memory=MemoryConfig(db_path=db_path, fts_enabled=True, embedding_enabled=False))

    app = ShamuApp(cfg=cfg)
    assert app.memory_store is not None

    # Seed memory record into SQLite store
    rec = MemoryRecord(
        id="mem-target-hw",
        content="Hardware: Lenovo LOQ with RTX 3050 Laptop GPU and i5-12450HX CPU",
        category="hardware",
        source="system",
    )
    await app.memory_store.store(rec)

    # Mock gateway stream_chat to return a realistic model response
    received_stream_history = []

    async def mock_stream_chat(history, tools=None):
        received_stream_history.extend(history)
        yield ChatChunk(text_delta="Your GPU is RTX 3050.")

    monkeypatch.setattr(app.gateway, "stream_chat", mock_stream_chat)

    # Spy on retriever.retrieve to verify arguments
    original_retrieve = app.retriever.retrieve
    retrieve_calls = []

    async def spy_retrieve(query, limit=None):
        retrieve_calls.append({"query": query, "limit": limit})
        return await original_retrieve(query, limit=limit)

    monkeypatch.setattr(app.retriever, "retrieve", spy_retrieve)

    # Execute chat turn via real AgentWorker._process_chat
    user_query = "What GPU is configured on this machine?"
    await app.worker._process_chat(user_query)

    # Verify retrieval was invoked with limit=3
    assert len(retrieve_calls) == 1
    assert retrieve_calls[0]["query"] == user_query
    assert retrieve_calls[0]["limit"] == 3

    # Verify history state: retrieved context is attached to Role.USER, not Role.SYSTEM
    assert len(app.worker.history) == 2
    user_msg = app.worker.history[0]
    assert user_msg.role == Role.USER
    assert "### Retrieved Relevant Knowledge & Context:" in user_msg.content
    assert "RTX 3050 Laptop GPU" in user_msg.content
    assert user_query in user_msg.content
    assert not any(m.role == Role.SYSTEM for m in app.worker.history)

    # Second message is assistant reply
    asst_msg = app.worker.history[1]
    assert asst_msg.role == Role.ASSISTANT
    assert asst_msg.content == "Your GPU is RTX 3050."

    assert app.worker.last_retrieval_error is None

    # Test follow-up turn does not duplicate the context header or inject into Role.SYSTEM
    await app.worker._process_chat("Tell me more about the CPU")
    assert not any(m.role == Role.SYSTEM for m in app.worker.history)
    user_msg_2 = app.worker.history[2]
    assert user_msg_2.role == Role.USER
    assert "Tell me more about the CPU" in user_msg_2.content

    app.quit()


@pytest.mark.asyncio
async def test_gui_context_compaction_integration(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr("shamu.ui.hotkey.GlobalHotkeyListener.start", lambda self: None)
    monkeypatch.setattr("shamu.app.AgentWorker.start", lambda self: None)

    db_path = str(tmp_path / "compaction_test.db")
    # Low token budget to trigger compaction during multi-turn chat
    cfg = ShamuConfig(memory=MemoryConfig(db_path=db_path, fts_enabled=True, context_token_budget=120, embedding_enabled=False))

    app = ShamuApp(cfg=cfg)

    async def mock_stream_chat(history, tools=None):
        yield ChatChunk(text_delta="Reply acknowledging turn.")

    monkeypatch.setattr(app.gateway, "stream_chat", mock_stream_chat)

    # Execute multiple chat turns to exceed budget
    for i in range(1, 6):
        await app.worker._process_chat(f"This is user prompt number {i} detailing background specifications.")

    # Context compactor must have kept history bounded within token budget
    assert app.worker.last_compaction_error is None

    # Verify pruned messages were archived into SQLiteMemoryStore
    archived = await app.memory_store.search_lexical("specifications")
    assert len(archived) >= 1
    assert any(m.category == "conversation_history" for m in archived)

    app.quit()


def test_gui_memory_initialization_failure_is_observable(qapp, monkeypatch):
    monkeypatch.setattr("shamu.ui.hotkey.GlobalHotkeyListener.start", lambda self: None)
    monkeypatch.setattr("shamu.app.AgentWorker.start", lambda self: None)

    def fail_store_init(*args, **kwargs):
        raise OSError("Permission denied creating memory database")

    monkeypatch.setattr("shamu.app.SQLiteMemoryStore.__init__", fail_store_init)

    app = ShamuApp()
    assert app.memory_init_error is not None
    assert "Permission denied creating memory database" in app.memory_init_error
    assert app.memory_store is None
    assert app.retriever is None
    assert app.compactor is None
    assert app.worker.retriever is None
    assert app.worker.compactor is None

    app.quit()


@pytest.mark.asyncio
async def test_gui_memory_retrieval_failure_diagnosable(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr("shamu.ui.hotkey.GlobalHotkeyListener.start", lambda self: None)
    monkeypatch.setattr("shamu.app.AgentWorker.start", lambda self: None)

    db_path = str(tmp_path / "retrieval_err_test.db")
    cfg = ShamuConfig(memory=MemoryConfig(db_path=db_path, fts_enabled=True, embedding_enabled=False))
    app = ShamuApp(cfg=cfg)

    async def broken_retrieve(query, limit=None):
        raise RuntimeError("FTS index corruption error")

    monkeypatch.setattr(app.worker.retriever, "retrieve", broken_retrieve)

    async def mock_stream_chat(history, tools=None):
        yield ChatChunk(text_delta="Still responding despite retrieval failure.")

    monkeypatch.setattr(app.gateway, "stream_chat", mock_stream_chat)

    await app.worker._process_chat("Hello system")
    assert app.worker.last_retrieval_error == "FTS index corruption error"
    # Verify chat execution didn't fail
    assert len(app.worker.history) >= 2
    assert app.worker.history[-1].content == "Still responding despite retrieval failure."

    app.quit()


@pytest.mark.asyncio
async def test_gui_context_compaction_failure_diagnosable(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr("shamu.ui.hotkey.GlobalHotkeyListener.start", lambda self: None)
    monkeypatch.setattr("shamu.app.AgentWorker.start", lambda self: None)

    db_path = str(tmp_path / "compact_err_test.db")
    cfg = ShamuConfig(memory=MemoryConfig(db_path=db_path, fts_enabled=True, embedding_enabled=False))
    app = ShamuApp(cfg=cfg)

    async def broken_compact(messages, budget_override=None):
        raise ValueError("Simulated token compaction failure")

    monkeypatch.setattr(app.worker.compactor, "compact", broken_compact)

    async def mock_stream_chat(history, tools=None):
        yield ChatChunk(text_delta="Chat works.")

    monkeypatch.setattr(app.gateway, "stream_chat", mock_stream_chat)

    await app.worker._process_chat("Compaction test message")
    assert app.worker.last_compaction_error == "Simulated token compaction failure"
    assert app.worker.history[-1].content == "Chat works."

    app.quit()







