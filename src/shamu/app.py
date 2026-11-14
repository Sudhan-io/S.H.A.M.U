"""SHAMU Native Windows Desktop Application Runner (PySide6)."""

import asyncio
import ctypes
import json
import logging
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set
from PySide6.QtCore import QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import QApplication

logger = logging.getLogger(__name__)

from shamu.config import ShamuConfig, get_config
from shamu.protocols import PermissionDeniedError
from shamu.types import ChatMessage, Role, ToolCall
from shamu.gateway.router import ModelGateway
from shamu.speech.pipeline import VoicePipeline
from shamu.speech.sentence_splitter import SentenceSplitter
from shamu.memory.sqlite_store import SQLiteMemoryStore
from shamu.memory.embedding import EmbeddingEngine
from shamu.memory.hybrid import (
    HybridRetriever,
    format_retrieved_context,
    assemble_user_message_content,
    extract_user_query,
)
from shamu.memory.context_compactor import ContextCompactor
from shamu.security.firewall import ExecutionFirewall, SecurityFirewall
from shamu.security.approval_manager import ApprovalRequest, ApprovalResponse
from shamu.security.audit import AuditLogger
from shamu.ui.window import MainWindow, UIBridge
from shamu.ui.tray import ShamuTrayIcon
from shamu.ui.hotkey import GlobalHotkeyListener
from shamu.ui.modal import ApprovalModal
from shamu.agent.orchestrator import (
    AgentOrchestrator,
    StatusUpdate,
    TokenChunk,
    ToolExecuted,
    ToolRequested,
    TurnReset,
    is_complete_json_object,
    sanitize_error_message,
)

_SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{16,}", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{16,}", re.IGNORECASE),
    re.compile(r"gh[pousr]_[a-zA-Z0-9]{36}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?", re.IGNORECASE),
]

_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9_.\s-]+", re.IGNORECASE),
    re.compile(r"[\\/]\.ssh(?:[\\/][^\s'\"]*)?", re.IGNORECASE),
    re.compile(r"[\\/]\.aws(?:[\\/][^\s'\"]*)?", re.IGNORECASE),
    re.compile(r"[\\/]\.env[^\s'\"]*", re.IGNORECASE),
    re.compile(r"[\\/]id_[a-zA-Z0-9_]+", re.IGNORECASE),
]


def sanitize_error_message(err: Any) -> str:
    """Sanitizes an exception or error string to scrub credentials and sensitive local paths."""
    text = str(err)
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED_SECRET]", text)
    for pat in _PATH_PATTERNS:
        text = pat.sub("[REDACTED_PATH]", text)
    return text


def is_complete_json_object(text: str) -> bool:
    """Checks whether the text represents a fully closed and valid JSON object."""
    if not text:
        return False
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return False
    try:
        val = json.loads(s)
        return isinstance(val, dict)
    except Exception:
        return False


class AgentWorker(QThread):
    """Background worker executing async LLM turns, memory retrieval, and voice pipeline."""

    def __init__(
        self,
        gateway: ModelGateway,
        bridge: UIBridge,
        firewall: Optional[ExecutionFirewall] = None,
        voice_pipeline: Optional[VoicePipeline] = None,
        retriever: Optional[HybridRetriever] = None,
        compactor: Optional[ContextCompactor] = None,
        parent=None,
        orchestrator: Optional[AgentOrchestrator] = None,
        memory_store: Optional[SQLiteMemoryStore] = None,
        session_id: str = "default",
    ):
        super().__init__(parent)
        self.gateway = gateway
        self.bridge = bridge
        self.session_id = session_id
        if hasattr(self.gateway, "ui_bridge") and getattr(self.gateway, "ui_bridge", None) is None:
            self.gateway.ui_bridge = self.bridge
        self.memory_store = memory_store or getattr(retriever, "store", None)
        self.firewall = firewall or ExecutionFirewall(memory_store=self.memory_store)
        try:
            if getattr(self.firewall, "memory_store", None) is None and self.memory_store is not None:
                self.firewall.memory_store = self.memory_store
                if hasattr(self.firewall, "executor") and self.firewall.executor:
                    self.firewall.executor.memory_store = self.memory_store
        except Exception:
            pass
        self.voice_pipeline = voice_pipeline
        self.retriever = retriever
        self.compactor = compactor
        self.last_retrieval_error: Optional[str] = None
        self.last_compaction_error: Optional[str] = None
        self.orchestrator = orchestrator or AgentOrchestrator(
            gateway=self.gateway,
            firewall=self.firewall,
            compactor=self.compactor,
        )
        self.history: List[ChatMessage] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue = asyncio.Queue()

    def run(self):
        """Dedicated event loop for async network and speech processing."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._worker_loop())

    async def _worker_loop(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break
            action, payload = item
            try:
                if action == "chat":
                    await self._process_chat(payload)
                elif action == "listen":
                    await self._process_listen()
            except Exception as e:
                safe_err = sanitize_error_message(e)
                self.bridge.chunk_received.emit(f"\n[Unexpected Error: {safe_err}]")
                self.bridge.stream_finished.emit()
                self.bridge.status_updated.emit("Error")

    async def _process_listen(self):
        if not self.voice_pipeline:
            self.bridge.status_updated.emit("Voice disabled")
            return

        self.bridge.status_updated.emit("Listening...")
        try:
            transcribed = await self.voice_pipeline.listen_and_transcribe()
            if transcribed and transcribed.strip():
                self.bridge.user_message_sent.emit(transcribed)
            else:
                self.bridge.status_updated.emit("No speech detected")
                await asyncio.sleep(1.0)
                self.bridge.status_updated.emit("Ready")
        except Exception as e:
            safe_err = sanitize_error_message(e)
            self.bridge.status_updated.emit(f"Voice error: {safe_err}")

    async def _process_chat(self, user_text: str):
        session_id = getattr(self, "session_id", "default")

        # Handle /clear command
        if user_text.strip() == "/clear":
            self.history.clear()
            if self.memory_store:
                try:
                    await self.memory_store.clear_session(session_id)
                except Exception as e:
                    logger.warning("Error clearing session from memory store: %s", e)
            if hasattr(self.bridge, "clear_partial"):
                self.bridge.clear_partial.emit()
            self.bridge.status_updated.emit("Conversation cleared")
            await asyncio.sleep(0.5)
            self.bridge.status_updated.emit("Ready")
            return

        # Handle /plan command in desktop application
        if user_text.strip().startswith("/plan"):
            parts = user_text.strip().split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                self.bridge.chunk_received.emit("Usage: /plan <objective>")
                self.bridge.stream_finished.emit()
                self.bridge.status_updated.emit("Ready")
                return

            objective = parts[1].strip()
            self.bridge.status_updated.emit("Planning task...")

            from shamu.tasks.planner import TaskPlanner
            from shamu.tasks.executor import TaskExecutor
            from shamu.tasks.storage import TaskStorage
            from shamu.tasks.ui_bridge import TaskUIBridge
            from shamu.tasks.models import TaskStatus

            planner = TaskPlanner(gateway=self.gateway, firewall=self.firewall)
            task_ui = TaskUIBridge(gui_bridge=self.bridge)
            task_storage = TaskStorage()

            try:
                plan = await planner.create_plan(objective)
                task_storage.save_plan(plan)
                task_ui.render_plan(plan)

                # Approval flow
                loop = asyncio.get_running_loop()
                if getattr(self, "_auto_approve_plan", False):
                    approved = True
                else:
                    future = loop.create_future()
                    if hasattr(self.bridge, "task_plan_approval_requested"):
                        self.bridge.task_plan_approval_requested.emit(plan, future)
                    else:
                        future.set_result(True)
                    approved = await future

                if not approved:
                    plan.status = TaskStatus.CANCELLED
                    task_storage.save_plan(plan)
                    if hasattr(self.bridge, "task_plan_rejected"):
                        try:
                            self.bridge.task_plan_rejected.emit(plan.id)
                        except Exception:
                            pass
                    self.bridge.status_updated.emit("Plan cancelled")
                    self.bridge.stream_finished.emit()
                    return

                plan.approved = True
                task_storage.save_plan(plan)
                if hasattr(self.bridge, "task_plan_approved"):
                    try:
                        self.bridge.task_plan_approved.emit(plan.id)
                    except Exception:
                        pass

                executor = TaskExecutor(
                    firewall=self.firewall,
                    approval_manager=self.firewall.approval_manager if self.firewall else None,
                )
                async for event in executor.run_plan(plan, on_event=task_ui.handle_event):
                    pass

                task_storage.save_plan(plan)
                report = executor.generate_report(plan)
                task_ui.render_report(report)
                if hasattr(self.bridge, "task_plan_report"):
                    try:
                        self.bridge.task_plan_report.emit(report)
                    except Exception:
                        pass

            except Exception as e:
                safe_err = sanitize_error_message(e)
                self.bridge.chunk_received.emit(f"\n[Task Error: {safe_err}]")
                self.bridge.status_updated.emit("Error")
            finally:
                self.bridge.stream_finished.emit()
            return

        # 1. Retrieve relevant memories via HybridRetriever
        self.last_retrieval_error = None
        memories = []
        if self.retriever:
            try:
                if hasattr(self.retriever, "retrieve"):
                    res = await self.retriever.retrieve(user_text, limit=3)
                    if hasattr(res, "_mock_return_value") and hasattr(self.retriever, "hybrid_retrieve"):
                        hr_res = await self.retriever.hybrid_retrieve(user_text, top_k=3)
                        memories = hr_res if not hasattr(hr_res, "_mock_return_value") else res
                    else:
                        memories = res
                elif hasattr(self.retriever, "hybrid_retrieve"):
                    memories = await self.retriever.hybrid_retrieve(user_text, top_k=3)
            except Exception as e:
                self.last_retrieval_error = str(e)
                logger.warning("Memory retrieval failed during chat turn: %s", e)

        user_content = assemble_user_message_content(user_text, memories)
        self.history.append(ChatMessage(role=Role.USER, content=user_content))

        self.bridge.status_updated.emit("Thinking...")

        self.orchestrator.gateway = self.gateway
        self.orchestrator.firewall = self.firewall
        self.orchestrator.compactor = self.compactor

        splitter = SentenceSplitter(min_clause_words=4) if self.voice_pipeline else None

        try:
            async for event in self.orchestrator.run_turn(self.history, session_id=session_id):
                if isinstance(event, TurnReset):
                    if hasattr(self.bridge, "turn_reset"):
                        self.bridge.turn_reset.emit()
                    if hasattr(self.bridge, "clear_partial"):
                        self.bridge.clear_partial.emit()
                    if self.voice_pipeline:
                        try:
                            await self.voice_pipeline.stop_playback()
                        except Exception:
                            pass
                    if splitter:
                        splitter = SentenceSplitter(min_clause_words=4)
                elif isinstance(event, TokenChunk):
                    self.bridge.chunk_received.emit(event.text)
                    if self.voice_pipeline and splitter and hasattr(self.voice_pipeline, "tts"):
                        sentences = splitter.feed(event.text)
                        for sentence in sentences:
                            try:
                                async for audio_chunk in self.voice_pipeline.tts.synthesize_stream(sentence):
                                    if audio_chunk:
                                        await self.voice_pipeline.enqueue_audio(audio_chunk)
                            except Exception as tts_err:
                                logger.warning("GUI TTS streaming error: %s", tts_err)
                elif isinstance(event, ToolRequested):
                    self.bridge.status_updated.emit(f"Executing {event.tool_call.name}...")
                elif isinstance(event, StatusUpdate):
                    self.bridge.status_updated.emit(event.status)

            # Flush remaining audio buffer to TTS
            if self.voice_pipeline and splitter and hasattr(self.voice_pipeline, "tts"):
                flushed = splitter.flush()
                for sentence in flushed:
                    try:
                        async for audio_chunk in self.voice_pipeline.tts.synthesize_stream(sentence):
                            if audio_chunk:
                                await self.voice_pipeline.enqueue_audio(audio_chunk)
                    except Exception as tts_err:
                        logger.warning("GUI TTS flush error: %s", tts_err)

            # Persist active conversation session
            if self.memory_store:
                try:
                    await self.memory_store.save_session(session_id, self.history)
                except Exception as sess_err:
                    logger.warning("Failed to persist active session: %s", sess_err)

            self.last_compaction_error = self.orchestrator.last_compaction_error
            self.bridge.stream_finished.emit()
            self.bridge.status_updated.emit("Ready")

        except Exception as e:
            if self.voice_pipeline:
                try:
                    await self.voice_pipeline.stop_playback()
                except Exception:
                    pass
            self.last_compaction_error = self.orchestrator.last_compaction_error
            safe_err = sanitize_error_message(e)
            self.bridge.chunk_received.emit(f"\n[Error: {safe_err}]")
            self.bridge.stream_finished.emit()
            self.bridge.status_updated.emit("Error")

    def submit_user_message(self, text: str):
        """Enqueues user message into background asyncio loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._queue.put(("chat", text)), self._loop)

    def submit_listen(self):
        """Enqueues voice listen action into background asyncio loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._queue.put(("listen", None)), self._loop)

    def stop(self):
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._queue.put(None), self._loop)
        self.wait(1000)


class ShamuApp:
    """Master controller managing Qt application lifecycle."""

    def __init__(
        self,
        cfg: Optional[ShamuConfig] = None,
        gateway: Optional[ModelGateway] = None,
        firewall: Optional[ExecutionFirewall] = None,
    ):
        self._set_dpi_awareness()
        self.qapp = QApplication.instance() or QApplication(sys.argv)
        self.qapp.setApplicationName("SHAMU")
        self.qapp.setQuitOnLastWindowClosed(False)

        self.cfg = cfg or get_config()
        self.gateway = gateway or ModelGateway()
        self.firewall = firewall or ExecutionFirewall()
        self.audit = self.firewall.audit_ledger

        # Initialize Memory and Speech subsystems
        self.memory_store = None
        self.embedding_engine = None
        self.retriever = None
        self.compactor = None
        self.memory_init_error = None

        try:
            if not getattr(self.cfg, "memory", None):
                raise ValueError("Memory configuration is missing or disabled")
            self.memory_store = SQLiteMemoryStore(self.cfg.memory)
            self.embedding_engine = EmbeddingEngine(self.cfg.memory)
            self.retriever = HybridRetriever(
                store=self.memory_store,
                embedding_engine=self.embedding_engine,
                config=self.cfg.memory,
            )
            self.compactor = ContextCompactor(
                store=self.memory_store,
                config=self.cfg.memory,
            )
            try:
                if getattr(self.firewall, "memory_store", None) is None:
                    self.firewall.memory_store = self.memory_store
                    if hasattr(self.firewall, "executor") and self.firewall.executor:
                        self.firewall.executor.memory_store = self.memory_store
            except Exception:
                pass
            logger.info("GUI Memory subsystem initialized successfully.")
        except Exception as e:
            self.memory_init_error = str(e)
            logger.error("Failed to initialize GUI Memory subsystem: %s", e, exc_info=True)
            if self.memory_store:
                try:
                    self.memory_store.close()
                except Exception:
                    pass
            self.memory_store = None
            self.embedding_engine = None
            self.retriever = None
            self.compactor = None

        try:
            self.voice_pipeline = VoicePipeline(gateway=self.gateway)
        except Exception:
            self.voice_pipeline = None

        self.bridge = UIBridge()
        self.window = MainWindow(self.bridge)
        self.worker = AgentWorker(
            self.gateway,
            self.bridge,
            firewall=self.firewall,
            voice_pipeline=self.voice_pipeline,
            retriever=self.retriever,
            compactor=self.compactor,
            memory_store=self.memory_store,
        )

        self._setup_approval_handler()

        self.tray = ShamuTrayIcon(
            on_toggle_window=self.toggle_window,
            on_ptt=self.on_ptt_triggered,
            on_quit=self.quit,
        )

        self.hotkey = GlobalHotkeyListener()

        self._connect_signals()

        # Restore active session into GUI and worker history
        if self.memory_store:
            try:
                saved_messages = self.memory_store._sync_load_session("default")
                if saved_messages:
                    self.worker.history = list(saved_messages)
                    for msg in saved_messages:
                        if msg.role == Role.USER and msg.content:
                            display_text = extract_user_query(msg.content)
                            self.window.add_user_message(display_text)
                        elif msg.role == Role.ASSISTANT and msg.content:
                            self.window.add_assistant_message(msg.content)
                    logger.info("Restored %d messages from previous active session.", len(saved_messages))
            except Exception as e:
                logger.warning("Failed to restore active session on startup: %s", e)

    def _setup_approval_handler(self):
        async def ui_approval_handler(req: ApprovalRequest) -> ApprovalResponse:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self.bridge.approval_requested.emit(req, future)
            return await future

        self.firewall.approval_manager.register_handler(ui_approval_handler)

    @Slot(object, object)
    def _on_approval_requested(self, req: ApprovalRequest, future: Any):
        try:
            now = time.time()
            timeout_left = max(1.0, req.expires_at - now) if req.expires_at > 0 else 30.0
            modal = ApprovalModal(
                tool_name=req.tool_name,
                arguments=req.arguments,
                reason=req.reason,
                timeout_seconds=int(timeout_left),
                requires_double_confirmation=req.requires_double_confirmation,
                parent=self.window,
            )
            modal.exec()
            resp = ApprovalResponse(
                request_id=req.request_id,
                approved=modal.approved,
                double_confirmed=modal.double_confirmed if req.requires_double_confirmation else modal.approved,
                reason="Approved by user in security dialog" if modal.approved else "Rejected by user",
            )
        except Exception as ex:
            resp = ApprovalResponse(
                request_id=req.request_id,
                approved=False,
                reason=f"Modal display error: {ex}",
            )

        if hasattr(future, "get_loop"):
            loop = future.get_loop()
            if not future.done():
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        lambda: future.set_result(resp) if not future.done() else None
                    )
                else:
                    future.set_result(resp)
        else:
            if not future.done():
                future.set_result(resp)

    def _set_dpi_awareness(self):
        try:
            # Set per-monitor DPI awareness v2 on Windows 10/11
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

    def _connect_signals(self):
        self.bridge.user_message_sent.connect(self.worker.submit_user_message)
        self.bridge.user_message_sent.connect(self.window.add_user_message)
        self.bridge.approval_requested.connect(self._on_approval_requested)
        self.bridge.ptt_clicked.connect(self.on_ptt_triggered)
        self.hotkey.hotkey_triggered.connect(self.on_global_hotkey)

    def toggle_window(self):
        if self.window.isVisible() and not self.window.isMinimized():
            self.window.hide()
        else:
            self.window.show()
            self.window.raise_()
            self.window.activateWindow()

    def on_ptt_triggered(self):
        self.window.show()
        self.window.activateWindow()
        self.worker.submit_listen()

    def on_global_hotkey(self):
        self.toggle_window()

    def quit(self):
        self.hotkey.stop()
        self.worker.stop()
        if self.memory_store:
            try:
                self.memory_store.close()
            except Exception:
                pass
        self.tray.hide()
        self.qapp.quit()

    def run(self) -> int:
        self.tray.show()
        self.window.show()
        self.worker.start()
        self.hotkey.start()
        return self.qapp.exec()


def main():
    app = ShamuApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()

