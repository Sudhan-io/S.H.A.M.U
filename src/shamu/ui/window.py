"""PySide6 Floating Companion Desktop Window for SHAMU."""

from typing import Optional, Any, Dict, List
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QScrollArea,
    QFrame,
)
from shamu.ui.theme import FLUENT_DARK_THEME


class UIBridge(QObject):
    """Thread-safe signals for updating the GUI from worker threads."""
    user_message_sent = Signal(str)
    ptt_clicked = Signal()
    chunk_received = Signal(str)
    status_updated = Signal(str)
    stream_finished = Signal()
    approval_requested = Signal(object, object)
    turn_reset = Signal()
    clear_partial = Signal()

    # M3 Task Planning & Execution signals with multi-arg and single-arg overloads
    task_plan_created = Signal((str, str, int), (object,), (object, object, object))
    task_step_progress = Signal((str, str, str), (object,), (object, object, object))
    task_plan_completed = Signal((str, bool, str), (object,), (object, object, object))
    task_plan_approval_requested = Signal(object, object)
    task_plan_approved = Signal((str,), (object,))
    task_plan_rejected = Signal((str,), (object,))
    task_plan_report = Signal(object)


class ChatCard(QFrame):
    """Visual speech card for user or assistant messages."""

    def __init__(self, text: str, is_user: bool = False, debounce_ms: int = 50, parent=None):
        super().__init__(parent)
        self.setObjectName("UserCard" if is_user else "AssistantCard")
        self._raw_text = text
        self.debounce_ms = debounce_ms
        self._pending_markdown = False

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_markdown)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)

        self.label = QLabel(self)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.label)

        self._render_markdown()

    def _render_markdown(self):
        self._pending_markdown = False
        if hasattr(self.label, "setMarkdown"):
            self.label.setMarkdown(self._raw_text)
        else:
            self.label.setText(self._raw_text)

    def append_text(self, new_text: str):
        self._raw_text += new_text
        if self.debounce_ms <= 0:
            self._render_markdown()
            return

        # Progressive text update immediately without expensive markdown document re-parsing
        self.label.setText(self._raw_text)
        self._pending_markdown = True

        # Debounce full markdown parsing
        if not self._render_timer.isActive():
            self._render_timer.start(self.debounce_ms)

    def flush(self):
        """Immediately flush and render pending markdown formatting."""
        if hasattr(self, "_render_timer") and self._render_timer.isActive():
            self._render_timer.stop()
        self._render_markdown()

    def clear_text(self):
        """Immediately reset all accumulated and rendered text (e.g. on provider failover)."""
        if hasattr(self, "_render_timer") and self._render_timer.isActive():
            self._render_timer.stop()
        self._raw_text = ""
        self._pending_markdown = False
        if hasattr(self.label, "setMarkdown"):
            self.label.setMarkdown("")
        else:
            self.label.setText("")

    def finalize(self):
        """Finalize card display when streaming finishes."""
        self.flush()


class TaskStepItem(QFrame):
    """Visual display for an individual step in a TaskPlan."""

    def __init__(
        self,
        step_id: str,
        title: str,
        tool_name: str = "",
        description: str = "",
        status: str = "pending",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("TaskStepItem")
        self.step_id = step_id
        self.step_title = title
        self.tool_name = tool_name
        self.description = description
        self.current_status = status

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Header: Step ID, Tool tag, Status Pill
        header = QHBoxLayout()
        header.setSpacing(6)

        self.id_label = QLabel(f"<b>Step {self.step_id}</b>", self)
        self.id_label.setStyleSheet("color: #00b4d8; font-size: 11px;")
        header.addWidget(self.id_label)

        if self.tool_name:
            self.tool_label = QLabel(f"<code>[{self.tool_name}]</code>", self)
            self.tool_label.setStyleSheet("color: #a8dadc; font-size: 11px;")
            header.addWidget(self.tool_label)

        header.addStretch()

        self.status_pill = QLabel(self)
        self.status_pill.setStyleSheet("font-size: 11px; font-weight: bold; padding: 2px 6px; border-radius: 4px;")
        header.addWidget(self.status_pill)
        layout.addLayout(header)

        # Title
        self.title_label = QLabel(self.step_title, self)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-weight: 500; color: #ffffff;")
        layout.addWidget(self.title_label)

        # Description (if any)
        if self.description:
            self.desc_label = QLabel(self.description, self)
            self.desc_label.setWordWrap(True)
            self.desc_label.setStyleSheet("font-size: 11px; color: #adb5bd;")
            layout.addWidget(self.desc_label)

        # Details / Result / Error container
        self.detail_label = QLabel(self)
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail_label.setVisible(False)
        layout.addWidget(self.detail_label)

        self.set_status(self.current_status)

    def set_status(self, status: str, detail: str = ""):
        self.current_status = status.lower()
        badge_styles = {
            "pending": ("⏳ Pending", "color: #adb5bd; background-color: #2b2b36;"),
            "in_progress": ("⚙️ Running", "color: #00b4d8; background-color: #1a3a4a;"),
            "running": ("⚙️ Running", "color: #00b4d8; background-color: #1a3a4a;"),
            "waiting_approval": ("⚠️ Approval", "color: #ffb703; background-color: #4a3810;"),
            "completed": ("✓ Done", "color: #52b788; background-color: #183d29;"),
            "failed": ("✗ Failed", "color: #ef233c; background-color: #4a1820;"),
            "skipped": ("⏭️ Skipped", "color: #6c757d; background-color: #242428;"),
        }

        text, style = badge_styles.get(
            self.current_status,
            (status.upper(), "color: #adb5bd; background-color: #2b2b36;")
        )
        self.status_pill.setText(text)
        self.status_pill.setStyleSheet(f"font-size: 11px; font-weight: bold; padding: 2px 6px; border-radius: 4px; {style}")

        if detail and detail.strip():
            self.detail_label.setVisible(True)
            if self.current_status == "failed":
                self.detail_label.setStyleSheet(
                    "color: #ef233c; font-size: 11px; background-color: #2d181c; padding: 4px 6px; border-radius: 4px;"
                )
                self.detail_label.setText(f"Error: {detail.strip()}")
            elif self.current_status == "completed":
                self.detail_label.setStyleSheet(
                    "color: #90e0ef; font-size: 11px; background-color: #16242c; padding: 4px 6px; border-radius: 4px;"
                )
                first_line = detail.strip().splitlines()[0]
                if len(first_line) > 100:
                    first_line = first_line[:97] + "..."
                self.detail_label.setText(first_line)
            else:
                self.detail_label.setStyleSheet("color: #ced4da; font-size: 11px;")
                self.detail_label.setText(detail.strip())


class TaskPlanCard(QFrame):
    """Visual presentation card for TaskPlan presentation, approvals, and progress updates."""

    def __init__(
        self,
        plan_id: str,
        objective: str,
        total_steps: int = 0,
        bridge: Optional[UIBridge] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("TaskPlanCard")
        self.plan_id = plan_id
        self.objective = objective
        self.total_steps = total_steps
        self.bridge = bridge
        self.approval_future: Optional[Any] = None
        self.step_items: Dict[str, TaskStepItem] = {}

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Header: Title and Status Badge
        header = QHBoxLayout()
        title_label = QLabel("📋 <b>Task Plan</b>", self)
        title_label.setStyleSheet("color: #00b4d8; font-size: 13px;")
        header.addWidget(title_label)

        header.addStretch()

        self.status_badge = QLabel("PENDING", self)
        self.status_badge.setStyleSheet(
            "color: #ffb703; background-color: #3d3010; font-size: 11px; font-weight: bold; padding: 3px 8px; border-radius: 4px;"
        )
        header.addWidget(self.status_badge)
        layout.addLayout(header)

        # Objective
        obj_label = QLabel(self.objective, self)
        obj_label.setWordWrap(True)
        obj_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #ffffff;")
        layout.addWidget(obj_label)

        # Metadata
        self.meta_label = QLabel(f"Plan ID: {self.plan_id} • Steps: {self.total_steps}", self)
        self.meta_label.setStyleSheet("font-size: 11px; color: #8d99ae;")
        layout.addWidget(self.meta_label)

        # Separator line
        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #2f2f3d; background-color: #2f2f3d; height: 1px;")
        layout.addWidget(sep)

        # Steps Container
        self.steps_container = QWidget(self)
        self.steps_layout = QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(0, 2, 0, 2)
        self.steps_layout.setSpacing(4)
        layout.addWidget(self.steps_container)

        # Approval Buttons Row
        self.approval_widget = QWidget(self)
        approval_row = QHBoxLayout(self.approval_widget)
        approval_row.setContentsMargins(0, 4, 0, 0)
        approval_row.setSpacing(8)

        self.approval_prompt = QLabel("Approval required to execute plan:", self)
        self.approval_prompt.setStyleSheet("font-size: 11px; color: #ced4da;")
        approval_row.addWidget(self.approval_prompt)

        approval_row.addStretch()

        self.reject_btn = QPushButton("✕ Reject", self)
        self.reject_btn.setStyleSheet("""
            QPushButton {
                background-color: #495057;
                color: #ffffff;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #6c757d; }
            QPushButton:disabled { background-color: #2b2b36; color: #6c757d; }
        """)
        self.reject_btn.clicked.connect(self._on_reject)
        approval_row.addWidget(self.reject_btn)

        self.approve_btn = QPushButton("✓ Approve Plan", self)
        self.approve_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a9d8f;
                color: #ffffff;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #38b000; }
            QPushButton:disabled { background-color: #2b2b36; color: #6c757d; }
        """)
        self.approve_btn.clicked.connect(self._on_approve)
        approval_row.addWidget(self.approve_btn)

        layout.addWidget(self.approval_widget)

        # Final Summary Label
        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-size: 12px; padding: 6px; border-radius: 6px;")
        self.summary_label.setVisible(False)
        layout.addWidget(self.summary_label)

    def populate_from_plan(self, plan: Any):
        """Populates steps from a TaskPlan model instance."""
        if hasattr(plan, "steps"):
            self.total_steps = len(plan.steps)
            self.meta_label.setText(f"Plan ID: {self.plan_id} • Steps: {self.total_steps}")
            for step in plan.steps:
                tool = getattr(step, "tool_name", "") or ""
                desc = getattr(step, "description", "") or ""
                status = getattr(step, "status", "pending")
                status_str = status.value if hasattr(status, "value") else str(status)
                self.add_or_update_step(step.id, step.title, tool_name=tool, description=desc, status=status_str)

    def add_or_update_step(
        self,
        step_id: str,
        title: str = "",
        tool_name: str = "",
        description: str = "",
        status: str = "pending",
        detail: str = "",
    ):
        if step_id in self.step_items:
            item = self.step_items[step_id]
            item.set_status(status, detail)
        else:
            item = TaskStepItem(
                step_id,
                title or f"Step {step_id}",
                tool_name=tool_name,
                description=description,
                status=status,
                parent=self,
            )
            if detail:
                item.set_status(status, detail)
            self.step_items[step_id] = item
            self.steps_layout.addWidget(item)

    def update_step_status(self, step_id: str, status: str, detail: str = ""):
        if step_id == "plan":
            return
        if step_id in self.step_items:
            self.step_items[step_id].set_status(status, detail)
        else:
            self.add_or_update_step(step_id, f"Step {step_id}", status=status, detail=detail)

    def set_approval_future(self, future: Any):
        self.approval_future = future
        self.approval_widget.setVisible(True)
        self.approve_btn.setEnabled(True)
        self.reject_btn.setEnabled(True)
        self.status_badge.setText("WAITING APPROVAL")
        self.status_badge.setStyleSheet(
            "color: #ffb703; background-color: #3d3010; font-size: 11px; font-weight: bold; padding: 3px 8px; border-radius: 4px;"
        )

    def _resolve_future(self, result: bool):
        fut = self.approval_future
        if fut is not None:
            if hasattr(fut, "get_loop"):
                loop = fut.get_loop()
                if not fut.done():
                    if loop.is_running():
                        loop.call_soon_threadsafe(lambda: fut.set_result(result) if not fut.done() else None)
                    else:
                        fut.set_result(result)
            elif hasattr(fut, "done") and not fut.done():
                fut.set_result(result)

    def _on_approve(self):
        self.approve_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)
        self.approval_prompt.setText("✓ Plan approved by user.")
        self.status_badge.setText("APPROVED")
        self.status_badge.setStyleSheet(
            "color: #52b788; background-color: #183d29; font-size: 11px; font-weight: bold; padding: 3px 8px; border-radius: 4px;"
        )
        if self.bridge and hasattr(self.bridge, "task_plan_approved"):
            try:
                self.bridge.task_plan_approved.emit(self.plan_id)
            except Exception:
                pass
        self._resolve_future(True)

    def _on_reject(self):
        self.approve_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)
        self.approval_prompt.setText("✕ Plan rejected by user.")
        self.status_badge.setText("CANCELLED")
        self.status_badge.setStyleSheet(
            "color: #ef233c; background-color: #4a1820; font-size: 11px; font-weight: bold; padding: 3px 8px; border-radius: 4px;"
        )
        if self.bridge and hasattr(self.bridge, "task_plan_rejected"):
            try:
                self.bridge.task_plan_rejected.emit(self.plan_id)
            except Exception:
                pass
        self._resolve_future(False)

    def mark_completed(self, success: bool, summary: str = ""):
        self.approval_widget.setVisible(False)
        if success:
            self.status_badge.setText("COMPLETED")
            self.status_badge.setStyleSheet(
                "color: #52b788; background-color: #183d29; font-size: 11px; font-weight: bold; padding: 3px 8px; border-radius: 4px;"
            )
            if summary:
                self.summary_label.setText(f"<b>Summary:</b> {summary}")
                self.summary_label.setStyleSheet(
                    "color: #d8f3dc; background-color: #162a22; font-size: 12px; padding: 6px; border-radius: 6px;"
                )
                self.summary_label.setVisible(True)
        else:
            self.status_badge.setText("FAILED")
            self.status_badge.setStyleSheet(
                "color: #ef233c; background-color: #4a1820; font-size: 11px; font-weight: bold; padding: 3px 8px; border-radius: 4px;"
            )
            if summary:
                self.summary_label.setText(f"<b>Failure:</b> {summary}")
                self.summary_label.setStyleSheet(
                    "color: #ffccd5; background-color: #381a1f; font-size: 12px; padding: 6px; border-radius: 6px;"
                )
                self.summary_label.setVisible(True)


class TaskReportCard(QFrame):
    """Visual presentation card for the final TaskExecutionReport."""

    def __init__(self, report: Any, parent=None):
        super().__init__(parent)
        self.setObjectName("TaskReportCard")
        self.report = report

        # Extract fields
        if hasattr(report, "objective"):
            self.objective = report.objective
            self.plan_id = report.plan_id
            status = report.status
            self.status_str = status.value.upper() if hasattr(status, "value") else str(status).upper()
            self.is_success = (self.status_str == "COMPLETED")
            self.completed_steps = report.completed_steps
            self.total_steps = report.total_steps
            self.failed_steps = getattr(report, "failed_steps", 0)
            self.duration = getattr(report, "duration_seconds", 0.0)
            self.final_output = getattr(report, "final_output", "")
            self.step_summaries = getattr(report, "step_summaries", [])
        elif isinstance(report, dict):
            self.objective = report.get("objective", "Task Execution")
            self.plan_id = report.get("plan_id", "N/A")
            self.status_str = str(report.get("status", "COMPLETED")).upper()
            self.is_success = (self.status_str == "COMPLETED")
            self.completed_steps = report.get("completed_steps", 0)
            self.total_steps = report.get("total_steps", 0)
            self.failed_steps = report.get("failed_steps", 0)
            self.duration = float(report.get("duration_seconds", 0.0))
            self.final_output = report.get("final_output", "")
            self.step_summaries = report.get("step_summaries", [])
        else:
            self.objective = str(report)
            self.plan_id = "N/A"
            self.status_str = "COMPLETED"
            self.is_success = True
            self.completed_steps = 0
            self.total_steps = 0
            self.failed_steps = 0
            self.duration = 0.0
            self.final_output = ""
            self.step_summaries = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Border color based on success
        border_col = "#2a9d8f" if self.is_success else "#ef233c"
        self.setStyleSheet(f"QFrame#TaskReportCard {{ border: 1px solid {border_col}; }}")

        # Header
        header = QHBoxLayout()
        icon = "📊" if self.is_success else "⚠️"
        title_label = QLabel(f"{icon} <b>Task Execution Report</b>", self)
        title_label.setStyleSheet(f"color: {'#52b788' if self.is_success else '#ef233c'}; font-size: 13px;")
        header.addWidget(title_label)

        header.addStretch()

        badge = QLabel(self.status_str, self)
        badge_style = (
            "color: #52b788; background-color: #183d29;"
            if self.is_success
            else "color: #ef233c; background-color: #4a1820;"
        )
        badge.setStyleSheet(f"font-size: 11px; font-weight: bold; padding: 3px 8px; border-radius: 4px; {badge_style}")
        header.addWidget(badge)
        layout.addLayout(header)

        # Objective
        obj_label = QLabel(f"<b>Objective:</b> {self.objective}", self)
        obj_label.setWordWrap(True)
        layout.addWidget(obj_label)

        # Metrics bar
        metrics = QLabel(
            f"<b>Plan ID:</b> {self.plan_id}  |  "
            f"<b>Steps:</b> {self.completed_steps}/{self.total_steps} (Failed: {self.failed_steps})  |  "
            f"<b>Duration:</b> {self.duration:.2f}s",
            self,
        )
        metrics.setStyleSheet("font-size: 11px; color: #a8dadc; background-color: #20242e; padding: 4px 8px; border-radius: 4px;")
        layout.addWidget(metrics)

        # Final Output / Summary
        if self.final_output:
            out_header = QLabel("<b>Final Summary / Output:</b>", self)
            out_header.setStyleSheet("font-size: 11px; color: #ced4da; margin-top: 4px;")
            layout.addWidget(out_header)

            out_box = QLabel(self.final_output, self)
            out_box.setWordWrap(True)
            out_box.setTextInteractionFlags(Qt.TextSelectableByMouse)
            out_box.setStyleSheet(
                "background-color: #161820; border: 1px solid #2e323e; border-radius: 6px; padding: 8px; font-size: 12px; color: #e0e0f0;"
            )
            layout.addWidget(out_box)

        # Step breakdown
        if self.step_summaries:
            breakdown_title = QLabel("<b>Step Breakdown:</b>", self)
            breakdown_title.setStyleSheet("font-size: 11px; color: #ced4da; margin-top: 4px;")
            layout.addWidget(breakdown_title)

            for step_info in self.step_summaries:
                s_id = step_info.get("id", "")
                s_title = step_info.get("title", "")
                s_tool = step_info.get("tool", "")
                s_status = step_info.get("status", "")
                s_ver = step_info.get("verified")
                s_err = step_info.get("error", "")

                row = QHBoxLayout()
                row.setSpacing(6)

                st_tag = QLabel(f"#{s_id}", self)
                st_tag.setStyleSheet("color: #00b4d8; font-weight: bold; font-size: 11px;")
                row.addWidget(st_tag)

                st_title = QLabel(s_title, self)
                st_title.setWordWrap(True)
                st_title.setStyleSheet("color: #ffffff; font-size: 11px;")
                row.addWidget(st_title, stretch=1)

                if s_tool:
                    st_tool = QLabel(f"[{s_tool}]", self)
                    st_tool.setStyleSheet("color: #a8dadc; font-size: 10px;")
                    row.addWidget(st_tool)

                if s_ver is True:
                    ver_badge = QLabel("✓ Verified", self)
                    ver_badge.setStyleSheet("color: #52b788; font-size: 10px;")
                    row.addWidget(ver_badge)
                elif s_ver is False:
                    ver_badge = QLabel("✗ Unverified", self)
                    ver_badge.setStyleSheet("color: #ffb703; font-size: 10px;")
                    row.addWidget(ver_badge)

                stat_badge = QLabel(s_status.upper(), self)
                color = "#52b788" if s_status == "completed" else ("#ef233c" if s_status == "failed" else "#6c757d")
                stat_badge.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 10px;")
                row.addWidget(stat_badge)

                layout.addLayout(row)

                if s_err:
                    err_lbl = QLabel(f"↳ Error: {s_err}", self)
                    err_lbl.setWordWrap(True)
                    err_lbl.setStyleSheet("color: #ef233c; font-size: 10px; margin-left: 16px;")
                    layout.addWidget(err_lbl)


class MainWindow(QMainWindow):
    """Floating companion window for SHAMU desktop interactions."""

    def __init__(self, bridge: Optional[UIBridge] = None):
        super().__init__()
        self.bridge = bridge or UIBridge()
        self.setWindowTitle("SHAMU Desktop Assistant")
        self.resize(440, 640)
        self.setStyleSheet(FLUENT_DARK_THEME)

        self.current_assistant_card: Optional[ChatCard] = None
        self.current_task_card: Optional[TaskPlanCard] = None
        self.task_cards: Dict[str, TaskPlanCard] = {}
        self._auto_scroll_enabled = True

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        # Header Bar
        header = QHBoxLayout()
        title_label = QLabel("<b>SHAMU</b>", self)
        title_label.setStyleSheet("font-size: 15px; color: #00b4d8;")
        header.addWidget(title_label)

        header.addStretch()

        self.status_pill = QLabel("● Ready", self)
        self.status_pill.setObjectName("StatusPill")
        header.addWidget(self.status_pill)

        self.pin_btn = QPushButton("📌", self)
        self.pin_btn.setCheckable(True)
        self.pin_btn.setFixedSize(30, 26)
        self.pin_btn.clicked.connect(self._toggle_always_on_top)
        header.addWidget(self.pin_btn)

        main_layout.addLayout(header)

        # Chat History Scroll Area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_layout.setSpacing(8)
        self.chat_layout.addStretch()
        self.scroll_area.setWidget(self.chat_container)
        vsb = self.scroll_area.verticalScrollBar()
        vsb.rangeChanged.connect(self._on_scroll_range_changed)
        vsb.valueChanged.connect(self._on_scroll_value_changed)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # Bottom Input Row
        input_row = QHBoxLayout()
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Type a message or /plan <objective>...")
        self.input_field.returnPressed.connect(self._on_send)
        input_row.addWidget(self.input_field, stretch=1)

        self.ptt_btn = QPushButton("🎤", self)
        self.ptt_btn.setObjectName("PTTButton")
        self.ptt_btn.setFixedSize(36, 36)
        self.ptt_btn.setToolTip("Push to Talk")
        self.ptt_btn.clicked.connect(self._on_ptt)
        input_row.addWidget(self.ptt_btn)

        self.send_btn = QPushButton("Send", self)
        self.send_btn.setFixedHeight(36)
        self.send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.send_btn)

        main_layout.addLayout(input_row)

    def _connect_signals(self):
        self.bridge.chunk_received.connect(self._on_chunk)
        self.bridge.status_updated.connect(self._on_status)
        self.bridge.stream_finished.connect(self._on_stream_finished)
        self.bridge.turn_reset.connect(self._on_turn_reset)
        self.bridge.clear_partial.connect(self.clear_partial)

        # M3 Task Planning & Execution signal connections
        self.bridge.task_plan_created.connect(self._on_task_plan_created)
        try:
            self.bridge.task_plan_created[object].connect(self._on_task_plan_created)
        except Exception:
            pass

        self.bridge.task_step_progress.connect(self._on_task_step_progress)
        try:
            self.bridge.task_step_progress[object].connect(self._on_task_step_progress)
        except Exception:
            pass

        self.bridge.task_plan_completed.connect(self._on_task_plan_completed)
        try:
            self.bridge.task_plan_completed[object].connect(self._on_task_plan_completed)
        except Exception:
            pass

        self.bridge.task_plan_approval_requested.connect(self._on_task_plan_approval_requested)
        self.bridge.task_plan_report.connect(self.display_task_report)

    def _toggle_always_on_top(self, checked: bool):
        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def _on_send(self):
        text = self.input_field.text().strip()
        if not text:
            return
        self.input_field.clear()
        self.bridge.user_message_sent.emit(text)

    def _on_ptt(self):
        self.bridge.ptt_clicked.emit()

    def add_user_message(self, text: str):
        if text.strip() == "/clear":
            self.clear_messages()
            return
        card = ChatCard(text, is_user=True, parent=self.chat_container)
        # Insert before bottom stretch
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, card)
        self._scroll_to_bottom()

    def add_assistant_message(self, text: str):
        card = ChatCard(text, is_user=False, parent=self.chat_container)
        # Insert before bottom stretch
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, card)
        self._scroll_to_bottom()

    def clear_messages(self):
        """Removes all chat cards from the conversation view."""
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.current_assistant_card = None
        self.current_task_card = None
        self.task_cards.clear()

    def start_assistant_turn(self):
        self.current_assistant_card = ChatCard("", is_user=False, parent=self.chat_container)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, self.current_assistant_card)
        self._scroll_to_bottom()

    def _on_chunk(self, delta: str):
        if self.current_assistant_card is None:
            self.start_assistant_turn()
        self.current_assistant_card.append_text(delta)
        self._scroll_to_bottom()

    def _on_status(self, status_text: str):
        self.status_pill.setText(f"● {status_text}")

    def _on_stream_finished(self):
        if self.current_assistant_card is not None:
            self.current_assistant_card.finalize()
        self.current_assistant_card = None

    def clear_partial(self):
        """Clears any partial tokens rendered in the current assistant card."""
        if self.current_assistant_card is not None:
            self.current_assistant_card.clear_text()

    def _on_turn_reset(self):
        """Handles mid-stream failover turn reset by clearing partial text."""
        self.clear_partial()

    # =========================================================================
    # Task Planning & Execution GUI Methods
    # =========================================================================

    def add_task_plan(
        self,
        plan_or_id: Any,
        objective: str = "",
        total_steps: int = 0,
    ) -> TaskPlanCard:
        """Presents a TaskPlanCard in the chat layout."""
        if hasattr(plan_or_id, "id") and hasattr(plan_or_id, "objective"):
            # TaskPlan object
            plan_id = plan_or_id.id
            obj = plan_or_id.objective
            steps = len(getattr(plan_or_id, "steps", []))
            card = TaskPlanCard(plan_id, obj, total_steps=steps, bridge=self.bridge, parent=self.chat_container)
            card.populate_from_plan(plan_or_id)
        else:
            plan_id = str(plan_or_id)
            obj = objective or f"Plan {plan_id}"
            card = TaskPlanCard(plan_id, obj, total_steps=total_steps, bridge=self.bridge, parent=self.chat_container)

        self.current_task_card = card
        self.task_cards[plan_id] = card
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, card)
        self._scroll_to_bottom()
        return card

    def _on_task_plan_created(self, *args):
        """Slot for task_plan_created signal."""
        if len(args) == 3:
            plan_id, objective, total_steps = args
            self.add_task_plan(plan_id, objective=objective, total_steps=total_steps)
        elif len(args) == 1:
            arg = args[0]
            if hasattr(arg, "id") and hasattr(arg, "objective"):
                self.add_task_plan(arg)
            elif hasattr(arg, "plan_id") and hasattr(arg, "objective"):
                self.add_task_plan(arg.plan_id, objective=arg.objective, total_steps=getattr(arg, "total_steps", 0))
            elif isinstance(arg, dict):
                self.add_task_plan(arg.get("id", "plan"), objective=arg.get("objective", ""), total_steps=arg.get("total_steps", 0))
            else:
                self.add_task_plan(str(arg))

    def _on_task_step_progress(self, *args):
        """Slot for task_step_progress signal."""
        step_id = ""
        status = "in_progress"
        detail = ""

        if len(args) == 3:
            step_id, status, detail = args
        elif len(args) == 1:
            ev = args[0]
            if hasattr(ev, "step_id"):
                step_id = ev.step_id
                event_name = type(ev).__name__
                if "Completed" in event_name:
                    status = "completed"
                    detail = getattr(ev, "result", "")
                elif "Failed" in event_name:
                    status = "failed"
                    detail = getattr(ev, "error", "")
                elif "WaitingApproval" in event_name:
                    status = "waiting_approval"
                    detail = getattr(ev, "tool_name", "")
                else:
                    status = "in_progress"
                    detail = getattr(ev, "step_title", "")
            elif isinstance(ev, dict):
                step_id = ev.get("step_id", "")
                status = ev.get("status", "in_progress")
                detail = ev.get("detail", "")

        if self.current_task_card:
            self.current_task_card.update_step_status(str(step_id), str(status), str(detail))
            self._scroll_to_bottom()

        if step_id and step_id != "plan":
            self._on_status(f"Step {step_id}: {status}")

    def _on_task_plan_completed(self, *args):
        """Slot for task_plan_completed signal."""
        plan_id = ""
        success = True
        summary = ""

        if len(args) == 3:
            plan_id, success, summary = args
        elif len(args) == 1:
            ev = args[0]
            if hasattr(ev, "plan_id") and hasattr(ev, "success"):
                plan_id = ev.plan_id
                success = ev.success
                summary = getattr(ev, "summary", "")
            elif hasattr(ev, "status"):
                # Possibly a TaskExecutionReport
                success = (str(ev.status).lower() in ("completed", "taskstatus.completed"))
                summary = getattr(ev, "final_output", "")
                self.display_task_report(ev)
                return

        card = self.task_cards.get(plan_id, self.current_task_card)
        if card:
            card.mark_completed(success, summary)
            self._scroll_to_bottom()

        self._on_status("Plan completed" if success else "Plan failed")

    def _on_task_plan_approval_requested(self, plan: Any, future: Any):
        """Slot when a task plan needs explicit user approval."""
        plan_id = getattr(plan, "id", None) or (plan if isinstance(plan, str) else "plan")
        card = self.task_cards.get(plan_id, self.current_task_card)
        if card is None:
            card = self.add_task_plan(plan)
        card.set_approval_future(future)
        self._scroll_to_bottom()

    def display_task_report(self, report: Any) -> TaskReportCard:
        """Presents a TaskReportCard in the chat layout."""
        card = TaskReportCard(report, parent=self.chat_container)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, card)
        self._scroll_to_bottom()
        return card

    def approve_current_task_plan(self):
        """Convenience helper to approve the active plan programmatically."""
        if self.current_task_card:
            self.current_task_card._on_approve()

    def reject_current_task_plan(self):
        """Convenience helper to reject the active plan programmatically."""
        if self.current_task_card:
            self.current_task_card._on_reject()

    # =========================================================================

    def connect_hotkey(self, hotkey_listener: Any):
        """Connect hotkey listener to notify the user of registration errors."""
        if hasattr(hotkey_listener, "registration_failed"):
            hotkey_listener.registration_failed.connect(self._on_hotkey_registration_failed)

    def _on_hotkey_registration_failed(self, error_msg: str):
        self._on_status("Hotkey Conflict")

    def _scroll_to_bottom(self):
        if self._auto_scroll_enabled:
            vsb = self.scroll_area.verticalScrollBar()
            vsb.setValue(vsb.maximum())

    def _on_scroll_range_changed(self, min_val: int, max_val: int):
        """Smoothly track new content when auto-scroll is active."""
        if self._auto_scroll_enabled:
            self.scroll_area.verticalScrollBar().setValue(max_val)

    def _on_scroll_value_changed(self, value: int):
        """Pause auto-scroll if user manually scrolls up; resume when scrolled back to bottom."""
        vsb = self.scroll_area.verticalScrollBar()
        self._auto_scroll_enabled = (value >= vsb.maximum() - 25)
