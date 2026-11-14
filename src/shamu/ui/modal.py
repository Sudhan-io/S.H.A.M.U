"""Interactive Tier 3 Security Approval Modal for dangerous tool confirmations."""

import html
import json
from typing import Any, Dict, Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QCheckBox,
)
from shamu.ui.theme import FLUENT_DARK_THEME


class ApprovalModal(QDialog):
    """Prominent modal dialog requiring explicit user approval for Tier 3 actions."""

    def __init__(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        reason: str,
        timeout_seconds: int = 30,
        requires_double_confirmation: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Security Confirmation Required — SHAMU")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.resize(460, 380)
        self.setStyleSheet(FLUENT_DARK_THEME)

        self.tool_name = tool_name
        self.arguments = arguments
        self.reason = reason
        self.timeout_remaining = timeout_seconds
        self.requires_double_confirmation = requires_double_confirmation
        self.approved = False
        self.double_confirmed = False
        self.confirm_checkbox: Optional[QCheckBox] = None

        self._init_ui(tool_name, arguments, reason)
        self._init_timer()

    def _init_ui(self, tool_name: str, arguments: Dict[str, Any], reason: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # Warning Title (Strictly Plain Text)
        self.title_label = QLabel("⚠️ High-Risk Action Confirmation", self)
        self.title_label.setTextFormat(Qt.PlainText)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffb703;")
        layout.addWidget(self.title_label)

        # Description with strict HTML escaping to prevent rich-text injection
        safe_tool = html.escape(str(tool_name), quote=True)
        escaped_reason = html.escape(str(reason), quote=True)
        safe_reason = escaped_reason.replace("\r\n", "<br>").replace("\n", "<br>")

        self.desc_label = QLabel(
            f"SHAMU is requesting to execute <b>{safe_tool}</b>.<br>"
            f"<span style='color: #adb5bd;'>{safe_reason}</span>",
            self,
        )
        self.desc_label.setWordWrap(True)
        self.desc_label.setTextFormat(Qt.RichText)
        layout.addWidget(self.desc_label)

        # Code review box showing exact parameters (Strictly Plain Text)
        self.args_header_label = QLabel("Proposed Arguments / Script:", self)
        self.args_header_label.setTextFormat(Qt.PlainText)
        layout.addWidget(self.args_header_label)

        self.code_view = QTextEdit(self)
        self.code_view.setReadOnly(True)
        self.code_view.setAcceptRichText(False)
        self.code_view.setStyleSheet("""
            QTextEdit {
                background-color: #141418;
                border: 1px solid #3a3a46;
                border-radius: 6px;
                color: #a8dadc;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                padding: 8px;
            }
        """)
        try:
            formatted_args = json.dumps(arguments, indent=2, default=str)
        except Exception:
            formatted_args = str(arguments)
        self.code_view.setPlainText(formatted_args)
        layout.addWidget(self.code_view, stretch=1)

        # Explicit Double-Confirmation Checkbox for Tier 3 High-Impact Actions
        if self.requires_double_confirmation:
            self.confirm_checkbox = QCheckBox(
                "I acknowledge this is a high-risk action and explicitly confirm execution",
                self,
            )
            self.confirm_checkbox.setStyleSheet("""
                QCheckBox {
                    color: #ffb703;
                    font-size: 12px;
                    font-weight: bold;
                    spacing: 8px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border: 1px solid #ffb703;
                    border-radius: 4px;
                    background-color: #1a1a24;
                }
                QCheckBox::indicator:checked {
                    background-color: #ffb703;
                }
            """)
            self.confirm_checkbox.toggled.connect(self._on_confirm_toggled)
            layout.addWidget(self.confirm_checkbox)

        # Countdown & Action Buttons
        button_row = QHBoxLayout()
        self.timer_label = QLabel(f"Auto-deny in {self.timeout_remaining}s", self)
        self.timer_label.setTextFormat(Qt.PlainText)
        self.timer_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        button_row.addWidget(self.timer_label)

        button_row.addStretch()

        self.deny_btn = QPushButton("Deny Action", self)
        self.deny_btn.setAutoDefault(False)
        self.deny_btn.setDefault(False)
        self.deny_btn.setStyleSheet("""
            QPushButton {
                background-color: #495057;
                color: #ffffff;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover { background-color: #6c757d; }
        """)
        self.deny_btn.clicked.connect(self._on_deny)
        button_row.addWidget(self.deny_btn)

        self.approve_btn = QPushButton("Approve & Execute", self)
        self.approve_btn.setAutoDefault(False)
        self.approve_btn.setDefault(False)
        self.approve_btn.setStyleSheet("""
            QPushButton {
                background-color: #d90429;
                color: #ffffff;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px 18px;
            }
            QPushButton:hover { background-color: #ef233c; }
            QPushButton:disabled {
                background-color: #3a3a46;
                color: #6c757d;
            }
        """)
        self.approve_btn.setEnabled(not self.requires_double_confirmation)
        self.approve_btn.clicked.connect(self._on_approve)
        button_row.addWidget(self.approve_btn)

        layout.addLayout(button_row)

    def _on_confirm_toggled(self, checked: bool):
        self.approve_btn.setEnabled(checked)

    def _init_timer(self):
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _tick(self):
        self.timeout_remaining -= 1
        self.timer_label.setText(f"Auto-deny in {self.timeout_remaining}s")
        if self.timeout_remaining <= 0:
            self._on_deny()

    def _on_approve(self):
        if self.requires_double_confirmation:
            if not (self.confirm_checkbox and self.confirm_checkbox.isChecked()):
                return
            self.double_confirmed = True
        self.approved = True
        self.accept()

    def _on_deny(self):
        self.reject()

    def reject(self):
        """Halt countdown timer and safely reject approval on Esc or explicit cancel."""
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        self.approved = False
        self.double_confirmed = False
        super().reject()

    def accept(self):
        """Halt countdown timer on acceptance."""
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        super().accept()

    def closeEvent(self, event):
        """Ensure timer is stopped when user closes the modal window."""
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        self.approved = False
        self.double_confirmed = False
        super().closeEvent(event)

    def keyPressEvent(self, event):
        """Prevent accidental approval via Return/Enter without user intent."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.approve_btn.hasFocus() and self.approve_btn.isEnabled():
                self._on_approve()
            elif self.deny_btn.hasFocus():
                self._on_deny()
            else:
                event.ignore()
            return
        if event.key() == Qt.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)


