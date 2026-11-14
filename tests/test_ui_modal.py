"""Unit tests validating HTML escaping and injection prevention in ApprovalModal (CANON-P2-01)."""

import html
import pytest
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QKeyEvent, QCloseEvent
from PySide6.QtWidgets import QApplication
from shamu.ui.modal import ApprovalModal


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_approval_modal_escapes_script_in_tool_name(qapp):
    """Ensure <script> tags in tool_name are escaped and cannot inject HTML."""
    malicious_tool = "<script>alert('xss-tool')</script>"
    modal = ApprovalModal(
        tool_name=malicious_tool,
        arguments={"param": "value"},
        reason="Test execution",
    )

    label_text = modal.desc_label.text()

    # Raw <script> tag must NOT be present
    assert "<script>" not in label_text
    assert "</script>" not in label_text

    # Escaped sequence must be present
    expected_escaped = html.escape(malicious_tool, quote=True)
    assert f"<b>{expected_escaped}</b>" in label_text
    assert modal.desc_label.textFormat() == Qt.RichText


def test_approval_modal_escapes_img_onerror_in_tool_name(qapp):
    """Ensure event handler tags in tool_name are safely escaped."""
    malicious_tool = '<img src="x" onerror="evil_call()">'
    modal = ApprovalModal(
        tool_name=malicious_tool,
        arguments={},
        reason="Harmless check",
    )

    label_text = modal.desc_label.text()
    assert "<img" not in label_text
    assert "&lt;img" in label_text
    assert html.escape(malicious_tool, quote=True) in label_text


def test_approval_modal_escapes_script_and_iframe_in_reason(qapp):
    """Ensure <script> and <iframe> in reason are strictly escaped."""
    malicious_reason = (
        "<script>window.location='http://evil.com'</script>"
        "<iframe src='http://evil.com/phish'></iframe>"
    )
    modal = ApprovalModal(
        tool_name="powershell_exec",
        arguments={"cmd": "whoami"},
        reason=malicious_reason,
    )

    label_text = modal.desc_label.text()
    assert "<script>" not in label_text
    assert "<iframe" not in label_text
    assert html.escape("<script>", quote=True) in label_text
    assert html.escape("<iframe", quote=True) in label_text


def test_approval_modal_escapes_css_overlay_in_reason(qapp):
    """Ensure CSS overlay styles in reason cannot hijack or spoof modal text."""
    spoof_payload = (
        '<div style="position:fixed;top:0;left:0;width:1000px;height:1000px;'
        'background:white;z-index:99999;">APPROVED BY ADMIN</div>'
    )
    modal = ApprovalModal(
        tool_name="delete_file",
        arguments={"path": "/critical/system"},
        reason=spoof_payload,
    )

    label_text = modal.desc_label.text()
    assert "<div style=" not in label_text
    assert html.escape(spoof_payload, quote=True) in label_text


def test_approval_modal_escapes_javascript_links_in_reason(qapp):
    """Ensure javascript: URI schemes and anchor tags are escaped."""
    malicious_reason = '<a href="javascript:alert(document.cookie)">Click here to verify</a>'
    modal = ApprovalModal(
        tool_name="system_exec",
        arguments={},
        reason=malicious_reason,
    )

    label_text = modal.desc_label.text()
    assert '<a href="javascript:' not in label_text
    assert html.escape(malicious_reason, quote=True) in label_text


def test_approval_modal_escapes_style_tags(qapp):
    """Ensure <style> injection cannot alter modal appearance or hide warnings."""
    malicious_reason = "<style>body { display: none; } label { color: red; }</style>Legit reason"
    modal = ApprovalModal(
        tool_name="powershell_exec",
        arguments={},
        reason=malicious_reason,
    )

    label_text = modal.desc_label.text()
    assert "<style>" not in label_text
    assert "</style>" not in label_text
    assert html.escape("<style>body { display: none; } label { color: red; }</style>", quote=True) in label_text


def test_approval_modal_escapes_html_entities(qapp):
    """Ensure HTML entity double-encoding attempts are safely escaped."""
    entity_payload = "&lt;script&gt;alert('nested')&lt;/script&gt;&amp;quot;"
    modal = ApprovalModal(
        tool_name="cmd_exec",
        arguments={},
        reason=entity_payload,
    )

    label_text = modal.desc_label.text()
    # &lt; should become &amp;lt;
    assert "&amp;lt;script&amp;gt;" in label_text
    assert "&amp;amp;quot;" in label_text


def test_approval_modal_preserves_multiline_reason_safely(qapp):
    """Ensure multiline reasons have linebreaks preserved as <br> while escaping tags."""
    multiline_reason = (
        "Line 1: High risk operation\r\n"
        "Line 2: <b>Bold text attempt</b>\n"
        "Line 3: Target: <critical_node>"
    )
    modal = ApprovalModal(
        tool_name="deploy_agent",
        arguments={},
        reason=multiline_reason,
    )

    label_text = modal.desc_label.text()
    # Should contain <br> for line breaks
    assert "<br>" in label_text
    # Should escape <b> and <critical_node>
    assert "<b>Bold text attempt</b>" not in label_text
    assert "&lt;b&gt;Bold text attempt&lt;/b&gt;" in label_text
    assert "&lt;critical_node&gt;" in label_text
    assert "Line 1: High risk operation<br>Line 2:" in label_text


def test_approval_modal_code_view_plain_text_protection(qapp):
    """Ensure arguments code view does not accept or render rich text."""
    malicious_args = {
        "command": "<script>alert('pwn')</script>",
        "style": "<style>body{color:red}</style>",
        "nested": {"param": "<img src=x onerror=1>"},
    }
    modal = ApprovalModal(
        tool_name="powershell_exec",
        arguments=malicious_args,
        reason="Code view test",
    )

    assert modal.code_view.isReadOnly() is True
    assert modal.code_view.acceptRichText() is False
    plain_content = modal.code_view.toPlainText()
    assert "<script>alert('pwn')</script>" in plain_content
    assert "<style>body{color:red}</style>" in plain_content


def test_approval_modal_static_labels_plain_text(qapp):
    """Ensure headers and timer labels are explicitly configured as plain text."""
    modal = ApprovalModal(
        tool_name="test_tool",
        arguments={},
        reason="Testing label formats",
    )

    assert modal.title_label.textFormat() == Qt.PlainText
    assert modal.args_header_label.textFormat() == Qt.PlainText
    assert modal.timer_label.textFormat() == Qt.PlainText


def test_approval_modal_handles_non_string_types(qapp):
    """Ensure non-string tool_name and reason do not raise exceptions."""
    modal = ApprovalModal(
        tool_name=None,
        arguments={"key": 123},
        reason=404,
    )

    label_text = modal.desc_label.text()
    assert "<b>None</b>" in label_text
    assert "404" in label_text


def test_approval_modal_clean_normal_formatting(qapp):
    """Ensure normal clean inputs retain valid formatting and layout."""
    tool_name = "powershell_exec"
    reason = "Executing system diagnostic script."
    modal = ApprovalModal(
        tool_name=tool_name,
        arguments={"script": "Get-Service"},
        reason=reason,
    )

    label_text = modal.desc_label.text()
    assert f"SHAMU is requesting to execute <b>{tool_name}</b>.<br><span style='color: #adb5bd;'>{reason}</span>" == label_text
    assert modal.approve_btn.isEnabled() is True
    assert modal.deny_btn.isEnabled() is True
    assert modal.timeout_remaining == 30
    assert not modal.approved

    modal._on_approve()
    assert modal.approved is True


def test_approval_modal_double_confirmation_and_denial(qapp):
    """Verify double confirmation and denial mechanisms in UI modal."""
    modal = ApprovalModal(
        tool_name="delete_file",
        arguments={"target": "malicious<script>file.txt"},
        reason="Removal of <untrusted> file",
        requires_double_confirmation=True,
    )

    assert modal.requires_double_confirmation is True
    assert modal.confirm_checkbox is not None
    assert modal.confirm_checkbox.isChecked() is False
    assert modal.approve_btn.isEnabled() is False

    # Attempt approve before checking
    modal._on_approve()
    assert modal.approved is False

    # Check and approve
    modal.confirm_checkbox.setChecked(True)
    assert modal.approve_btn.isEnabled() is True
    modal._on_approve()
    assert modal.approved is True
    assert modal.double_confirmed is True


def test_approval_modal_timer_and_key_events(qapp):
    """Verify timer countdown and rejection on Escape / Enter safety."""
    modal = ApprovalModal(
        tool_name="test_tool",
        arguments={},
        reason="Escape test",
        timeout_seconds=5,
    )
    assert modal.timer.isActive() is True

    # Tick timer
    modal._tick()
    assert modal.timeout_remaining == 4
    assert "4s" in modal.timer_label.text()

    # Enter key without focus does not approve
    enter_ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Return, Qt.NoModifier)
    modal.keyPressEvent(enter_ev)
    assert modal.approved is False

    # Escape key rejects and stops timer
    esc_ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    modal.keyPressEvent(esc_ev)
    assert modal.timer.isActive() is False
    assert modal.approved is False

