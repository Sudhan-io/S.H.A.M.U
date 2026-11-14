import pytest
from PySide6.QtWidgets import QApplication
from shamu.ui.modal import ApprovalModal


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_approval_modal_display(qapp):
    modal = ApprovalModal(
        tool_name="powershell_exec",
        arguments={"command": "Get-Process"},
        reason="High-risk system shell execution.",
        timeout_seconds=30,
    )

    assert "powershell_exec" in modal.code_view.toPlainText() or "Get-Process" in modal.code_view.toPlainText()
    assert modal.timeout_remaining == 30
    assert not modal.approved

    # Simulate user approval
    modal._on_approve()
    assert modal.approved is True


def test_approval_modal_denial(qapp):
    modal = ApprovalModal(
        tool_name="delete_file",
        arguments={"path": "important.txt"},
        reason="Destructive file removal.",
    )

    # Simulate user rejection
    modal._on_deny()
    assert modal.approved is False


def test_approval_modal_requires_double_confirmation(qapp):
    """SEC-05: Double confirmation modal requires explicit checkbox acknowledgement."""
    modal = ApprovalModal(
        tool_name="delete_file",
        arguments={"path": "critical.db"},
        reason="Permanent deletion.",
        requires_double_confirmation=True,
    )
    assert modal.requires_double_confirmation is True
    assert hasattr(modal, "confirm_checkbox")
    assert modal.confirm_checkbox.isChecked() is False
    assert modal.approve_btn.isEnabled() is False
    assert modal.double_confirmed is False

    # Calling _on_approve without ticking checkbox does not grant approval
    modal._on_approve()
    assert modal.approved is False
    assert modal.double_confirmed is False


def test_approval_modal_double_confirmation_success(qapp):
    """SEC-05: Ticking explicit confirmation checkbox enables approval with double_confirmed=True."""
    modal = ApprovalModal(
        tool_name="delete_file",
        arguments={"path": "critical.db"},
        reason="Permanent deletion.",
        requires_double_confirmation=True,
    )
    modal.confirm_checkbox.setChecked(True)
    assert modal.approve_btn.isEnabled() is True

    modal._on_approve()
    assert modal.approved is True
    assert modal.double_confirmed is True


def test_approval_modal_esc_stops_timer(qapp):
    """UI-05: Pressing Esc or rejecting modal halts countdown timer and denies action."""
    from PySide6.QtCore import Qt, QEvent
    from PySide6.QtGui import QKeyEvent

    modal = ApprovalModal(
        tool_name="powershell_exec",
        arguments={"command": "dir"},
        reason="Check files.",
    )
    assert modal.timer.isActive() is True

    # Simulate Escape key press
    esc_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    modal.keyPressEvent(esc_event)

    assert modal.timer.isActive() is False
    assert modal.approved is False


def test_approval_modal_close_stops_timer(qapp):
    """UI-05: Closing modal window stops the countdown timer."""
    from PySide6.QtGui import QCloseEvent

    modal = ApprovalModal(
        tool_name="delete_file",
        arguments={"path": "temp.txt"},
        reason="Delete temp.",
    )
    assert modal.timer.isActive() is True

    close_event = QCloseEvent()
    modal.closeEvent(close_event)

    assert modal.timer.isActive() is False
    assert modal.approved is False


def test_approval_modal_return_does_not_accidental_approve(qapp):
    """UI-06: Accidental Return or Enter does not trigger approve button without user intent."""
    from PySide6.QtCore import Qt, QEvent
    from PySide6.QtGui import QKeyEvent

    modal = ApprovalModal(
        tool_name="powershell_exec",
        arguments={"command": "format C:"},
        reason="Dangerous format.",
    )

    # Approve button must not be default or autoDefault
    assert modal.approve_btn.isDefault() is False
    assert modal.approve_btn.autoDefault() is False

    # Simulate pressing Return without focus on approve button
    return_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Return, Qt.NoModifier)
    modal.keyPressEvent(return_event)
    assert modal.approved is False
    assert modal.timer.isActive() is True

    # Simulate pressing Enter (numpad)
    enter_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Enter, Qt.NoModifier)
    modal.keyPressEvent(enter_event)
    assert modal.approved is False
    assert modal.timer.isActive() is True

    # Clean up
    modal.reject()
    assert modal.timer.isActive() is False



