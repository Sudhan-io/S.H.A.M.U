import pytest
from PySide6.QtCore import QObject
from shamu.ui.hotkey import GlobalHotkeyListener, MOD_CONTROL, MOD_SHIFT, VK_SPACE


def test_hotkey_listener_initialization():
    listener = GlobalHotkeyListener(
        hotkey_id=1,
        modifiers=(MOD_CONTROL | MOD_SHIFT),
        vk_code=VK_SPACE,
    )
    assert listener.hotkey_id == 1
    assert listener.modifiers == (MOD_CONTROL | MOD_SHIFT)
    assert listener.vk_code == VK_SPACE
    assert hasattr(listener, "hotkey_triggered")
    assert hasattr(listener, "registration_failed")


def test_hotkey_registration_failure(monkeypatch):
    """UX-03: When RegisterHotKey fails, registration_failed signal is emitted with error description."""
    import ctypes
    from shamu.ui import hotkey

    # Mock RegisterHotKey returning 0 (failure)
    class MockUser32:
        def RegisterHotKey(self, *args):
            return 0
        def UnregisterHotKey(self, *args):
            pass

    monkeypatch.setattr(hotkey, "user32", MockUser32())

    listener = GlobalHotkeyListener()
    failed_messages = []
    listener.registration_failed.connect(lambda msg: failed_messages.append(msg))

    listener.run()

    assert len(failed_messages) == 1
    assert "failed" in failed_messages[0].lower()


def test_hotkey_captures_last_error_with_use_last_error(monkeypatch):
    """UI-08: Captures Win32 error code using ctypes.get_last_error()."""
    import ctypes
    from shamu.ui import hotkey

    class FailingUser32:
        def RegisterHotKey(self, *args):
            ctypes.set_last_error(1409)  # ERROR_HOTKEY_ALREADY_REGISTERED
            return 0
        def UnregisterHotKey(self, *args):
            pass

    monkeypatch.setattr(hotkey, "user32", FailingUser32())

    listener = GlobalHotkeyListener()
    emitted = []
    listener.registration_failed.connect(lambda msg: emitted.append(msg))

    listener.run()

    assert listener.last_error == 1409
    assert len(emitted) == 1
    assert "1409" in emitted[0]
    assert listener.last_error_message == emitted[0]


def test_hotkey_registration_failure_notifies_parent(monkeypatch):
    """UI-09: Registration failure signal notifies user via parent status update."""
    import ctypes
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QLabel, QApplication
    from shamu.ui import hotkey

    app = QApplication.instance() or QApplication([])

    class MockParent(QObject):
        def __init__(self):
            super().__init__()
            self.status_pill = QLabel("● Ready")

    class FailingUser32:
        def RegisterHotKey(self, *args):
            ctypes.set_last_error(5)  # ERROR_ACCESS_DENIED
            return 0
        def UnregisterHotKey(self, *args):
            pass

    monkeypatch.setattr(hotkey, "user32", FailingUser32())

    parent = MockParent()
    listener = GlobalHotkeyListener(parent=parent)
    listener.run()

    assert "Hotkey Conflict" in parent.status_pill.text()
    assert listener.last_error == 5



