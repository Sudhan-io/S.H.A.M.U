import pytest
from unittest.mock import MagicMock
from PySide6.QtWidgets import QApplication, QSystemTrayIcon
from PySide6.QtGui import QIcon

from shamu.ui.tray import ShamuTrayIcon, create_default_icon


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_create_default_icon(qapp):
    icon = create_default_icon()
    assert isinstance(icon, QIcon)
    assert not icon.isNull()
    pixmap = icon.pixmap(32, 32)
    assert not pixmap.isNull()
    assert pixmap.width() == 32
    assert pixmap.height() == 32


def test_shamu_tray_icon_init_and_properties(qapp):
    mock_toggle = MagicMock()
    mock_ptt = MagicMock()
    mock_quit = MagicMock()

    tray = ShamuTrayIcon(
        on_toggle_window=mock_toggle,
        on_ptt=mock_ptt,
        on_quit=mock_quit,
    )

    assert tray.toolTip() == "SHAMU Desktop Assistant (Active)"
    assert tray.contextMenu() is not None

    actions = tray.contextMenu().actions()
    action_texts = [a.text() for a in actions if not a.isSeparator()]
    assert "Show SHAMU" in action_texts
    assert "Push to Talk (Mic)" in action_texts
    assert "Quit SHAMU" in action_texts


def test_shamu_tray_menu_actions_trigger_callbacks(qapp):
    mock_toggle = MagicMock()
    mock_ptt = MagicMock()
    mock_quit = MagicMock()

    tray = ShamuTrayIcon(
        on_toggle_window=mock_toggle,
        on_ptt=mock_ptt,
        on_quit=mock_quit,
    )

    actions = {a.text(): a for a in tray.contextMenu().actions() if not a.isSeparator()}

    actions["Show SHAMU"].trigger()
    mock_toggle.assert_called_once()

    actions["Push to Talk (Mic)"].trigger()
    mock_ptt.assert_called_once()

    actions["Quit SHAMU"].trigger()
    mock_quit.assert_called_once()


def test_shamu_tray_activation_signals(qapp):
    mock_toggle = MagicMock()
    mock_ptt = MagicMock()
    mock_quit = MagicMock()

    tray = ShamuTrayIcon(
        on_toggle_window=mock_toggle,
        on_ptt=mock_ptt,
        on_quit=mock_quit,
    )

    # Trigger (single click on Windows tray)
    tray._on_activated(QSystemTrayIcon.Trigger)
    assert mock_toggle.call_count == 1

    # DoubleClick
    tray._on_activated(QSystemTrayIcon.DoubleClick)
    assert mock_toggle.call_count == 2

    # MiddleClick or other reasons should not toggle
    tray._on_activated(QSystemTrayIcon.MiddleClick)
    tray._on_activated(QSystemTrayIcon.Context)
    assert mock_toggle.call_count == 2
