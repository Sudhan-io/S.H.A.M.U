"""SHAMU Native Windows GUI Subsystem."""

from shamu.ui.window import MainWindow, UIBridge
from shamu.ui.tray import ShamuTrayIcon
from shamu.ui.hotkey import GlobalHotkeyListener
from shamu.ui.modal import ApprovalModal
from shamu.ui.theme import FLUENT_DARK_THEME

__all__ = [
    "MainWindow",
    "UIBridge",
    "ShamuTrayIcon",
    "GlobalHotkeyListener",
    "ApprovalModal",
    "FLUENT_DARK_THEME",
]

