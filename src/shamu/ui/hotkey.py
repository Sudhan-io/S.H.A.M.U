"""Native Windows Global Hotkey Listener using Win32 RegisterHotKey."""

import ctypes
from ctypes import wintypes
import logging
from typing import Optional
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
VK_SPACE = 0x20

try:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
except Exception:
    user32 = getattr(ctypes, "windll", None).user32 if hasattr(ctypes, "windll") else None

if user32 is not None and hasattr(user32, "RegisterHotKey"):
    try:
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
        user32.RegisterHotKey.restype = wintypes.BOOL
        user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.UnregisterHotKey.restype = wintypes.BOOL
    except Exception:
        pass


class GlobalHotkeyListener(QThread):
    """Listens for global Windows hotkey triggers even when window is unfocused."""
    hotkey_triggered = Signal()
    registration_failed = Signal(str)

    def __init__(
        self,
        hotkey_id: int = 1,
        modifiers: int = (MOD_CONTROL | MOD_SHIFT),
        vk_code: int = VK_SPACE,
        parent=None,
    ):
        super().__init__(parent)
        self.hotkey_id = hotkey_id
        self.modifiers = modifiers
        self.vk_code = vk_code
        self._running = False
        self.last_error: Optional[int] = None
        self.last_error_message: Optional[str] = None

        # Connect registration failure signal to notify the user
        self.registration_failed.connect(self._on_registration_failed)

    def _on_registration_failed(self, error_message: str):
        """Notification handler for hotkey registration failure."""
        self.last_error_message = error_message
        logger.warning("Global hotkey registration warning: %s", error_message)
        parent = self.parent()
        if parent is not None:
            if hasattr(parent, "status_pill"):
                parent.status_pill.setText("● Hotkey Conflict")
            elif hasattr(parent, "bridge") and hasattr(parent.bridge, "status_updated"):
                parent.bridge.status_updated.emit("Hotkey Conflict")

    def run(self):
        self._running = True
        # Register global hotkey
        success = user32.RegisterHotKey(None, self.hotkey_id, self.modifiers, self.vk_code)
        if not success:
            err = ctypes.get_last_error()
            if err == 0 and hasattr(ctypes, "GetLastError"):
                err = ctypes.GetLastError()
            self.last_error = err
            err_desc = ""
            try:
                if err:
                    err_desc = f": {ctypes.FormatError(err).strip()}"
            except Exception:
                pass
            self.registration_failed.emit(
                f"Global hotkey registration failed (Win32 error {err}{err_desc}). Hotkey may be reserved by another application."
            )
            return

        msg = wintypes.MSG()
        try:
            while self._running:
                # PeekMessage / GetMessage to process Windows message queue
                if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # PM_REMOVE
                    if msg.message == WM_HOTKEY and msg.wParam == self.hotkey_id:
                        self.hotkey_triggered.emit()
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
                self.msleep(30)
        finally:
            user32.UnregisterHotKey(None, self.hotkey_id)

    def stop(self):
        self._running = False
        self.wait(1000)
