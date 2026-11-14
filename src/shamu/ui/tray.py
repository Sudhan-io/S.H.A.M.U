"""Windows System Tray Icon and Context Menu for SHAMU."""

from typing import Callable, Optional
from PySide6.QtGui import QIcon, QPixmap, QColor, QPainter
from PySide6.QtWidgets import QSystemTrayIcon, QMenu, QApplication


def create_default_icon() -> QIcon:
    """Generates a stylish programmatic 32x32 circular icon for the system tray."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(QColor(0, 0, 0, 0))  # Transparent

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    # Circle background
    painter.setBrush(QColor("#0077b6"))
    painter.setPen(QColor("#00b4d8"))
    painter.drawEllipse(2, 2, 28, 28)

    # Letter 'S' in center
    painter.setPen(QColor("#ffffff"))
    font = painter.font()
    font.setBold(True)
    font.setPixelSize(16)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), 0x0084, "S")  # AlignCenter
    painter.end()

    return QIcon(pixmap)


class ShamuTrayIcon(QSystemTrayIcon):
    """System Tray controller enabling background dock and quick actions."""

    def __init__(
        self,
        on_toggle_window: Callable[[], None],
        on_ptt: Callable[[], None],
        on_quit: Callable[[], None],
        parent=None,
    ):
        super().__init__(create_default_icon(), parent)
        self.setToolTip("SHAMU Desktop Assistant (Active)")

        self.on_toggle_window = on_toggle_window
        self.on_ptt = on_ptt
        self.on_quit = on_quit

        self._init_menu()
        self.activated.connect(self._on_activated)

    def _init_menu(self):
        menu = QMenu()
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e1e24;
                color: #ffffff;
                border: 1px solid #32323e;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item:selected {
                background-color: #0077b6;
                border-radius: 4px;
            }
        """)

        show_act = menu.addAction("Show SHAMU")
        show_act.triggered.connect(self.on_toggle_window)

        ptt_act = menu.addAction("Push to Talk (Mic)")
        ptt_act.triggered.connect(self.on_ptt)

        menu.addSeparator()

        quit_act = menu.addAction("Quit SHAMU")
        quit_act.triggered.connect(self.on_quit)

        self.setContextMenu(menu)

    def _on_activated(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.on_toggle_window()

