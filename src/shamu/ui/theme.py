"""Windows 11 Fluent Dark Theme QSS Stylesheet for SHAMU."""

FLUENT_DARK_THEME = """
QWidget {
    background-color: #1a1a20;
    color: #f0f0f5;
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    border: none;
    background: #141418;
    width: 6px;
    margin: 0px;
    border-radius: 3px;
}

QScrollBar::handle:vertical {
    background: #3e3e4a;
    min-height: 20px;
    border-radius: 3px;
}

QScrollBar::handle:vertical:hover {
    background: #00b4d8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QLineEdit {
    background-color: #24242d;
    border: 1px solid #3a3a46;
    border-radius: 8px;
    padding: 10px 14px;
    color: #ffffff;
    font-size: 13px;
}

QLineEdit:focus {
    border: 1px solid #00b4d8;
    background-color: #282834;
}

QPushButton {
    background-color: #0077b6;
    color: #ffffff;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
}

QPushButton:hover {
    background-color: #0096c7;
}

QPushButton:pressed {
    background-color: #03045e;
}

QPushButton#PTTButton {
    background-color: #00b4d8;
    color: #03045e;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 14px;
}

QPushButton#PTTButton:hover {
    background-color: #48cae4;
}

QLabel#StatusPill {
    background-color: #24242d;
    border: 1px solid #3a3a46;
    border-radius: 12px;
    padding: 4px 10px;
    font-size: 11px;
    color: #90e0ef;
}

QFrame#UserCard {
    background-color: #0077b6;
    border-radius: 10px;
    padding: 10px;
    margin: 4px;
}

QFrame#AssistantCard {
    background-color: #24242d;
    border: 1px solid #32323e;
    border-radius: 10px;
    padding: 10px;
    margin: 4px;
}

QFrame#TaskPlanCard {
    background-color: #1a1a24;
    border: 1px solid #00b4d8;
    border-radius: 10px;
    padding: 12px;
    margin: 4px;
}

QFrame#TaskStepItem {
    background-color: #242430;
    border: 1px solid #323242;
    border-radius: 6px;
    padding: 8px;
    margin-top: 3px;
    margin-bottom: 3px;
}

QFrame#TaskReportCard {
    background-color: #1a1a24;
    border: 1px solid #2a9d8f;
    border-radius: 10px;
    padding: 12px;
    margin: 4px;
}
"""

