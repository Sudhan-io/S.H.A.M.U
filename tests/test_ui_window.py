import pytest
from PySide6.QtWidgets import QApplication
from shamu.ui.window import MainWindow, UIBridge, ChatCard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ui_bridge_signals(qapp):
    bridge = UIBridge()
    received_chunks = []
    statuses = []

    bridge.chunk_received.connect(lambda text: received_chunks.append(text))
    bridge.status_updated.connect(lambda status: statuses.append(status))

    bridge.chunk_received.emit("Hello world")
    bridge.status_updated.emit("Ready")

    assert received_chunks == ["Hello world"]
    assert statuses == ["Ready"]


def test_main_window_structure(qapp):
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)

    assert "SHAMU" in window.windowTitle()
    assert window.input_field is not None
    assert window.ptt_btn is not None
    assert window.status_pill.text() == "● Ready"

    # Test adding messages
    window.add_user_message("Test user input")
    assert window.chat_layout.count() >= 2  # Stretch + card

    # Test streaming chunk to assistant card
    window._on_chunk("Hello from assistant")
    assert window.current_assistant_card is not None
    assert "Hello from assistant" in window.current_assistant_card.label.text()


def test_chat_card_markdown_rendering(qapp):
    """UX-01: ChatCard renders Markdown formatted text correctly."""
    card = ChatCard("**Bold Title**\n\n- item 1\n- item 2")
    assert card._raw_text == "**Bold Title**\n\n- item 1\n- item 2"

    card.append_text("\n\n```python\nprint(1)\n```")
    assert "print(1)" in card._raw_text
    assert "print(1)" in card.label.text()


def test_window_auto_scroll_range_changed(qapp):
    """UX-02: Auto-scroll tracks bottom on rangeChanged when auto-scroll is enabled."""
    window = MainWindow()
    vsb = window.scroll_area.verticalScrollBar()
    assert window._auto_scroll_enabled is True

    # Simulate rangeChanged with valid scrollbar range
    vsb.setRange(0, 500)
    window._on_scroll_range_changed(0, 500)
    assert vsb.value() == 500

    # User scrolls up -> auto_scroll paused
    window._on_scroll_value_changed(100)
    assert window._auto_scroll_enabled is False

    # New content arrives while user scrolled up -> does not interrupt user reading
    vsb.setRange(0, 600)
    window._on_scroll_range_changed(0, 600)
    assert vsb.value() != 600

    # User scrolls back to bottom -> auto_scroll resumed
    window._on_scroll_value_changed(590)
    assert window._auto_scroll_enabled is True


def test_ui_bridge_approval_requested_signal(qapp):
    """UIBridge defines approval_requested signal that emits (req, future)."""
    bridge = UIBridge()
    received = []
    bridge.approval_requested.connect(lambda req, fut: received.append((req, fut)))

    dummy_req = {"tool": "run_shell"}
    dummy_fut = object()
    bridge.approval_requested.emit(dummy_req, dummy_fut)

    assert len(received) == 1
    assert received[0] == (dummy_req, dummy_fut)


def test_streaming_chunk_respects_manual_scroll(qapp):
    """Streaming tokens do not override manual user scrolling when reading history."""
    window = MainWindow()
    vsb = window.scroll_area.verticalScrollBar()
    vsb.setRange(0, 1000)
    vsb.setValue(200)

    # User manually scrolled up
    window._auto_scroll_enabled = False
    window._scroll_to_bottom()
    assert vsb.value() == 200

    # Incoming streaming chunks should not force scrollbar to bottom
    window._on_chunk("streamed token")
    assert vsb.value() == 200

    # When auto scroll is enabled, scrollbar moves to maximum
    window._auto_scroll_enabled = True
    window._scroll_to_bottom()
    assert vsb.value() == vsb.maximum()


def test_main_window_on_send(qapp):
    """_on_send emits user_message_sent and clears input field."""
    bridge = UIBridge()
    window = MainWindow(bridge=bridge)
    emitted = []
    bridge.user_message_sent.connect(lambda text: emitted.append(text))

    window.input_field.setText("Hello SHAMU")
    window._on_send()

    assert emitted == ["Hello SHAMU"]
    assert window.input_field.text() == ""


def test_chat_card_debouncing_and_progressive_update(qapp):
    """UI-11: ChatCard debounces markdown re-parsing while progressively updating text."""
    card = ChatCard("", debounce_ms=100)
    assert card._raw_text == ""
    assert not card._render_timer.isActive()

    # Append fast tokens simulating streaming
    for token in ["Hello", " **world**", ", welcome", " to", " SHAMU!"]:
        card.append_text(token)

    # Progressive plain text is updated immediately
    assert "Hello **world**, welcome to SHAMU!" in card.label.text()
    # Timer is scheduled (debouncing markdown parsing)
    assert card._render_timer.isActive() is True
    assert card._pending_markdown is True

    # Immediate flush executes markdown rendering
    card.flush()
    assert card._render_timer.isActive() is False
    assert card._pending_markdown is False
    assert "SHAMU!" in card._raw_text


def test_window_stream_finished_finalizes_assistant_card(qapp):
    """UI-11: MainWindow stream finished finalizes current assistant card."""
    window = MainWindow()
    window._on_chunk("Final token")
    card = window.current_assistant_card
    assert card is not None
    assert card._render_timer.isActive() is True

    window._on_stream_finished()
    assert card._render_timer.isActive() is False
    assert window.current_assistant_card is None


def test_window_connect_hotkey_failure_notification(qapp):
    """UI-09: MainWindow.connect_hotkey connects registration failure signal."""
    from shamu.ui.hotkey import GlobalHotkeyListener
    window = MainWindow()
    listener = GlobalHotkeyListener()
    window.connect_hotkey(listener)

    assert window.status_pill.text() == "● Ready"
    listener.registration_failed.emit("Conflict error")
    assert "Hotkey Conflict" in window.status_pill.text()






