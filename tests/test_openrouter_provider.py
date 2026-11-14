import pytest
import httpx
from shamu.types import ChatMessage, Role
from shamu.protocols import ProviderAuthError, ProviderUnavailableError, ShamuError
from shamu.gateway.openrouter_provider import OpenRouterProvider


@pytest.mark.asyncio
async def test_openrouter_missing_key():
    provider = OpenRouterProvider(api_key=None)
    with pytest.raises(ProviderAuthError) as exc_info:
        async for _ in provider.stream_chat([ChatMessage(role=Role.USER, content="Hi")]):
            pass
    assert exc_info.value.provider == "openrouter"


@pytest.mark.asyncio
async def test_openrouter_streaming_success(monkeypatch):
    lines = [
        b'data: {"choices":[{"delta":{"content":"Cloud"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" response"},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    class MockStreamResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            for l in lines:
                yield l.decode("utf-8")

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def stream(self, method, url, **kwargs):
            headers = kwargs.get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer mock-token"
            return MockStreamResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    provider = OpenRouterProvider(api_key="mock-token")
    chunks = []
    async for chunk in provider.stream_chat([ChatMessage(role=Role.USER, content="Hello")]):
        chunks.append(chunk)

    text = "".join(c.text_delta for c in chunks)
    assert text == "Cloud response"


@pytest.mark.asyncio
async def test_openrouter_multi_tool_streaming_separated_indices(monkeypatch):
    """ARCH-02: Ensure parallel tool calls with separate indices do not overwrite or concatenate."""
    lines = [
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{\\"path\\": "}}]},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"first.txt\\"}"}}]},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","function":{"name":"write_file","arguments":"{\\"path\\": "}}]},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\\"second.txt\\", \\"content\\": \\"ok\\"}"}}]},"finish_reason":null}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    class MockStreamResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            for l in lines:
                yield l.decode("utf-8")

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def stream(self, method, url, **kwargs):
            return MockStreamResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    provider = OpenRouterProvider(api_key="mock-key")
    chunks = []
    async for chunk in provider.stream_chat([ChatMessage(role=Role.USER, content="Run tools")]):
        if chunk.is_tool_call and chunk.tool_call_delta:
            chunks.append(chunk.tool_call_delta)

    # We must have chunks for call_1 and call_2
    call_1_chunks = [c for c in chunks if c.id == "call_1"]
    call_2_chunks = [c for c in chunks if c.id == "call_2"]

    assert len(call_1_chunks) >= 2
    assert len(call_2_chunks) >= 2

    # The final arguments for call_1 must be purely first.txt, NOT contaminated with second.txt
    assert call_1_chunks[-1].arguments == '{"path": "first.txt"}'
    assert call_1_chunks[-1].name == "read_file"

    # The final arguments for call_2 must be purely second.txt, NOT contaminated with first.txt
    assert call_2_chunks[-1].arguments == '{"path": "second.txt", "content": "ok"}'
    assert call_2_chunks[-1].name == "write_file"


@pytest.mark.asyncio
async def test_openrouter_connection_pool_reuse():
    """ARCH-06 / PERF-06: Consecutive calls to OpenRouterProvider reuse the same persistent HTTP client."""
    client_instance = httpx.AsyncClient()
    provider = OpenRouterProvider(api_key="mock-key", client=client_instance)

    assert provider._get_client() is client_instance
    assert provider._get_client() is client_instance

    # Test clean close
    await provider.aclose()
    assert provider._client is None
    assert client_instance.is_closed is True


@pytest.mark.asyncio
async def test_openrouter_empty_choices_sse_chunk(monkeypatch):
    """SSE chunk with empty choices list must not raise IndexError and must continue."""
    lines = [
        b'data: {"choices": []}\n\n',
        b'data: {"choices":[{"delta":{"content":"Cloud recovered"},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    class MockStreamResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            for l in lines:
                yield l.decode("utf-8")

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def stream(self, method, url, **kwargs):
            return MockStreamResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    provider = OpenRouterProvider(api_key="mock-key")
    chunks = []
    async for chunk in provider.stream_chat([ChatMessage(role=Role.USER, content="Hi")]):
        chunks.append(chunk)

    text = "".join(c.text_delta for c in chunks)
    assert text == "Cloud recovered"


@pytest.mark.asyncio
async def test_openrouter_mid_stream_rate_limit_error(monkeypatch):
    """SSE mid-stream error with 429 raises ProviderUnavailableError."""
    lines = [
        b'data: {"choices":[{"delta":{"content":"Starting cloud stream..."},"finish_reason":null}]}\n\n',
        b'data: {"error":{"code": 429, "message": "Provider rate limit exceeded"}}\n\n',
    ]

    class MockStreamResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            for l in lines:
                yield l.decode("utf-8")

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def stream(self, method, url, **kwargs):
            return MockStreamResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    provider = OpenRouterProvider(api_key="mock-key")
    with pytest.raises(ProviderUnavailableError) as exc_info:
        async for _ in provider.stream_chat([ChatMessage(role=Role.USER, content="Hi")]):
            pass

    assert exc_info.value.provider == "openrouter"
    assert "rate limit" in str(exc_info.value).lower() or "429" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openrouter_mid_stream_generic_error(monkeypatch):
    """SSE mid-stream error without 429 raises ShamuError."""
    lines = [
        b'data: {"error":{"message": "Upstream model crashed", "code": 502}}\n\n',
    ]

    class MockStreamResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            for l in lines:
                yield l.decode("utf-8")

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def stream(self, method, url, **kwargs):
            return MockStreamResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    provider = OpenRouterProvider(api_key="mock-key")
    with pytest.raises(ShamuError) as exc_info:
        async for _ in provider.stream_chat([ChatMessage(role=Role.USER, content="Hi")]):
            pass

    assert not isinstance(exc_info.value, ProviderUnavailableError)
    assert "mid-stream error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openrouter_transport_and_http_errors_wrapped(monkeypatch):
    """TransportError, ProtocolError, HTTPError during streaming must wrap to ProviderUnavailableError."""
    errors_to_test = [
        httpx.RemoteProtocolError("connection closed mid-stream"),
        httpx.TransportError("network interface dropped"),
        httpx.HTTPError("general http stream error"),
    ]

    for err in errors_to_test:
        class MockFailingStreamResponse:
            status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            async def aiter_lines(self):
                raise err
                yield ""

        class MockClient:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            def stream(self, method, url, **kwargs):
                return MockFailingStreamResponse()

        monkeypatch.setattr(httpx, "AsyncClient", MockClient)

        provider = OpenRouterProvider(api_key="mock-key")
        with pytest.raises(ProviderUnavailableError) as exc_info:
            async for _ in provider.stream_chat([ChatMessage(role=Role.USER, content="Hi")]):
                pass
        assert exc_info.value.provider == "openrouter"




