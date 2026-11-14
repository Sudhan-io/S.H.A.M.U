import json
import pytest
import httpx
from shamu.types import ChatMessage, Role, ToolDefinition
from shamu.protocols import ProviderUnavailableError, ContextWindowExceededError, ShamuError
from shamu.gateway.ollama_provider import OllamaProvider


@pytest.mark.asyncio
async def test_ollama_streaming_success(monkeypatch):
    lines = [
        b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
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

    provider = OllamaProvider(base_url="http://mock-ollama:11434")
    chunks = []
    async for chunk in provider.stream_chat([ChatMessage(role=Role.USER, content="Hi")]):
        chunks.append(chunk)

    text = "".join(c.text_delta for c in chunks)
    assert text == "Hello world"
    assert any(c.finish_reason == "stop" for c in chunks)


@pytest.mark.asyncio
async def test_ollama_connection_error(monkeypatch):
    class MockFailingClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def stream(self, method, url, **kwargs):
            raise httpx.ConnectError("Failed to connect to localhost:11434")

    monkeypatch.setattr(httpx, "AsyncClient", MockFailingClient)

    provider = OllamaProvider()
    with pytest.raises(ProviderUnavailableError) as exc_info:
        async for _ in provider.stream_chat([ChatMessage(role=Role.USER, content="Hi")]):
            pass
    assert exc_info.value.provider == "ollama"
    assert "unreachable" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ollama_tool_call_delta(monkeypatch):
    lines = [
        b'data: {"choices":[{"delta":{"tool_calls":[{"id":"call_abc","function":{"name":"search","arguments":"{\\"q\\": "}}]},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\\"python\\"}"}}]},"finish_reason":null}]}\n\n',
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

    provider = OllamaProvider()
    tool_chunks = []
    async for chunk in provider.stream_chat([ChatMessage(role=Role.USER, content="Search")]):
        if chunk.is_tool_call:
            tool_chunks.append(chunk)

    assert len(tool_chunks) == 2
    final_tc = tool_chunks[-1].tool_call_delta
    assert final_tc.name == "search"
    assert final_tc.arguments == '{"q": "python"}'


@pytest.mark.asyncio
async def test_ollama_provider_injects_inference_options(monkeypatch):
    from shamu.config import InferenceConfig

    captured_payloads = []

    class MockStreamResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
            yield 'data: [DONE]\n\n'

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def stream(self, method, url, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            return MockStreamResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    inf_cfg = InferenceConfig(
        num_ctx=2048,
        keep_alive="10m",
        flash_attention=True,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )
    provider = OllamaProvider(inference_config=inf_cfg)
    async for _ in provider.stream_chat([ChatMessage(role=Role.USER, content="Hello")]):
        pass

    assert len(captured_payloads) == 1
    sent_opts = captured_payloads[0]["options"]
    assert sent_opts["num_ctx"] == 2048
    assert sent_opts["keep_alive"] == "10m"
    assert sent_opts["flash_attention"] is True
    assert sent_opts["cache_type_k"] == "q8_0"
    assert sent_opts["cache_type_v"] == "q8_0"


@pytest.mark.asyncio
async def test_ollama_multi_tool_streaming_separated_indices(monkeypatch):
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

    provider = OllamaProvider(base_url="http://mock-ollama:11434")
    chunks = []
    async for chunk in provider.stream_chat([ChatMessage(role=Role.USER, content="Run tools")]):
        if chunk.is_tool_call and chunk.tool_call_delta:
            chunks.append(chunk.tool_call_delta)

    call_1_chunks = [c for c in chunks if c.id == "call_1"]
    call_2_chunks = [c for c in chunks if c.id == "call_2"]

    assert len(call_1_chunks) >= 2
    assert len(call_2_chunks) >= 2

    assert call_1_chunks[-1].arguments == '{"path": "first.txt"}'
    assert call_1_chunks[-1].name == "read_file"

    assert call_2_chunks[-1].arguments == '{"path": "second.txt", "content": "ok"}'
    assert call_2_chunks[-1].name == "write_file"


@pytest.mark.asyncio
async def test_ollama_http_500_raises_provider_unavailable(monkeypatch):
    """ARCH-05: HTTP 5xx errors from Ollama must raise ProviderUnavailableError to trigger router fallback."""
    class Mock500Response:
        status_code = 500
        text = "Internal Server Error: model failed to load"
        async def aread(self):
            return b"Internal Server Error: model failed to load"
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockClient500:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def stream(self, method, url, **kwargs):
            return Mock500Response()
        async def post(self, url, **kwargs):
            return Mock500Response()

    monkeypatch.setattr(httpx, "AsyncClient", MockClient500)

    provider = OllamaProvider(base_url="http://mock-ollama:11434")

    # In stream_chat
    with pytest.raises(ProviderUnavailableError) as exc_info:
        async for _ in provider.stream_chat([ChatMessage(role=Role.USER, content="Hello")]):
            pass
    assert exc_info.value.provider == "ollama"
    assert "500" in str(exc_info.value)

    # In complete_chat
    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.complete_chat([ChatMessage(role=Role.USER, content="Hello")])
    assert exc_info.value.provider == "ollama"


@pytest.mark.asyncio
async def test_ollama_connection_pool_reuse():
    """ARCH-06 / PERF-06: Consecutive calls to OllamaProvider reuse the same persistent HTTP client."""
    client_instance = httpx.AsyncClient()
    provider = OllamaProvider(client=client_instance)

    assert provider._get_client() is client_instance
    assert provider._get_client() is client_instance

    # Test clean close
    await provider.aclose()
    assert provider._client is None
    assert client_instance.is_closed is True


@pytest.mark.asyncio
async def test_ollama_empty_choices_sse_chunk(monkeypatch):
    """SSE chunk with empty choices list must not raise IndexError and must continue."""
    lines = [
        b'data: {"choices": []}\n\n',
        b'data: {"choices":[{"delta":{"content":"Recovered"},"finish_reason":"stop"}]}\n\n',
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

    provider = OllamaProvider()
    chunks = []
    async for chunk in provider.stream_chat([ChatMessage(role=Role.USER, content="Hi")]):
        chunks.append(chunk)

    text = "".join(c.text_delta for c in chunks)
    assert text == "Recovered"


@pytest.mark.asyncio
async def test_ollama_mid_stream_rate_limit_error(monkeypatch):
    """SSE mid-stream error with 429 raises ProviderUnavailableError."""
    lines = [
        b'data: {"choices":[{"delta":{"content":"Start..."},"finish_reason":null}]}\n\n',
        b'data: {"error":{"code": 429, "message": "Too many requests"}}\n\n',
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

    provider = OllamaProvider()
    with pytest.raises(ProviderUnavailableError) as exc_info:
        async for _ in provider.stream_chat([ChatMessage(role=Role.USER, content="Hi")]):
            pass

    assert exc_info.value.provider == "ollama"
    assert "rate limit" in str(exc_info.value).lower() or "429" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ollama_mid_stream_generic_error(monkeypatch):
    """SSE mid-stream error without 429 raises ShamuError."""
    lines = [
        b'data: {"error":{"message": "Internal processing failure", "code": 500}}\n\n',
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

    provider = OllamaProvider()
    with pytest.raises(ShamuError) as exc_info:
        async for _ in provider.stream_chat([ChatMessage(role=Role.USER, content="Hi")]):
            pass

    assert not isinstance(exc_info.value, ProviderUnavailableError)
    assert "mid-stream error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ollama_transport_and_http_errors_wrapped(monkeypatch):
    """TransportError, ProtocolError, HTTPError during streaming must wrap to ProviderUnavailableError."""
    errors_to_test = [
        httpx.RemoteProtocolError("connection closed abruptly"),
        httpx.TransportError("transport failure"),
        httpx.HTTPError("generic http error"),
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

        provider = OllamaProvider()
        with pytest.raises(ProviderUnavailableError) as exc_info:
            async for _ in provider.stream_chat([ChatMessage(role=Role.USER, content="Hi")]):
                pass
        assert exc_info.value.provider == "ollama"






