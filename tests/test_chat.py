"""Offline contract tests for the vLLM provider plugin."""

import asyncio

import httpx

from trussium.capabilities.chat import (
    ChatCompletionRequest,
    ChatMessage,
    ChatRole,
    ChatStreamErrorEvent,
    ChatStreamEvent,
)
from trussium_provider_vllm import VLLMChatCapability


def request(stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="qwen",
        messages=[ChatMessage(role=ChatRole.USER, content="hello")],
        stream=stream,
    )


def test_complete_normalizes_vllm_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "chat-1",
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
        )

    capability = VLLMChatCapability(
        "http://vllm",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://vllm"
        ),
    )
    response = asyncio.run(capability.complete(request()))
    assert response.provider == "vllm"
    assert response.choices[0].message.content == "hi"


def test_stream_normalizes_sse_lifecycle() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = "\n".join(
            [
                'data: {"id":"chat-1","model":"qwen","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}',
                'data: {"id":"chat-1","model":"qwen","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}',
                "data: [DONE]",
            ]
        )
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vllm"
    )
    capability = VLLMChatCapability("http://vllm", client=client)

    async def collect() -> list[ChatStreamEvent]:
        return [event async for event in capability.stream(request(True))]

    events = asyncio.run(collect())
    assert [event.type for event in events] == ["start", "delta", "end"]


def test_http_errors_are_bounded() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(429)),
        base_url="http://vllm",
    )
    capability = VLLMChatCapability("http://vllm", client=client)

    async def collect() -> list[ChatStreamEvent]:
        return [event async for event in capability.stream(request(True))]

    events = asyncio.run(collect())
    assert isinstance(events[0], ChatStreamErrorEvent)
    assert events[0].code == "vllm_rate_limited"
