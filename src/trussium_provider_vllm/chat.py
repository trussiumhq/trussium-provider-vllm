"""vLLM OpenAI-compatible chat adapter."""

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from trussium.capabilities.chat import (
    ChatCapability,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatRole,
    ChatStreamDeltaEvent,
    ChatStreamEndEvent,
    ChatStreamErrorEvent,
    ChatStreamEvent,
    ChatStreamStartEvent,
    FinishReason,
    TokenUsage,
)
from trussium.errors import ProviderError


class VLLMProviderError(ProviderError):
    """Raised when vLLM returns a response that cannot be normalized."""


class VLLMChatCapability(ChatCapability):
    """Normalize vLLM's OpenAI-compatible chat-completions endpoint."""

    provider_name = "vllm"
    provider_display_name = "vLLM"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the adapter with a private vLLM base URL."""
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"))
        self._owns_client = client is None
        self._api_key = api_key

    async def aclose(self) -> None:
        """Close the HTTP client when the adapter owns it."""
        if self._owns_client:
            await self._client.aclose()

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute and normalize one non-streaming vLLM response."""
        try:
            response = await self._client.post(
                "/chat/completions",
                headers=self._headers(),
                json=self._payload(request, stream=False),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise self._status_error(error.response.status_code) from error
        except httpx.RequestError as error:
            raise VLLMProviderError(
                "vLLM connection failed", code="vllm_connection"
            ) from error

        return self._normalize_response(response.json())

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatStreamEvent]:
        """Execute and normalize vLLM SSE chat events."""
        response_id: str | None = None
        model = request.model
        started = False
        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                headers=self._headers(),
                json=self._payload(request, stream=True),
            ) as response:
                if response.is_error:
                    yield self._stream_error(self._status_error(response.status_code))
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        response_id = str(
                            chunk.get("id") or response_id or "vllm-response"
                        )
                        model = str(chunk.get("model") or model)
                        if not response_id:
                            raise ValueError("missing response id")
                        if chunk.get("choices"):
                            choice = chunk["choices"][0]
                            delta = choice.get("delta", {}).get("content")
                            if delta:
                                if not started:
                                    started = True
                                    yield ChatStreamStartEvent(
                                        id=response_id,
                                        provider=self.provider_name,
                                        model=model,
                                    )
                                yield ChatStreamDeltaEvent(
                                    id=response_id, content=delta
                                )
                            finish_reason = choice.get("finish_reason")
                            if finish_reason:
                                yield ChatStreamEndEvent(
                                    id=response_id,
                                    finish_reason=self._finish_reason(finish_reason),
                                    usage=self._usage(chunk.get("usage")),
                                )
                                return
                    except (ValueError, TypeError, json.JSONDecodeError):
                        yield ChatStreamErrorEvent(
                            id=response_id,
                            code="vllm_invalid_stream",
                            message="vLLM returned an invalid streaming event.",
                        )
                        return
        except httpx.RequestError:
            yield ChatStreamErrorEvent(
                id=response_id, code="vllm_connection", message="vLLM connection failed"
            )
            return

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    @staticmethod
    def _payload(request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        return {
            "model": request.model,
            "messages": [
                message.model_dump(mode="json") for message in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "stream": stream,
        }

    def _normalize_response(self, payload: Mapping[str, Any]) -> ChatCompletionResponse:
        try:
            choice = payload["choices"][0]
            message = choice["message"]
            return ChatCompletionResponse(
                id=str(payload["id"]),
                provider=self.provider_name,
                model=str(payload["model"]),
                choices=[
                    ChatCompletionChoice(
                        index=int(choice.get("index", 0)),
                        message=ChatMessage(
                            role=ChatRole(str(message["role"])),
                            content=str(message["content"]),
                        ),
                        finish_reason=self._finish_reason(str(choice["finish_reason"])),
                    )
                ],
                usage=self._usage(payload.get("usage")),
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise VLLMProviderError(
                "vLLM returned an invalid chat response", code="vllm_invalid_response"
            ) from error

    @staticmethod
    def _usage(value: Any) -> TokenUsage:
        usage = value if isinstance(value, Mapping) else {}
        return TokenUsage(
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
        )

    @staticmethod
    def _finish_reason(value: str) -> FinishReason:
        return {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALL,
            "content_filter": FinishReason.CONTENT_FILTER,
        }.get(value, FinishReason.ERROR)

    @staticmethod
    def _status_error(status_code: int) -> VLLMProviderError:
        code = "vllm_rate_limited" if status_code == 429 else "vllm_http_error"
        return VLLMProviderError("vLLM request failed", code=code)

    @staticmethod
    def _stream_error(error: VLLMProviderError) -> ChatStreamErrorEvent:
        return ChatStreamErrorEvent(id=None, code=error.code, message=error.message)
