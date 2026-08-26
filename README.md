# Trussium vLLM provider plugin

Standalone Python adapter for self-hosted [vLLM](https://github.com/vllm-project/vllm)
OpenAI-compatible chat completions. The application owner runs vLLM separately,
installs this package, and explicitly registers `VLLMChatCapability`.

```python
import httpx
from trussium.capabilities import CapabilityRegistry
from trussium_provider_vllm import VLLMChatCapability

registry = CapabilityRegistry()
registry.register(
    "chat.completions",
    VLLMChatCapability(
        "http://vllm.internal:8000/v1",
        api_key="optional-private-gateway-key",
    ),
)
registry.seal()
```

The adapter normalizes JSON and SSE responses, maps bounded HTTP failures,
preserves provider/model metadata, and does not log credentials or payloads. It
does not install, configure, or discover vLLM. Dynamic loading remains outside
the package; see Trussium ADR-0008.

Run offline tests with `uv run pytest`.
