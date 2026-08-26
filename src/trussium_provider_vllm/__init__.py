"""Standalone Trussium provider plugin for vLLM."""

from trussium_provider_vllm.chat import VLLMChatCapability, VLLMProviderError

__all__ = ["VLLMChatCapability", "VLLMProviderError"]
