"""Thin wrapper around the OpenAI-compatible chat completion API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from openai import OpenAI

from .models import ProviderConfig


class LLMClientError(RuntimeError):
    """Raised when the underlying LLM call fails."""


class ChatClient(Protocol):
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass
class OpenAICompatibleClient:
    provider: ProviderConfig
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    def _openai(self) -> OpenAI:
        cached: Any = self._client
        if cached is None:
            cached = OpenAI(
                api_key=self.provider.api_key,
                base_url=self.provider.base_url,
                timeout=self.provider.request_timeout,
            )
            self._client = cached
        return cast(OpenAI, cached)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        client = self._openai()
        try:
            response = client.chat.completions.create(
                model=self.provider.model,
                temperature=self.provider.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # pragma: no cover - network failure paths
            raise LLMClientError(str(exc)) from exc
        choices = response.choices
        if not choices:
            raise LLMClientError("LLM returned no choices")
        message = choices[0].message
        content = message.content or ""
        if not content:
            raise LLMClientError("LLM returned empty content")
        return content
