"""Tests for the OpenAI-compatible LLM client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from dataset_generator.llm import LLMClientError, OpenAICompatibleClient
from dataset_generator.models import ProviderConfig


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.next_content: str = "ok"
        self.next_choices: list[object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.next_choices is not None:
            return SimpleNamespace(choices=self.next_choices)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=_FakeMessage(self.next_content))]
        )


class _FakeOpenAI:
    instances: ClassVar[list[_FakeOpenAI]] = []

    def __init__(self, *, api_key: str, base_url: str | None, timeout: float) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=_FakeCompletions())
        _FakeOpenAI.instances.append(self)


@pytest.fixture(autouse=True)
def _reset_fake_openai() -> None:
    _FakeOpenAI.instances = []


def _patch_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dataset_generator.llm.OpenAI", _FakeOpenAI)


def test_client_uses_provider_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_openai(monkeypatch)
    provider = ProviderConfig(
        model="custom-model",
        api_key="sk-test",
        base_url="https://example.com/v1",
        temperature=0.3,
        request_timeout=42.0,
    )
    client = OpenAICompatibleClient(provider)

    result = client.chat("sys", "usr")

    assert result == "ok"
    assert len(_FakeOpenAI.instances) == 1
    instance = _FakeOpenAI.instances[0]
    assert instance.api_key == "sk-test"
    assert instance.base_url == "https://example.com/v1"
    assert instance.timeout == 42.0
    completions = instance.chat.completions
    assert isinstance(completions, _FakeCompletions)
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["model"] == "custom-model"
    assert call["temperature"] == 0.3


def test_client_defaults_base_url_to_none(monkeypatch) -> None:
    _patch_openai(monkeypatch)
    provider = ProviderConfig(
        model="custom-model",
        api_key="sk-test",
        base_url=None,
        temperature=0.5,
        request_timeout=30.0,
    )
    client = OpenAICompatibleClient(provider)
    client.chat("sys", "usr")
    instance = _FakeOpenAI.instances[0]
    assert instance.base_url is None


def test_client_propagates_empty_content_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_openai(monkeypatch)
    provider = ProviderConfig(
        model="m",
        api_key="k",
        base_url=None,
        temperature=0.0,
        request_timeout=30.0,
    )
    client = OpenAICompatibleClient(provider)
    client.chat("sys", "usr")  # triggers lazy construction
    instance = _FakeOpenAI.instances[0]
    instance.chat.completions.next_content = ""
    with pytest.raises(LLMClientError):
        client.chat("sys", "usr")


def test_client_propagates_no_choices_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_openai(monkeypatch)
    provider = ProviderConfig(
        model="m",
        api_key="k",
        base_url=None,
        temperature=0.0,
        request_timeout=30.0,
    )
    client = OpenAICompatibleClient(provider)
    client.chat("sys", "usr")  # triggers lazy construction
    instance = _FakeOpenAI.instances[0]
    instance.chat.completions.next_choices = []
    with pytest.raises(LLMClientError):
        client.chat("sys", "usr")
