"""Tests for the LLM protocol + factory contract (T2-05).

We do not load the real Qwen weights here - the production
``MLXLLm`` is exercised by smoke tests run by hand. These tests
cover the typed contract and the lazy / health-check semantics.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from sdet_brain.llm import ILLM, ChatMessage, LLMError, get_llm
from sdet_brain.llm.mlx_provider import DEFAULT_MODEL, MLXLLm


class _FakeLLM:
    """Inline fake satisfying :class:`ILLM`."""

    model_name = "fake/llm"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._loaded = False

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if not prompt.strip():
            raise LLMError("prompt empty")
        self._loaded = True
        self.calls.append(prompt)
        return f"echo:{prompt[:40]}"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        if not messages:
            raise LLMError("messages empty")
        return self.generate(messages[-1].content)

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        full = self.chat(messages, max_tokens=max_tokens, temperature=temperature)
        for token in full.split():
            yield token + " "

    def health_check(self) -> bool:
        return self._loaded


def test_factory_returns_mlxllm_with_default_model() -> None:
    llm = get_llm()
    assert isinstance(llm, MLXLLm)
    assert llm.model_name == DEFAULT_MODEL


def test_factory_honours_override() -> None:
    llm = get_llm("mlx-community/something-else")
    assert llm.model_name == "mlx-community/something-else"


def test_fake_satisfies_illm_protocol() -> None:
    fake = _FakeLLM()
    assert isinstance(fake, ILLM)


def test_fake_chat_round_trip() -> None:
    fake = _FakeLLM()
    out = fake.chat([ChatMessage(role="user", content="hello brain")])
    assert "hello brain" in out


def test_fake_empty_messages_raises() -> None:
    fake = _FakeLLM()
    with pytest.raises(LLMError):
        fake.chat([])


def test_fake_empty_prompt_raises() -> None:
    fake = _FakeLLM()
    with pytest.raises(LLMError):
        fake.generate("   ")


def test_fake_stream_yields_tokens() -> None:
    fake = _FakeLLM()
    chunks = list(fake.chat_stream([ChatMessage(role="user", content="alpha beta")]))
    assert chunks  # non-empty
    assert "".join(chunks).startswith("echo:alpha")


def test_health_check_starts_false_until_used() -> None:
    fake = _FakeLLM()
    assert fake.health_check() is False
    fake.generate("warm me up")
    assert fake.health_check() is True


def test_mlx_provider_health_check_starts_false() -> None:
    """Real provider must not load weights just because health_check ran."""
    provider = MLXLLm(model_name="mlx-community/totally-fake-for-test")
    assert provider.health_check() is False
    assert provider.is_loaded is False


def test_settings_expose_default_llm_router_cache_size() -> None:
    """The new env-driven field defaults to 1 (single resident model)."""
    from sdet_brain.config import Settings

    assert Settings().llm_router_cache_size == 1


def test_get_router_uses_settings_cache_size() -> None:
    """Factory must propagate ``cache_size`` from ``Settings`` into router."""
    from sdet_brain.config import Settings
    from sdet_brain.llm.factory import get_router, reset_router_for_tests

    reset_router_for_tests()
    cfg = Settings(llm_router_cache_size=3)
    router = get_router(cfg)
    try:
        assert router.cache_size == 3
    finally:
        reset_router_for_tests()
