"""Tests for the tiered LLM router (T4-03)."""

from __future__ import annotations

import pytest

from sdet_brain.llm.factory import get_router, reset_router_for_tests
from sdet_brain.llm.mlx_provider import MLXLLm
from sdet_brain.llm.router import (
    DEFAULT_FAST_MODEL,
    DEFAULT_INSTRUCT_MODEL,
    DEFAULT_REASONING_MODEL,
    LLMRouter,
)


class _FakeILLM:
    """Stand-in for ``MLXLLm`` that mimics the unload contract.

    The router only introspects ``_model`` / ``_tokenizer`` on
    ``MLXLLm`` instances. This fake quacks like one so eviction tests
    do not need real MLX weights.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: object | None = object()
        self._tokenizer: object | None = object()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


def _patch_router_with_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap ``MLXLLm`` inside the router module for ``_FakeILLM``."""
    import sdet_brain.llm.router as router_module

    monkeypatch.setattr(router_module, "MLXLLm", _FakeILLM)


def _as_fake(provider: object) -> _FakeILLM:
    """Narrow a router return value to ``_FakeILLM`` for test assertions."""
    assert isinstance(provider, _FakeILLM)
    return provider


def test_router_select_model_uses_fast_for_fast_task() -> None:
    router = LLMRouter()
    assert router.select_model("fast") == DEFAULT_FAST_MODEL


def test_router_select_model_uses_instruct_for_summarize_and_chat() -> None:
    router = LLMRouter()
    assert router.select_model("summarize") == DEFAULT_INSTRUCT_MODEL
    assert router.select_model("chat") == DEFAULT_INSTRUCT_MODEL


def test_router_select_model_uses_reasoning_for_decompose_and_judge() -> None:
    router = LLMRouter()
    assert router.select_model("reasoning") == DEFAULT_REASONING_MODEL
    assert router.select_model("decompose") == DEFAULT_REASONING_MODEL
    assert router.select_model("judge") == DEFAULT_REASONING_MODEL


def test_router_disabled_returns_instruct_model_for_every_task() -> None:
    """When routing is off the brain behaves like v0.3.0 single-model mode."""
    router = LLMRouter(enabled=False)
    for task in ["fast", "summarize", "chat", "reasoning", "decompose", "judge"]:
        assert router.select_model(task) == DEFAULT_INSTRUCT_MODEL  # type: ignore[arg-type]


def test_router_get_caches_per_model() -> None:
    """Two ``get`` calls with the same task return the same provider."""
    router = LLMRouter()
    first = router.get("fast")
    second = router.get("fast")
    assert first is second
    assert isinstance(first, MLXLLm)
    assert first.model_name == DEFAULT_FAST_MODEL


def test_router_get_returns_distinct_providers_per_tier() -> None:
    router = LLMRouter()
    fast = router.get("fast")
    summary = router.get("summarize")
    reasoning = router.get("reasoning")
    assert fast is not summary
    assert summary is not reasoning
    assert {fast.model_name, summary.model_name, reasoning.model_name} == {
        DEFAULT_FAST_MODEL,
        DEFAULT_INSTRUCT_MODEL,
        DEFAULT_REASONING_MODEL,
    }


def test_router_loaded_models_reflects_get_calls() -> None:
    """``loaded_models`` mirrors the cache. With ``cache_size=2`` both
    distinct tiers stay resident; the default ``cache_size=1`` would
    evict the first, which is the LRU cap behaviour covered
    separately below."""
    router = LLMRouter(cache_size=2)
    assert router.loaded_models() == []
    router.get("fast")
    router.get("summarize")
    loaded = sorted(router.loaded_models())
    assert loaded == sorted([DEFAULT_FAST_MODEL, DEFAULT_INSTRUCT_MODEL])


def test_router_disabled_collapses_cache_to_one() -> None:
    """Disabled routing means every task hits the instruct model only."""
    router = LLMRouter(enabled=False)
    router.get("fast")
    router.get("summarize")
    router.get("reasoning")
    assert router.loaded_models() == [DEFAULT_INSTRUCT_MODEL]


def test_router_custom_models() -> None:
    router = LLMRouter(
        fast_model="my/fast",
        instruct_model="my/instruct",
        reasoning_model="my/reasoning",
    )
    assert router.select_model("fast") == "my/fast"
    assert router.select_model("chat") == "my/instruct"
    assert router.select_model("judge") == "my/reasoning"


def test_factory_get_router_is_singleton() -> None:
    reset_router_for_tests()
    a = get_router()
    b = get_router()
    assert a is b


def test_factory_reset_router_for_tests_drops_singleton() -> None:
    reset_router_for_tests()
    a = get_router()
    reset_router_for_tests()
    b = get_router()
    assert a is not b


# --- LRU cache cap ---------------------------------------------------------


def test_default_cache_size_is_one() -> None:
    """A fresh router must cap concurrent MLX models at one by default."""
    router = LLMRouter()
    assert router.cache_size == 1


def test_cache_size_below_one_raises_value_error() -> None:
    """``cache_size`` below 1 is meaningless and must be rejected."""
    with pytest.raises(ValueError, match="cache_size must be >= 1"):
        LLMRouter(cache_size=0)


def test_cache_evicts_lru_when_full_size_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """size=1: warming a second tier must evict the first."""
    _patch_router_with_fake(monkeypatch)
    router = LLMRouter(
        fast_model="model-a",
        instruct_model="model-b",
        reasoning_model="model-c",
        cache_size=1,
    )
    first = _as_fake(router.get("fast"))
    second = _as_fake(router.get("summarize"))
    assert first.is_loaded is False, "evicted provider must be unloaded"
    assert second.is_loaded is True
    assert list(router._cache.keys()) == ["model-b"]


def test_cache_respects_configured_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """size=2: third distinct model evicts the oldest."""
    _patch_router_with_fake(monkeypatch)
    router = LLMRouter(
        fast_model="model-a",
        instruct_model="model-b",
        reasoning_model="model-c",
        cache_size=2,
    )
    a = _as_fake(router.get("fast"))
    b = _as_fake(router.get("summarize"))
    c = _as_fake(router.get("reasoning"))
    assert a.is_loaded is False, "oldest entry must be evicted"
    assert b.is_loaded is True
    assert c.is_loaded is True
    assert list(router._cache.keys()) == ["model-b", "model-c"]


def test_cache_lru_touch_on_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """size=2: reusing an entry moves it to the back of the LRU queue."""
    _patch_router_with_fake(monkeypatch)
    router = LLMRouter(
        fast_model="model-a",
        instruct_model="model-b",
        reasoning_model="model-c",
        cache_size=2,
    )
    a = _as_fake(router.get("fast"))
    b = _as_fake(router.get("summarize"))
    a_again = _as_fake(router.get("fast"))
    c = _as_fake(router.get("reasoning"))
    assert a is a_again
    assert b.is_loaded is False
    assert a.is_loaded is True
    assert c.is_loaded is True


def test_evicted_provider_is_unloaded_via_internal_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eviction must null both ``_model`` and ``_tokenizer``."""
    _patch_router_with_fake(monkeypatch)
    router = LLMRouter(
        fast_model="model-a",
        instruct_model="model-b",
        cache_size=1,
    )
    evicted = _as_fake(router.get("fast"))
    router.get("summarize")
    assert evicted._model is None
    assert evicted._tokenizer is None


def test_repeated_calls_construct_provider_once_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """100 ``get`` calls on the same task must build the provider once.

    Regression guard: prior to capping plus LRU semantics the router
    silently grew its cache and never released MLX weights.
    """
    import sdet_brain.llm.router as router_module

    construct_count = 0
    original_init = _FakeILLM.__init__

    def counting_init(self: _FakeILLM, model_name: str) -> None:
        nonlocal construct_count
        construct_count += 1
        original_init(self, model_name)

    monkeypatch.setattr(_FakeILLM, "__init__", counting_init)
    monkeypatch.setattr(router_module, "MLXLLm", _FakeILLM)

    router = LLMRouter(cache_size=1)
    for _ in range(100):
        router.get("fast")

    assert construct_count == 1
