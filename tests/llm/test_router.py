"""Tests for the tiered LLM router (T4-03)."""

from __future__ import annotations

from sdet_brain.llm.factory import get_router, reset_router_for_tests
from sdet_brain.llm.mlx_provider import MLXLLm
from sdet_brain.llm.router import (
    DEFAULT_FAST_MODEL,
    DEFAULT_INSTRUCT_MODEL,
    DEFAULT_REASONING_MODEL,
    LLMRouter,
)


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
    router = LLMRouter()
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
