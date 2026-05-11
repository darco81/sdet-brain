"""LLM factory (T2-05 / T4-03).

`get_llm()` returns an :class:`ILLM` for one specific model and is
kept for callers that want a single named instance. `get_router()`
returns the shared :class:`LLMRouter` that maps task tiers onto
distinct models with per-model caching.
"""

from __future__ import annotations

import logging
from threading import Lock

from sdet_brain.config import Settings, get_settings
from sdet_brain.llm.mlx_provider import DEFAULT_MODEL, MLXLLm
from sdet_brain.llm.protocol import ILLM
from sdet_brain.llm.router import LLMRouter

logger = logging.getLogger(__name__)

_ROUTER: LLMRouter | None = None
_ROUTER_LOCK = Lock()


def get_llm(model_name: str | None = None) -> ILLM:
    """Build an :class:`ILLM` for ``model_name`` (default: Qwen3-Next-80B-Instruct-4bit)."""
    name = model_name or DEFAULT_MODEL
    logger.info("LLM factory selected MLX provider with model=%s", name)
    return MLXLLm(model_name=name)


def get_router(settings: Settings | None = None) -> LLMRouter:
    """Return the process-wide :class:`LLMRouter`.

    The router is built lazily on first call and reused thereafter so
    every tool ends up with the same per-model cache.
    """
    global _ROUTER
    if _ROUTER is not None:
        return _ROUTER
    with _ROUTER_LOCK:
        if _ROUTER is None:
            cfg = settings or get_settings()
            _ROUTER = LLMRouter(
                fast_model=cfg.llm_fast_model,
                instruct_model=cfg.llm_model,
                reasoning_model=cfg.llm_reasoning_model,
                enabled=cfg.llm_routing_enabled,
                cache_size=cfg.llm_router_cache_size,
            )
    assert _ROUTER is not None  # noqa: S101 - lock guarantees this
    return _ROUTER


def reset_router_for_tests() -> None:
    """Drop the cached router so a fresh test fixture can install its own."""
    global _ROUTER
    _ROUTER = None
