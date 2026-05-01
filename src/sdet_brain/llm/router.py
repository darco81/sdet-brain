"""Tiered LLM routing (T4-03).

Brain v0.3.0 used a single 80B model for every LLM call. That's
over-provisioned for fast tasks (HyDE rewrites) and the wrong model
for reasoning tasks (decomposition, judging). The router maps a
``task_type`` onto the right MLX model and caches one provider
instance per model, so the second call to a given task pays no cold
start.

The router is intentionally a thin layer: no smart heuristics, no
dynamic model selection beyond the task->model table. If we ever need
a "model X for queries longer than N tokens" rule, it lives here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from threading import Lock
from typing import Final, Literal

from sdet_brain.llm.mlx_provider import MLXLLm
from sdet_brain.llm.protocol import ILLM, ChatMessage

logger = logging.getLogger(__name__)

TaskType = Literal[
    "fast",
    "summarize",
    "chat",
    "reasoning",
    "decompose",
    "judge",
]

# Defaults for each task tier. ``Settings`` can override via env vars.
DEFAULT_FAST_MODEL: Final[str] = "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit"
DEFAULT_INSTRUCT_MODEL: Final[str] = (
    "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"
)
DEFAULT_REASONING_MODEL: Final[str] = (
    "mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit"
)


class LLMRouter:
    """Picks the right :class:`ILLM` for a task and caches by model name.

    Each call to :meth:`get` returns a provider instance keyed by the
    resolved model name. The first ``generate`` / ``chat`` call on a
    given provider pays the MLX cold start; subsequent calls reuse the
    warm weights. The router itself never loads anything.
    """

    def __init__(
        self,
        *,
        fast_model: str = DEFAULT_FAST_MODEL,
        instruct_model: str = DEFAULT_INSTRUCT_MODEL,
        reasoning_model: str = DEFAULT_REASONING_MODEL,
        enabled: bool = True,
    ) -> None:
        self._fast_model = fast_model
        self._instruct_model = instruct_model
        self._reasoning_model = reasoning_model
        self._enabled = enabled
        self._cache: dict[str, ILLM] = {}
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def select_model(self, task: TaskType) -> str:
        """Return the model id appropriate for ``task``.

        When routing is disabled (``Settings.llm_routing_enabled=False``)
        every task falls back to the instruct-tier model so behaviour
        matches v0.3.0.
        """
        if not self._enabled:
            return self._instruct_model
        if task == "fast":
            return self._fast_model
        if task in ("reasoning", "decompose", "judge"):
            return self._reasoning_model
        # summarize, chat, default
        return self._instruct_model

    def get(self, task: TaskType) -> ILLM:
        """Return a cached :class:`ILLM` for ``task``."""
        model_name = self.select_model(task)
        cached = self._cache.get(model_name)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(model_name)
            if cached is None:
                logger.info(
                    "LLMRouter creating provider task=%s model=%s",
                    task,
                    model_name,
                )
                cached = MLXLLm(model_name=model_name)
                self._cache[model_name] = cached
        return cached

    # Convenience pass-through methods so callers can use the router
    # itself like an ``ILLM`` for the most common task tiers.

    def generate(
        self,
        prompt: str,
        *,
        task: TaskType = "summarize",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        return self.get(task).generate(
            prompt, max_tokens=max_tokens, temperature=temperature
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        task: TaskType = "chat",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        return self.get(task).chat(
            messages, max_tokens=max_tokens, temperature=temperature
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        task: TaskType = "chat",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        return self.get(task).chat_stream(
            messages, max_tokens=max_tokens, temperature=temperature
        )

    def loaded_models(self) -> list[str]:
        """Return model names that have a provider in the cache.

        Note: a cached provider may still be cold (lazy load happens on
        the first generate / chat call), so this list reflects "router
        was asked for this model", not "MLX weights resident in RAM".
        """
        return list(self._cache.keys())
