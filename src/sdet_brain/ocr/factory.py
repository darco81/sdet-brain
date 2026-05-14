"""OCR engine selection with health-check based fallback chain.

``get_ocr_engine(settings)`` walks a (provider, model) chain derived
from ``settings.ocr_provider`` and the configured model fields. The
first link whose builder + ``health_check()`` succeeds wins.

Mac (default — ``ocr_provider="mlx-vlm"``)::

    1. mlx-vlm  + ocr_mlx_vlm_model           ~3 s/img
    2. ollama   + ocr_ollama_primary_model    ~5-10 s/img
    3. ollama   + ocr_ollama_fallback_model   ~40 s/img  (skipped if None)

Win (``ocr_provider="ollama"`` + ``ocr_ollama_fallback_model=None``)::

    1. ollama   + ocr_ollama_primary_model    ~10-15 s/img
       (chain ends — qwen fallback skipped, mlx-vlm builder raises)

The winning selection is cached process-wide in a thread-safe
singleton so heavyweight model weights load only once. Tests call
``reset_ocr_engine()`` between cases to isolate state.

Providers themselves land in M2 (MLX-VLM) and M3 (Ollama); the
builders here raise ``OCRError`` until those modules exist.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from sdet_brain.config import OCRProvider, Settings
from sdet_brain.ocr.protocol import IOCREngine, OCRError

OCREngineBuilder = Callable[[Settings, str], IOCREngine]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCREngineSelection:
    """Result of ``get_ocr_engine`` — the active engine plus the chain it walked."""

    engine: IOCREngine
    provider: OCRProvider
    model: str
    fell_back: bool
    attempted: tuple[tuple[OCRProvider, str], ...]


def _select_prompt(settings: Settings, model: str) -> str:
    """Pick the right prompt for the requested model.

    DeepSeek-OCR variants understand the ``<|grounding|>`` token and
    benefit from the grounding prompt; everything else (Qwen2.5-VL,
    llava, etc.) expects plain instructions.
    """
    if "deepseek" in model.lower():
        return settings.ocr_grounding_prompt
    return settings.ocr_general_prompt


def _build_mlx_vlm(settings: Settings, model: str) -> IOCREngine:
    """Instantiate the MLX-VLM provider with the requested model.

    Returns an unloaded engine — model weights are pulled on the
    first ``extract_text`` call so factory bootstrap stays cheap.
    """
    try:
        from sdet_brain.ocr.mlx_vlm_provider import MLXVLMOCREngine
    except ImportError as exc:  # pragma: no cover - defensive
        raise OCRError(
            f"MLX-VLM provider module could not be imported (model={model!r}).",
        ) from exc
    return MLXVLMOCREngine(
        model_name=model,
        default_prompt=_select_prompt(settings, model),
        quality_min_chars=settings.ocr_quality_min_chars,
    )


def _build_ollama(settings: Settings, model: str) -> IOCREngine:
    """Instantiate the Ollama HTTP provider for the requested model tag."""
    try:
        from sdet_brain.ocr.ollama_provider import OllamaOCREngine
    except ImportError as exc:  # pragma: no cover - defensive
        raise OCRError(
            f"Ollama provider module could not be imported (model={model!r}).",
        ) from exc
    return OllamaOCREngine(
        model_name=model,
        default_prompt=_select_prompt(settings, model),
        quality_min_chars=settings.ocr_quality_min_chars,
        keep_alive=settings.ocr_keep_alive,
        timeout_seconds=settings.ocr_timeout_seconds,
    )


_BUILDERS: dict[OCRProvider, OCREngineBuilder] = {
    "mlx-vlm": _build_mlx_vlm,
    "ollama": _build_ollama,
}


def _resolve_chain(settings: Settings) -> list[tuple[OCRProvider, str]]:
    """Return the ordered ``(provider, model)`` chain for these settings."""
    mlx_link: tuple[OCRProvider, str] = ("mlx-vlm", settings.ocr_mlx_vlm_model)
    ollama_primary: tuple[OCRProvider, str] = (
        "ollama",
        settings.ocr_ollama_primary_model,
    )
    ollama_fallback: tuple[OCRProvider, str] | None = (
        ("ollama", settings.ocr_ollama_fallback_model)
        if settings.ocr_ollama_fallback_model
        else None
    )

    if settings.ocr_provider == "mlx-vlm":
        chain: list[tuple[OCRProvider, str]] = [mlx_link, ollama_primary]
        if ollama_fallback is not None:
            chain.append(ollama_fallback)
        return chain

    chain = [ollama_primary]
    if ollama_fallback is not None:
        chain.append(ollama_fallback)
    chain.append(mlx_link)
    return chain


def _try_build(
    provider: OCRProvider, model: str, settings: Settings
) -> IOCREngine | None:
    builder = _BUILDERS[provider]
    try:
        candidate = builder(settings, model)
    except OCRError as exc:
        logger.warning(
            "OCR provider %s (model=%s) could not be initialised: %s",
            provider,
            model,
            exc,
        )
        return None
    if not candidate.health_check():
        logger.warning(
            "OCR provider %s (model=%s) failed health_check; trying next link.",
            provider,
            model,
        )
        return None
    return candidate


_engine_lock = threading.Lock()
_cached_selection: OCREngineSelection | None = None


def get_ocr_engine(settings: Settings) -> OCREngineSelection:
    """Build the OCR engine, walking the fallback chain on failures.

    Subsequent calls return the cached selection until
    ``reset_ocr_engine`` is invoked. The cache is implicit: the
    first call's settings define the engine for the process lifetime.
    """
    global _cached_selection

    # Local capture so mypy's narrowing doesn't fight the double-checked
    # locking pattern (the global can be mutated by another thread).
    cached = _cached_selection
    if cached is not None:
        return cached

    with _engine_lock:
        cached = _cached_selection
        if cached is not None:
            return cached

        chain = _resolve_chain(settings)
        attempted: list[tuple[OCRProvider, str]] = []
        for provider, model in chain:
            attempted.append((provider, model))
            engine = _try_build(provider, model, settings)
            if engine is None:
                continue
            selection = OCREngineSelection(
                engine=engine,
                provider=provider,
                model=model,
                fell_back=len(attempted) > 1,
                attempted=tuple(attempted),
            )
            _cached_selection = selection
            return selection

        summary = ", ".join(f"{p}:{m}" for p, m in attempted) or "<empty chain>"
        raise OCRError(f"No OCR provider available. Tried: {summary}.")


def reset_ocr_engine() -> None:
    """Drop the cached selection so the next ``get_ocr_engine`` rebuilds.

    Tests use this in fixtures to isolate runs; production callers
    normally rely on the singleton across the process lifetime.
    """
    global _cached_selection
    with _engine_lock:
        _cached_selection = None
