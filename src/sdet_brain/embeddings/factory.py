"""Provider selection with health-check based auto-fallback.

`get_embedder(settings)` reads `EMBEDDING_PROVIDER` and returns an
`IEmbedder` honouring the user's preference. If the primary provider's
`health_check()` fails (e.g. MLX unavailable on a Linux VPS), we fall
back to the other provider when its credentials/runtime are present.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from sdet_brain.config import EmbeddingProvider, Settings
from sdet_brain.embeddings.gemini_provider import GeminiEmbedder
from sdet_brain.embeddings.mlx_provider import MLXEmbedder
from sdet_brain.embeddings.protocol import EmbeddingError, IEmbedder

EmbedderBuilder = Callable[[Settings], IEmbedder]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedderSelection:
    """Result of `get_embedder` - the active provider plus the chain it walked."""

    embedder: IEmbedder
    provider: EmbeddingProvider
    fell_back: bool
    attempted: tuple[EmbeddingProvider, ...]


def _build_mlx(settings: Settings) -> IEmbedder:
    return MLXEmbedder(
        model_name=settings.mlx_model,
        vector_size=settings.mlx_vector_size,
        mrl_truncate_to=settings.mlx_mrl_truncate_to,
    )


def _build_gemini(settings: Settings) -> IEmbedder:
    if not settings.gemini_api_key:
        raise EmbeddingError(
            "Gemini provider requested but GEMINI_API_KEY is not set.",
        )
    return GeminiEmbedder(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_embedding_model,
        vector_size=settings.gemini_vector_size,
    )


_BUILDERS: dict[EmbeddingProvider, EmbedderBuilder] = {
    "mlx": _build_mlx,
    "gemini": _build_gemini,
}


def _try_build(
    provider: EmbeddingProvider, settings: Settings
) -> IEmbedder | None:
    builder = _BUILDERS[provider]
    try:
        candidate = builder(settings)
    except EmbeddingError as exc:
        logger.warning("Provider %s could not be initialised: %s", provider, exc)
        return None
    if not candidate.health_check():
        logger.warning("Provider %s failed health_check; will try fallback.", provider)
        return None
    return candidate


def get_embedder(settings: Settings) -> EmbedderSelection:
    """Build an embedding provider, falling back to the alternate on failure."""
    primary = settings.embedding_provider
    secondary: EmbeddingProvider = "gemini" if primary == "mlx" else "mlx"

    attempted: list[EmbeddingProvider] = [primary]
    embedder = _try_build(primary, settings)
    if embedder is not None:
        return EmbedderSelection(
            embedder=embedder,
            provider=primary,
            fell_back=False,
            attempted=tuple(attempted),
        )

    logger.info("Falling back from %s to %s", primary, secondary)
    attempted.append(secondary)
    embedder = _try_build(secondary, settings)
    if embedder is not None:
        return EmbedderSelection(
            embedder=embedder,
            provider=secondary,
            fell_back=True,
            attempted=tuple(attempted),
        )

    raise EmbeddingError(
        f"No embedding provider available. Tried: {', '.join(attempted)}.",
    )
