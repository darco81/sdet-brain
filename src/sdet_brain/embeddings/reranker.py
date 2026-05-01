"""Cross-encoder reranker for search results (T2-04).

A reranker reorders an over-fetched candidate set produced by the
bi-encoder embedder so the most relevant chunks bubble to the top.
The bi-encoder is fast but coarse; the cross-encoder is slow but
accurate. The retrieve-30, rerank, return-5 pattern keeps both fast
and accurate.

Default model: ``jinaai/jina-reranker-v2-base-multilingual`` (PL+EN
content-match). The fastembed registry does NOT include
``BAAI/bge-reranker-v2-m3`` (preferred in the original spec); the
multilingual Jina model is the closest substitute. Override via
``RERANK_MODEL`` env var.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Final, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "jinaai/jina-reranker-v2-base-multilingual"
TEST_FALLBACK_MODEL: Final[str] = "Xenova/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K_RETRIEVE: Final[int] = 30
DEFAULT_TOP_K_RETURN: Final[int] = 5


class RerankerError(RuntimeError):
    """Raised when a reranker cannot score the candidate set."""


@dataclass(frozen=True)
class RerankCandidate:
    """A single candidate the reranker scores.

    The reranker is index-agnostic: callers pass an opaque ``payload``
    (the original Qdrant point, search result, etc.) which the
    reranker preserves verbatim in the returned :class:`RerankResult`.
    Only ``text`` is fed to the cross-encoder model.
    """

    text: str
    payload: Any = None


@dataclass(frozen=True)
class RerankResult:
    """A reranked candidate with its cross-encoder score and original payload."""

    text: str
    score: float
    payload: Any


@runtime_checkable
class IReranker(Protocol):
    """Provider-agnostic reranker contract."""

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Score the candidates against the query and return top-K reordered."""
        ...

    def health_check(self) -> bool:
        """Return ``True`` if the model can score a probe pair."""
        ...


class FastembedReranker:
    """Cross-encoder reranker backed by fastembed's TextCrossEncoder.

    Lazy-loads the ONNX model on the first ``rerank()`` call so process
    startup stays cheap. Thread-safe lazy load via a lock.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k_default: int = DEFAULT_TOP_K_RETURN,
    ) -> None:
        self._model_name = model_name
        self._top_k_default = top_k_default
        self._lock = Lock()
        self._encoder: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        with self._lock:
            if self._encoder is None:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder
                except ImportError as exc:  # pragma: no cover - dep guard
                    raise RerankerError(
                        "fastembed is not installed; cannot load reranker."
                    ) from exc
                logger.info(
                    "Loading cross-encoder reranker %s (lazy)", self._model_name
                )
                self._encoder = TextCrossEncoder(model_name=self._model_name)
        assert self._encoder is not None  # noqa: S101 - lock guarantees this
        return self._encoder

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        if not candidates:
            return []
        if not query.strip():
            raise RerankerError("query must not be empty")

        encoder = self._ensure_loaded()
        documents = [c.text for c in candidates]
        try:
            raw_scores = list(encoder.rerank(query, documents))
        except Exception as exc:
            raise RerankerError(f"Reranker failed: {exc}") from exc

        if len(raw_scores) != len(candidates):
            raise RerankerError(
                f"Reranker returned {len(raw_scores)} scores for "
                f"{len(candidates)} candidates."
            )

        results = [
            RerankResult(text=c.text, score=float(s), payload=c.payload)
            for c, s in zip(candidates, raw_scores, strict=True)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: top_k if top_k is not None else self._top_k_default]

    def health_check(self) -> bool:
        try:
            self.rerank(
                "health probe query",
                [RerankCandidate(text="health probe document")],
                top_k=1,
            )
        except Exception:
            logger.warning("Reranker health check failed", exc_info=True)
            return False
        return True


def get_reranker(model_name: str | None = None) -> FastembedReranker:
    """Build a :class:`FastembedReranker` honouring the env-var defaults."""
    return FastembedReranker(model_name=model_name or DEFAULT_MODEL)
