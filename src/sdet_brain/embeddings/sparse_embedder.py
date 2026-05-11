"""BM25 sparse-vector embedder backed by fastembed (T2-03).

Hybrid search pairs a dense bi-encoder (semantic) with a sparse term
matcher (BM25) and fuses the two ranked lists with Reciprocal Rank
Fusion. The dense path catches synonyms; the sparse path keeps exact
keyword matches honest. Without it, queries like ``"WCAG 2.2 AA"`` get
beaten by semantically similar but term-irrelevant chunks.

Default model is ``Qdrant/bm25`` (built-in tokenizer). Lazy-loads on
first call so process startup stays cheap.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from threading import Lock
from typing import Any, Final, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_BM25_MODEL: Final[str] = "Qdrant/bm25"


@dataclass(frozen=True)
class SparseVector:
    """A sparse vector as Qdrant expects it: parallel index/value lists."""

    indices: list[int]
    values: list[float]


class SparseEmbeddingError(RuntimeError):
    """Raised when the sparse embedder cannot produce vectors."""


@runtime_checkable
class ISparseEmbedder(Protocol):
    """Provider-agnostic sparse embedding contract."""

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    def embed(self, texts: Sequence[str]) -> list[SparseVector]:
        """Encode a batch of strings into sparse vectors."""
        ...

    def health_check(self) -> bool:
        """Return ``True`` if the encoder can score a probe input."""
        ...


class FastembedBM25:
    """`fastembed.SparseTextEmbedding` wrapper.

    Lazy-loads the BM25 model (a small ONNX/Rust tokenizer; not a
    neural model so the load is fast and cheap). Thread-safe lazy
    init via a lock.
    """

    def __init__(self, model_name: str = DEFAULT_BM25_MODEL) -> None:
        self._model_name = model_name
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
                    from fastembed import SparseTextEmbedding
                except ImportError as exc:  # pragma: no cover - dep guard
                    raise SparseEmbeddingError(
                        "fastembed is not installed; cannot load BM25 embedder."
                    ) from exc
                logger.info(
                    "Loading sparse embedder %s (lazy)", self._model_name
                )
                self._encoder = SparseTextEmbedding(model_name=self._model_name)
        assert self._encoder is not None  # noqa: S101 - lock guarantees this
        return self._encoder

    def embed(self, texts: Sequence[str]) -> list[SparseVector]:
        if not texts:
            return []
        encoder = self._ensure_loaded()
        try:
            raw = list(encoder.embed(list(texts)))
        except Exception as exc:
            raise SparseEmbeddingError(f"BM25 embedding failed: {exc}") from exc
        return [
            SparseVector(
                indices=[int(idx) for idx in vec.indices],
                values=[float(val) for val in vec.values],
            )
            for vec in raw
        ]

    def health_check(self) -> bool:
        try:
            self.embed(["health probe"])
        except Exception:
            logger.warning("Sparse embedder health check failed", exc_info=True)
            return False
        return True


@cache
def _build_sparse_embedder(model_name: str) -> FastembedBM25:
    return FastembedBM25(model_name=model_name)


def get_sparse_embedder(model_name: str | None = None) -> FastembedBM25:
    """Return the process-wide :class:`FastembedBM25` for ``model_name``.

    Cached per resolved model id so every caller - and every
    module-level ``_SPARSE`` helper across the server - shares one
    wrapper. Without this, a non-caching factory plus seven
    independent module singletons spawned a fresh ``FastembedBM25``
    on each cold tool/route, each allocating its own ONNX session
    and BM25 vocabulary; production logs showed 27+ such loads in a
    single 45h process, growing the heap unbounded.

    ``None`` and the configured default resolve to the same cache
    entry so explicit-vs-implicit callers share the same wrapper.
    """
    return _build_sparse_embedder(model_name or DEFAULT_BM25_MODEL)
