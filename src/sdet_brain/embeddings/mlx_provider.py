"""MLX-backed embedding provider for Apple Silicon.

Loads a HuggingFace embedding model through `mlx-embeddings` lazily so
process startup stays cheap. The first ``embed()`` call triggers the
download/compile and pays the warm-up cost.

Matryoshka Representation Learning (T4-01): some embedding models
(e.g. ``mlx-community/Qwen3-Embedding-8B-4bit-DWQ``) emit a higher
native dimension (4096) but were trained so that the leading slice
of the vector retains most of the semantic signal (~95% retention at
1024 dims). The provider exposes ``mrl_truncate_to`` so we can keep
the existing 1024-dim Qdrant collection while upgrading the model.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Final

from sdet_brain.embeddings.protocol import EmbeddingError

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_BATCH_SIZE: Final[int] = 32


class MLXEmbedder:
    """Embedding provider using MLX on the Apple Neural Engine / GPU.

    Parameters
    ----------
    model_name:
        HuggingFace model id loaded through ``mlx-embeddings``.
    vector_size:
        Expected output dimension after any MRL truncation. Used by
        :class:`QdrantStorage` to size the collection.
    batch_size:
        Number of texts encoded per ``mlx_embeddings.generate`` call.
    mrl_truncate_to:
        When set, the leading ``mrl_truncate_to`` floats of each
        embedding are kept and the rest dropped. ``None`` keeps the
        full native dimension.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        vector_size: int = 1024,
        batch_size: int = DEFAULT_BATCH_SIZE,
        mrl_truncate_to: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._vector_size = vector_size
        self._batch_size = batch_size
        self._mrl_truncate_to = mrl_truncate_to
        self._lock = Lock()
        # mlx-embeddings has no published py.typed stubs; we keep these as Any.
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    @property
    def vector_size(self) -> int:
        return self._vector_size

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        with self._lock:
            if self._model is None or self._tokenizer is None:
                try:
                    from mlx_embeddings import load
                except ImportError as exc:  # pragma: no cover - platform guard
                    raise EmbeddingError(
                        "mlx-embeddings is not installed. "
                        "On non-Apple platforms switch EMBEDDING_PROVIDER=gemini."
                    ) from exc
                logger.info("Loading MLX embedding model %s (lazy)", self._model_name)
                self._model, self._tokenizer = load(self._model_name)
        assert self._model is not None  # noqa: S101 - lock guarantees this
        assert self._tokenizer is not None  # noqa: S101
        return self._model, self._tokenizer

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model, tokenizer = self._ensure_loaded()
        try:
            from mlx_embeddings import generate
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError("mlx-embeddings is not installed") from exc

        results: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            output = generate(model, tokenizer, batch)
            # `generate` returns a transformers-style `BaseModelOutput`;
            # the dense per-text embedding lives on ``text_embeds`` (shape
            # ``(batch, vector_size)``). Falling back to
            # ``last_hidden_state`` mean-pooling would silently produce a
            # different vector space, so we treat a missing attribute as
            # an unsupported model.
            embeddings = getattr(output, "text_embeds", None)
            if embeddings is None:
                raise EmbeddingError(
                    f"MLX model {self._model_name!r} did not return `text_embeds`."
                )
            results.extend(embeddings.tolist())
        if not results:
            return results

        if self._mrl_truncate_to is not None:
            native_dim = len(results[0])
            if self._mrl_truncate_to > native_dim:
                raise EmbeddingError(
                    f"mrl_truncate_to={self._mrl_truncate_to} exceeds native "
                    f"embedding dim {native_dim} from {self._model_name!r}."
                )
            results = [vec[: self._mrl_truncate_to] for vec in results]

        produced = len(results[0])
        if produced != self._vector_size:
            logger.warning(
                "MLX produced vector_size=%d but configured %d; updating to match.",
                produced,
                self._vector_size,
            )
            self._vector_size = produced
        return results

    def health_check(self) -> bool:
        try:
            output = self.embed(["sdet brain health probe"])
        except Exception:
            logger.warning("MLX health check failed", exc_info=True)
            return False
        return len(output) == 1 and len(output[0]) == self._vector_size
