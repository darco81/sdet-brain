"""Google Gemini embedding provider used as the cloud fallback.

Wraps `google-genai`'s ``Client.models.embed_content`` with retries and
exponential backoff via ``tenacity``. Gemini Flash text-embedding-004
returns 768-dim vectors - mismatched with the MLX 1024-dim default, so
the application configures Qdrant per-environment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sdet_brain.embeddings.protocol import EmbeddingError

if TYPE_CHECKING:
    from google.genai import Client

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "text-embedding-004"
DEFAULT_VECTOR_SIZE: Final[int] = 768
MAX_RETRY_ATTEMPTS: Final[int] = 4


class GeminiTransientError(RuntimeError):
    """Wraps Gemini API errors that warrant a retry (429 / 5xx / network)."""


def _new_retrying() -> Retrying:
    return Retrying(
        reraise=True,
        retry=retry_if_exception_type(GeminiTransientError),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
    )


class GeminiEmbedder:
    """Cloud-based embeddings via Google's `google-genai` SDK."""

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY is required for the Gemini provider.")
        self._api_key = api_key
        self._model_name = model_name
        self._vector_size = vector_size
        self._client: Client | None = None

    @property
    def vector_size(self) -> int:
        return self._vector_size

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_client(self) -> Any:
        if self._client is None:
            from google.genai import Client

            self._client = Client(api_key=self._api_key)
        return self._client

    def _embed_once(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._get_client().models.embed_content(
                model=self._model_name,
                contents=list(texts),
            )
        except Exception as exc:
            message = str(exc)
            transient = any(token in message for token in ("429", "503", "timeout", "RST_STREAM"))
            if transient:
                raise GeminiTransientError(message) from exc
            raise EmbeddingError(f"Gemini embedding failed: {message}") from exc

        if response.embeddings is None:
            raise EmbeddingError("Gemini returned no embeddings.")
        vectors: list[list[float]] = []
        for item in response.embeddings:
            values = item.values
            if values is None:
                raise EmbeddingError("Gemini returned an embedding without values.")
            vectors.append(list(values))
        return vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            for attempt in _new_retrying():
                with attempt:
                    vectors = self._embed_once(texts)
        except RetryError as exc:  # pragma: no cover - reraise=True keeps inner type
            raise EmbeddingError("Gemini retries exhausted.") from exc

        if vectors:
            produced = len(vectors[0])
            if produced != self._vector_size:
                logger.warning(
                    "Gemini produced vector_size=%d but configured %d; updating to match.",
                    produced,
                    self._vector_size,
                )
                self._vector_size = produced
        return vectors

    def health_check(self) -> bool:
        try:
            output = self.embed(["sdet brain health probe"])
        except EmbeddingError:
            logger.warning("Gemini health check failed", exc_info=True)
            return False
        return len(output) == 1 and len(output[0]) == self._vector_size
