"""Google Gemini embedding provider used as the cloud fallback.

Wraps `google-genai`'s ``Client.models.embed_content`` with retries and
exponential backoff via ``tenacity``. Uses ``gemini-embedding-001`` and
requests ``output_dimensionality`` equal to the configured vector size
(1024 by default, matching the MLX collection) so the fallback writes
vectors compatible with the live Qdrant collection. A produced
dimensionality that disagrees with the configured size is a hard error
rather than a silent re-config, which previously let 768-dim vectors
land in a 1024-dim collection.
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

DEFAULT_MODEL: Final[str] = "gemini-embedding-001"
DEFAULT_VECTOR_SIZE: Final[int] = 1024
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
        from google.genai import types

        try:
            response = self._get_client().models.embed_content(
                model=self._model_name,
                contents=list(texts),
                config=types.EmbedContentConfig(output_dimensionality=self._vector_size),
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
                raise EmbeddingError(
                    f"Gemini returned {produced}-dim vectors but {self._vector_size} "
                    f"was configured (the live collection dimensionality). Set "
                    f"GEMINI_VECTOR_SIZE to match the collection, or recreate the "
                    f"collection — never write mismatched vectors."
                )
        return vectors

    def health_check(self) -> bool:
        try:
            output = self.embed(["sdet brain health probe"])
        except EmbeddingError:
            logger.warning("Gemini health check failed", exc_info=True)
            return False
        return len(output) == 1 and len(output[0]) == self._vector_size
