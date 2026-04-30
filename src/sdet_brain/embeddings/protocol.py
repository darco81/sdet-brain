"""Embedding provider contract shared by MLX, Gemini, and future backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IEmbedder(Protocol):
    """Provider-agnostic embedding interface.

    Implementations encode text into dense float vectors. The protocol is
    runtime-checkable so the factory can verify a candidate object before
    wiring it up.
    """

    @property
    def vector_size(self) -> int:
        """Dimensionality of vectors produced by ``embed``."""
        ...

    @property
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. ``Qwen/Qwen3-Embedding-0.6B``)."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of strings.

        Returns one vector of length ``vector_size`` per input string,
        in the same order. Empty input returns an empty list.
        """
        ...

    def health_check(self) -> bool:
        """Return ``True`` if the provider can encode a probe input."""
        ...


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot produce vectors."""
