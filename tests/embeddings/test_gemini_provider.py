"""Gemini provider unit tests using fakes (no live API calls)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from sdet_brain.embeddings import gemini_provider
from sdet_brain.embeddings.gemini_provider import (
    GeminiEmbedder,
    GeminiTransientError,
)
from sdet_brain.embeddings.protocol import EmbeddingError


@dataclass
class _FakeEmbeddingItem:
    values: list[float] | None


@dataclass
class _FakeResponse:
    embeddings: list[_FakeEmbeddingItem] | None


class _FakeModels:
    def __init__(
        self,
        responses: list[_FakeResponse | Exception],
    ) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def embed_content(self, *, model: str, contents: list[str]) -> _FakeResponse:
        self.calls.append({"model": model, "contents": list(contents)})
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self.models = _FakeModels(responses)


def _make_embedder(monkeypatch: pytest.MonkeyPatch, fake_client: _FakeClient) -> GeminiEmbedder:
    embedder = GeminiEmbedder(api_key="test-key")
    monkeypatch.setattr(embedder, "_get_client", lambda: fake_client)
    return embedder


def test_constructor_rejects_empty_api_key() -> None:
    with pytest.raises(EmbeddingError):
        GeminiEmbedder(api_key="")


def test_embed_returns_vectors_for_each_text(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        [
            _FakeResponse(
                embeddings=[
                    _FakeEmbeddingItem(values=[0.1, 0.2, 0.3]),
                    _FakeEmbeddingItem(values=[0.4, 0.5, 0.6]),
                ]
            )
        ]
    )
    embedder = _make_embedder(monkeypatch, fake)
    result = embedder.embed(["alpha", "beta"])
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert embedder.vector_size == 3
    assert fake.models.calls[0]["contents"] == ["alpha", "beta"]


def test_embed_retries_on_transient_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient(
        [
            RuntimeError("503 Service Unavailable"),
            RuntimeError("429 Too Many Requests"),
            _FakeResponse(embeddings=[_FakeEmbeddingItem(values=[0.0, 1.0])]),
        ]
    )
    # Speed up retries.
    monkeypatch.setattr(gemini_provider, "MAX_RETRY_ATTEMPTS", 4)
    embedder = _make_embedder(monkeypatch, fake)
    result = embedder.embed(["x"])
    assert result == [[0.0, 1.0]]
    assert len(fake.models.calls) == 3


def test_embed_raises_on_permanent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient([RuntimeError("400 Bad Request")])
    embedder = _make_embedder(monkeypatch, fake)
    with pytest.raises(EmbeddingError):
        embedder.embed(["x"])


def test_health_check_true_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        [_FakeResponse(embeddings=[_FakeEmbeddingItem(values=[0.0, 0.0, 0.0])])]
    )
    embedder = _make_embedder(monkeypatch, fake)
    assert embedder.health_check() is True


def test_health_check_false_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient([RuntimeError("400 Bad Request")])
    embedder = _make_embedder(monkeypatch, fake)
    assert embedder.health_check() is False


def test_transient_error_is_retried_marker() -> None:
    """Sanity-check that GeminiTransientError is a retriable exception type."""
    assert issubclass(GeminiTransientError, RuntimeError)
