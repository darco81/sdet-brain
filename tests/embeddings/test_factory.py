"""Factory fallback logic - exercised with in-process fakes (no MLX/Gemini)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from sdet_brain.config import Settings
from sdet_brain.embeddings import factory as factory_module
from sdet_brain.embeddings.factory import EmbedderSelection, get_embedder
from sdet_brain.embeddings.protocol import EmbeddingError


class _StubEmbedder:
    def __init__(self, *, model_name: str, healthy: bool, vector_size: int = 8) -> None:
        self._model_name = model_name
        self._healthy = healthy
        self._vector_size = vector_size

    @property
    def vector_size(self) -> int:
        return self._vector_size

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * self._vector_size for _ in texts]

    def health_check(self) -> bool:
        return self._healthy


@pytest.fixture
def patched_builders(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    state: dict[str, Any] = {"mlx_healthy": True, "gemini_healthy": True}

    def build_mlx(_: Settings) -> _StubEmbedder:
        return _StubEmbedder(model_name="stub-mlx", healthy=state["mlx_healthy"])

    def build_gemini(_: Settings) -> _StubEmbedder:
        return _StubEmbedder(model_name="stub-gemini", healthy=state["gemini_healthy"])

    monkeypatch.setitem(factory_module._BUILDERS, "mlx", build_mlx)
    monkeypatch.setitem(factory_module._BUILDERS, "gemini", build_gemini)
    yield state


def _settings(provider: str = "mlx") -> Settings:
    return Settings(embedding_provider=provider, gemini_api_key="test-key")  # type: ignore[arg-type]


def test_primary_succeeds_when_healthy(patched_builders: dict[str, Any]) -> None:
    selection = get_embedder(_settings("mlx"))
    assert isinstance(selection, EmbedderSelection)
    assert selection.provider == "mlx"
    assert selection.fell_back is False
    assert selection.attempted == ("mlx",)
    assert selection.embedder.model_name == "stub-mlx"


def test_falls_back_when_primary_unhealthy(patched_builders: dict[str, Any]) -> None:
    patched_builders["mlx_healthy"] = False
    selection = get_embedder(_settings("mlx"))
    assert selection.provider == "gemini"
    assert selection.fell_back is True
    assert selection.attempted == ("mlx", "gemini")
    assert selection.embedder.model_name == "stub-gemini"


def test_raises_when_both_providers_unhealthy(
    patched_builders: dict[str, Any],
) -> None:
    patched_builders["mlx_healthy"] = False
    patched_builders["gemini_healthy"] = False
    with pytest.raises(EmbeddingError) as excinfo:
        get_embedder(_settings("mlx"))
    assert "No embedding provider available" in str(excinfo.value)
    assert "mlx" in str(excinfo.value)
    assert "gemini" in str(excinfo.value)


def test_gemini_first_falls_back_to_mlx_when_unhealthy(
    patched_builders: dict[str, Any],
) -> None:
    patched_builders["gemini_healthy"] = False
    selection = get_embedder(_settings("gemini"))
    assert selection.provider == "mlx"
    assert selection.fell_back is True
    assert selection.attempted == ("gemini", "mlx")
