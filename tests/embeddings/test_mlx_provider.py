"""MLX provider tests - skipped on non-Apple-Silicon runners."""

from __future__ import annotations

import importlib.util
import platform
import sys

import pytest

mlx_available = (
    sys.platform == "darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx_embeddings") is not None
)

pytestmark = pytest.mark.skipif(
    not mlx_available,
    reason="mlx-embeddings only runs on Apple Silicon (darwin/arm64).",
)


def test_mlx_embedder_lazy_does_not_load_on_construction() -> None:
    from sdet_brain.embeddings.mlx_provider import MLXEmbedder

    embedder = MLXEmbedder(model_name="Qwen/Qwen3-Embedding-0.6B", vector_size=1024)
    assert embedder.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert embedder.vector_size == 1024
    # `_model` is private but the lazy contract is "do not load on init".
    assert embedder._model is None
    assert embedder._tokenizer is None


def test_mlx_embed_empty_input_returns_empty_list() -> None:
    from sdet_brain.embeddings.mlx_provider import MLXEmbedder

    embedder = MLXEmbedder()
    assert embedder.embed([]) == []
