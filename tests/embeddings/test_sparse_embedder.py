"""Sparse BM25 embedder factory tests.

The real ``FastembedBM25`` downloads BM25 vocabulary on first call and
spins up an ONNX session per instance. Tests exercise the factory's
caching contract without touching the network: ``_ensure_loaded`` is
never invoked, so no ONNX weights are pulled.

These tests pin the singleton invariant that prevents the production
memory regression where 7 module-level ``_SPARSE`` caches plus a
non-caching factory spawned 27+ ``Loading sparse embedder`` events
over a single 45h process lifetime, growing the Python heap unbounded.
"""

from __future__ import annotations

import pytest

from sdet_brain.embeddings import sparse_embedder as sparse_embedder_module
from sdet_brain.embeddings.sparse_embedder import (
    DEFAULT_BM25_MODEL,
    FastembedBM25,
    get_sparse_embedder,
)


@pytest.fixture(autouse=True)
def _clear_factory_cache() -> None:
    """Reset the factory cache between tests so each starts cold."""
    cached_builder = getattr(sparse_embedder_module, "_build_sparse_embedder", None)
    cache_clear = getattr(cached_builder, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def test_factory_returns_same_instance_for_default_model() -> None:
    """Repeated calls with the default model must return the same wrapper."""
    first = get_sparse_embedder()
    second = get_sparse_embedder()
    assert first is second


def test_factory_returns_same_instance_for_explicit_default() -> None:
    """``None`` and the default model id collapse to the same cache entry."""
    implicit = get_sparse_embedder()
    explicit = get_sparse_embedder(DEFAULT_BM25_MODEL)
    assert implicit is explicit


def test_factory_returns_distinct_instance_for_different_model() -> None:
    """Different model ids must yield distinct cached instances."""
    default = get_sparse_embedder(DEFAULT_BM25_MODEL)
    alternative = get_sparse_embedder("Qdrant/bm42-all-minilm-l6-v2-attentions")
    assert default is not alternative


def test_factory_constructs_underlying_class_once_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """100 calls with the same model id must allocate ``FastembedBM25`` once.

    Regression test: prior to caching, production logs showed 27
    ``Loading sparse embedder`` events in a single process. Each
    construction allocated its own ONNX runtime and vocabulary state,
    accumulating ~hundreds of MB of unreclaimed heap.
    """
    construct_count = 0
    original_init = FastembedBM25.__init__

    def counting_init(self: FastembedBM25, *args: object, **kwargs: object) -> None:
        nonlocal construct_count
        construct_count += 1
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(FastembedBM25, "__init__", counting_init)

    for _ in range(100):
        get_sparse_embedder()

    assert construct_count == 1
