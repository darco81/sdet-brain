"""Unit tests for the four core MCP tools.

The tools are exercised against a live Qdrant container with a 16-dim
deterministic fake embedder. We seed a temporary collection per test
so search / list_sources / get_chunk_neighbors can be observed in
isolation.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from qdrant_client.models import PointStruct, SparseVector

from sdet_brain.config import Settings
from sdet_brain.embeddings.factory import EmbedderSelection
from sdet_brain.ingestion.pipeline import ingest_path as run_ingest
from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools.get_chunk_neighbors import get_chunk_neighbors
from sdet_brain.server.tools.ingest import ingest_path
from sdet_brain.server.tools.list_sources import list_sources
from sdet_brain.server.tools.search import search
from sdet_brain.storage.qdrant_client import QdrantStorage

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
VECTOR_SIZE = 16


def _qdrant_reachable(url: str) -> bool:
    try:
        return httpx.get(f"{url}/readyz", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module")
def qdrant_url() -> str:
    if not _qdrant_reachable(QDRANT_URL):
        pytest.skip(f"Qdrant not reachable at {QDRANT_URL}")
    return QDRANT_URL


@pytest.fixture
def storage(qdrant_url: str) -> Iterator[QdrantStorage]:
    with QdrantStorage(qdrant_url) as client:
        yield client


@pytest.fixture
def collection(storage: QdrantStorage) -> Iterator[str]:
    name = f"sdet_brain_tools_test_{os.getpid()}_{id(storage)}"
    storage.ensure_hybrid_collection(name, VECTOR_SIZE)
    yield name
    if storage.collection_exists(name):
        storage.client.delete_collection(collection_name=name)


class _FakeEmbedder:
    vector_size = VECTOR_SIZE
    model_name = "fake/deterministic"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [(((hash(text) >> i) & 0xFF) / 255.0) for i in range(VECTOR_SIZE)] for text in texts
        ]

    def health_check(self) -> bool:
        return True


@pytest.fixture
def state(storage: QdrantStorage) -> AppState:
    embedder = _FakeEmbedder()
    selection = EmbedderSelection(
        embedder=embedder,  # type: ignore[arg-type]
        provider="mlx",
        fell_back=False,
        attempted=("mlx",),
    )
    return AppState(settings=Settings(), storage=storage, selection=selection)


def _seed_chunks(
    storage: QdrantStorage,
    collection: str,
    source_path: str,
    chunk_count: int,
    source_type: str = "drafts",
) -> None:
    embedder = _FakeEmbedder()
    payloads = []
    for index in range(chunk_count):
        dense = embedder.embed([f"chunk-{index}"])[0]
        payloads.append(
            PointStruct(
                id=index + abs(hash(source_path)) % 1_000_000,
                vector={
                    "dense": dense,
                    "bm25": SparseVector(indices=[index, index + 1], values=[1.0, 0.5]),
                },
                payload={
                    "text": f"text body for chunk {index}",
                    "source_path": source_path,
                    "source_type": source_type,
                    "chunk_index": index,
                    "total_chunks": chunk_count,
                    "heading_path": f"Heading {index}",
                    "has_code": False,
                    "char_count": 12,
                    "token_estimate": 3,
                    "frontmatter": {},
                    "content_hash": f"hash-{source_path}-{chunk_count}",
                    "created_at": f"2026-04-30T12:0{index % 10}:00Z",
                },
            )
        )
    storage.upsert_points(collection, payloads)


def test_search_returns_markdown_hits(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunks(storage, collection, "/var/sdet-brain-fixtures/voice.md", 3)
    output = search(state, query="voice", limit=3, collection=collection)
    assert "Search results for" in output
    assert "voice.md" in output
    assert "score:" in output
    assert "text body for chunk" in output


def test_search_filters_by_source_type(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunks(storage, collection, "/var/sdet-brain-fixtures/draft.md", 2, source_type="drafts")
    _seed_chunks(
        state.storage, collection, "/var/sdet-brain-fixtures/article.md", 2, source_type="articles"
    )  # type: ignore[arg-type]
    drafts_only = search(
        state,
        query="draft",
        limit=10,
        source_type="drafts",
        collection=collection,
    )
    assert "draft.md" in drafts_only
    assert "article.md" not in drafts_only


def test_search_empty_query_raises(state: AppState, collection: str) -> None:
    from sdet_brain.server.tools._helpers import ToolError

    with pytest.raises(ToolError):
        search(state, query=" ", collection=collection)


def test_search_returns_no_matches_message_when_corpus_empty(
    state: AppState, collection: str
) -> None:
    output = search(state, query="anything", collection=collection)
    assert "No matches" in output


def test_list_sources_groups_by_path(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunks(storage, collection, "/var/sdet-brain-fixtures/a.md", 2)
    _seed_chunks(storage, collection, "/var/sdet-brain-fixtures/b.md", 5)
    output = list_sources(state, collection=collection)
    assert "2 files" in output
    assert "/var/sdet-brain-fixtures/a.md" in output
    assert "/var/sdet-brain-fixtures/b.md" in output
    assert "(2 chunks" in output
    assert "(5 chunks" in output


def test_list_sources_filter_returns_only_one_type(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunks(storage, collection, "/var/sdet-brain-fixtures/d.md", 2, source_type="drafts")
    _seed_chunks(
        storage, collection, "/var/sdet-brain-fixtures/p.md", 1, source_type="project-knowledge"
    )
    drafts = list_sources(state, source_type="drafts", collection=collection)
    assert "/var/sdet-brain-fixtures/d.md" in drafts
    assert "/var/sdet-brain-fixtures/p.md" not in drafts


def test_get_chunk_neighbors_returns_window(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunks(storage, collection, "/var/sdet-brain-fixtures/win.md", 7)
    output = get_chunk_neighbors(
        state,
        source_path="/var/sdet-brain-fixtures/win.md",
        chunk_index=3,
        window=2,
        collection=collection,
    )
    # Range 1..5 inclusive, target chunk 3.
    for index in range(1, 6):
        assert f"chunk {index}/7" in output
    assert "chunk 0/" not in output
    assert "chunk 6/" not in output
    assert "(target)" in output


def test_get_chunk_neighbors_clamps_at_zero(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunks(storage, collection, "/var/sdet-brain-fixtures/clamp.md", 4)
    output = get_chunk_neighbors(
        state,
        source_path="/var/sdet-brain-fixtures/clamp.md",
        chunk_index=0,
        window=2,
        collection=collection,
    )
    # Range 0..2, no negative chunks.
    assert "chunk 0/4" in output
    assert "chunk 2/4" in output
    assert "chunk -" not in output


def test_get_chunk_neighbors_clamps_at_total(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunks(storage, collection, "/var/sdet-brain-fixtures/clamp2.md", 4)
    output = get_chunk_neighbors(
        state,
        source_path="/var/sdet-brain-fixtures/clamp2.md",
        chunk_index=3,
        window=5,
        collection=collection,
    )
    # Range 0..3 (clamped to total-1).
    assert "chunk 0/4" in output
    assert "chunk 3/4" in output
    # Window asked for 8, but the file only has 4 chunks; nothing past 3.
    assert "chunk 4/4" not in output


def test_ingest_tool_routes_to_pipeline(state: AppState, collection: str, tmp_path: Path) -> None:
    file_path = tmp_path / "tool_ingest.md"
    file_path.write_text("# Heading\n\nbody " + ("alpha " * 30), encoding="utf-8")
    output = ingest_path(state, path=str(file_path), collection=collection)
    assert "Ingest summary" in output
    assert "Files processed: **1**" in output
    # Re-run via the pipeline to confirm the chunks landed.
    second = run_ingest(file_path, state.storage, state.embedder, collection=collection)  # type: ignore[arg-type]
    assert second.files_skipped == 1


def test_ingest_tool_rejects_missing_path(state: AppState) -> None:
    from sdet_brain.server.tools._helpers import ToolError

    with pytest.raises(ToolError):
        ingest_path(state, path="/var/sdet-brain-fixtures/does-not-exist.md")
