"""End-to-end ingestion tests against a live Qdrant container.

The embedder is replaced with a deterministic 16-dim fake so the tests
stay milliseconds fast and reproducible. Storage is real - we already
keep a Qdrant container up via docker-compose for T1-02.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from sdet_brain.ingestion.pipeline import IngestStats, ingest_path
from sdet_brain.ingestion.source_classifier import SourceConfig
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
    name = f"sdet_brain_pipeline_test_{os.getpid()}_{id(storage)}"
    storage.ensure_hybrid_collection(name, VECTOR_SIZE)
    yield name
    if storage.collection_exists(name):
        storage.client.delete_collection(collection_name=name)


class _FakeEmbedder:
    """Deterministic 16-dim fake (Settings-compatible)."""

    vector_size = VECTOR_SIZE
    model_name = "fake/deterministic"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [
            [(((hash(text) >> i) & 0xFF) / 255.0) for i in range(VECTOR_SIZE)] for text in texts
        ]

    def health_check(self) -> bool:
        return True


class _FakeSparseEmbedder:
    """Deterministic sparse fake so pipeline tests stay offline."""

    model_name = "fake/sparse"

    def embed(self, texts):  # type: ignore[no-untyped-def]
        from sdet_brain.embeddings.sparse_embedder import SparseVector

        out = []
        for text in texts:
            base = abs(hash(text))
            out.append(
                SparseVector(
                    indices=[base % 1024, (base + 1) % 1024],
                    values=[1.0, 0.5],
                )
            )
        return out

    def health_check(self) -> bool:
        return True


@pytest.fixture
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


@pytest.fixture(autouse=True)
def _patch_sparse_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the production sparse embedder factory with a fake.

    Pipeline ingest defaults to ``get_sparse_embedder()`` when no
    explicit one is passed; that triggers a real fastembed download.
    Tests must stay offline, so we monkeypatch the factory to return
    the deterministic fake instead.
    """
    monkeypatch.setattr(
        "sdet_brain.ingestion.pipeline.get_sparse_embedder",
        _FakeSparseEmbedder,
    )


def _write_md(tmp_path: Path, name: str, body: str) -> Path:
    file_path = tmp_path / name
    file_path.write_text(textwrap.dedent(body), encoding="utf-8")
    return file_path


def test_ingest_single_file_creates_chunks(
    tmp_path: Path,
    storage: QdrantStorage,
    collection: str,
    fake_embedder: _FakeEmbedder,
) -> None:
    file_path = _write_md(
        tmp_path,
        "doc.md",
        """\
        ---
        title: T1-05 fixture
        ---

        # Heading

        Body paragraph for ingestion. %s
        """
        % ("alpha " * 60),
    )
    stats = ingest_path(
        file_path,
        storage,
        fake_embedder,
        collection=collection,
    )
    assert stats.files_processed == 1
    assert stats.chunks_created >= 1
    assert stats.errors == []
    assert storage.count(collection) == stats.chunks_created


def test_ingest_directory_processes_multiple_files(
    tmp_path: Path,
    storage: QdrantStorage,
    collection: str,
    fake_embedder: _FakeEmbedder,
) -> None:
    for i in range(3):
        _write_md(tmp_path, f"doc-{i}.md", f"# File {i}\n\nbody {'word ' * 30}")
    stats = ingest_path(tmp_path, storage, fake_embedder, collection=collection)
    assert stats.files_processed == 3
    assert stats.chunks_created >= 3
    assert storage.count(collection) == stats.chunks_created


def test_reingest_unchanged_file_is_cache_hit(
    tmp_path: Path,
    storage: QdrantStorage,
    collection: str,
    fake_embedder: _FakeEmbedder,
) -> None:
    file_path = _write_md(tmp_path, "doc.md", "# H\n\nshort body\n")
    first = ingest_path(file_path, storage, fake_embedder, collection=collection)
    assert first.files_processed == 1

    second = ingest_path(file_path, storage, fake_embedder, collection=collection)
    assert second.files_processed == 0
    assert second.files_skipped == 1
    assert second.chunks_created == 0


def test_reingest_modified_file_replaces_chunks(
    tmp_path: Path,
    storage: QdrantStorage,
    collection: str,
    fake_embedder: _FakeEmbedder,
) -> None:
    file_path = _write_md(tmp_path, "doc.md", "# Original\n\nfirst body\n")
    ingest_path(file_path, storage, fake_embedder, collection=collection)
    initial_count = storage.count(collection)

    file_path.write_text(
        "# Updated\n\n" + ("rewritten body. " * 80),
        encoding="utf-8",
    )
    stats = ingest_path(file_path, storage, fake_embedder, collection=collection)
    assert stats.files_processed == 1
    assert stats.chunks_replaced == 1
    # Total chunks may be larger or smaller than before, but the old
    # ones are gone - so the count equals exactly the new chunk total.
    assert storage.count(collection) == stats.chunks_created
    assert storage.count(collection) >= initial_count or stats.chunks_created >= 1


def test_force_flag_re_embeds_even_when_hash_matches(
    tmp_path: Path,
    storage: QdrantStorage,
    collection: str,
    fake_embedder: _FakeEmbedder,
) -> None:
    file_path = _write_md(tmp_path, "doc.md", "# H\n\nidentical body\n")
    ingest_path(file_path, storage, fake_embedder, collection=collection)

    forced = ingest_path(
        file_path,
        storage,
        fake_embedder,
        collection=collection,
        force_reindex=True,
    )
    assert forced.files_processed == 1
    assert forced.files_skipped == 0
    assert forced.chunks_replaced == 1


def test_source_classifier_tags_payload(
    tmp_path: Path,
    storage: QdrantStorage,
    collection: str,
    fake_embedder: _FakeEmbedder,
) -> None:
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()
    _write_md(drafts_dir, "voice-sample.md", "# Voice\n\nsample body\n")
    config = SourceConfig(drafts_dirs=(drafts_dir,))
    ingest_path(
        drafts_dir,
        storage,
        fake_embedder,
        source_config=config,
        collection=collection,
    )
    points, _ = storage.client.scroll(
        collection_name=collection, limit=1, with_payload=True, with_vectors=False
    )
    assert points
    payload = points[0].payload or {}
    assert payload.get("source_type") == "drafts"


def test_stats_summary_contains_key_counts() -> None:
    stats = IngestStats(
        files_processed=2,
        files_skipped=1,
        chunks_created=10,
        chunks_replaced=3,
    )
    summary = stats.summary()
    assert "Processed 2 files" in summary
    assert "created 10 chunks" in summary
    assert "skipped 1 files" in summary
    assert "replaced 3 chunks" in summary
