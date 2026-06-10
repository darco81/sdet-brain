"""Watcher behaviour tests using simulated watchdog events."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from watchdog.events import (
    DirCreatedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
)

from sdet_brain.ingestion import watcher as watcher_module
from sdet_brain.ingestion.pipeline import IngestStats
from sdet_brain.ingestion.watcher import BrainWatcher, is_relevant_path
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
    name = f"sdet_brain_watcher_test_{os.getpid()}_{id(storage)}"
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


class _FakeSparseEmbedder:
    """Offline sparse stub matching the watcher pipeline contract."""

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


@pytest.fixture(autouse=True)
def _patch_sparse_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sdet_brain.ingestion.pipeline.get_sparse_embedder",
        _FakeSparseEmbedder,
    )


def _make_watcher(storage: QdrantStorage, collection: str, paths: list[Path]) -> BrainWatcher:
    return BrainWatcher(
        paths,
        storage,
        _FakeEmbedder(),
        collection=collection,
        debounce_ms=200,
    )


# ----------------------------------------------------------------------
# is_relevant_path
# ----------------------------------------------------------------------


def test_is_relevant_path_accepts_markdown() -> None:
    assert is_relevant_path(Path("/var/sdet-brain/notes/readme.md"))


def test_is_relevant_path_rejects_non_markdown() -> None:
    assert not is_relevant_path(Path("/var/sdet-brain/notes/.DS_Store"))
    assert not is_relevant_path(Path("/var/sdet-brain/notes/readme.txt"))


def test_is_relevant_path_rejects_hidden_dirs() -> None:
    assert not is_relevant_path(Path("/var/sdet-brain/.git/HEAD.md"))


def test_is_relevant_path_rejects_node_modules() -> None:
    assert not is_relevant_path(Path("/var/sdet-brain/node_modules/foo/readme.md"))


# ----------------------------------------------------------------------
# debounce + delete handling using simulated events (no Qdrant required)
# ----------------------------------------------------------------------


def test_debouncing_collapses_rapid_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Five rapid `on_modified` events must result in one ingest call."""
    file_path = tmp_path / "doc.md"
    file_path.write_text("# Heading\n\nbody\n", encoding="utf-8")

    fake_storage = MagicMock(spec=QdrantStorage)
    fake_embedder = _FakeEmbedder()
    watcher = BrainWatcher(
        [tmp_path],
        fake_storage,
        fake_embedder,
        debounce_ms=50,
    )

    ingest_calls: list[Path] = []

    def fake_ingest(path, *args, **kwargs):
        ingest_calls.append(path)
        return IngestStats(files_processed=1, chunks_created=2)

    monkeypatch.setattr(watcher_module, "ingest_path", fake_ingest)

    for _ in range(5):
        watcher.on_modified(FileModifiedEvent(str(file_path)))

    # Pretend enough time has passed and run the worker once.
    time = watcher_module.time  # alias for monkeypatching
    monkeypatch.setattr(time, "monotonic", lambda: 1e9)
    watcher._process_due()

    assert ingest_calls == [file_path.resolve()]
    assert watcher.stats.files_ingested == 1
    assert watcher.stats.chunks_created == 2


def test_delete_event_triggers_delete_by_filter(tmp_path: Path) -> None:
    fake_storage = MagicMock(spec=QdrantStorage)
    watcher = BrainWatcher(
        [tmp_path],
        fake_storage,
        _FakeEmbedder(),
        debounce_ms=200,
    )
    md_file = tmp_path / "removed.md"
    watcher.on_deleted(FileDeletedEvent(str(md_file)))

    assert fake_storage.delete_by_filter.called
    args, _ = fake_storage.delete_by_filter.call_args
    assert args[0] == watcher_module.COLLECTION_NAME
    assert watcher.stats.files_deleted == 1


def test_filter_ignores_non_markdown_and_hidden(tmp_path: Path) -> None:
    fake_storage = MagicMock(spec=QdrantStorage)
    watcher = BrainWatcher(
        [tmp_path],
        fake_storage,
        _FakeEmbedder(),
        debounce_ms=200,
    )

    watcher.on_modified(FileModifiedEvent(str(tmp_path / ".DS_Store")))
    watcher.on_modified(FileCreatedEvent(str(tmp_path / "image.png")))
    watcher.on_modified(FileModifiedEvent(str(tmp_path / "node_modules/x.md")))
    # Real .md should NOT be filtered.
    watcher.on_modified(FileModifiedEvent(str(tmp_path / "real.md")))

    assert watcher.stats.events_received == 4
    assert watcher.stats.events_filtered == 3


def test_directory_events_are_ignored(tmp_path: Path) -> None:
    fake_storage = MagicMock(spec=QdrantStorage)
    watcher = BrainWatcher(
        [tmp_path],
        fake_storage,
        _FakeEmbedder(),
        debounce_ms=200,
    )
    watcher.on_modified(DirCreatedEvent(str(tmp_path / "subdir")))
    assert watcher.stats.events_received == 0


# ----------------------------------------------------------------------
# end-to-end smoke against live Qdrant
# ----------------------------------------------------------------------


def test_live_observer_reindexes_on_modify(
    storage: QdrantStorage, collection: str, tmp_path: Path
) -> None:
    file_path = tmp_path / "doc.md"
    file_path.write_text("# Original\n\nbody\n", encoding="utf-8")

    watcher = _make_watcher(storage, collection, [tmp_path])
    with watcher:
        # Simulate a save by writing then triggering the worker manually
        # rather than waiting on FSEvents (more deterministic).
        file_path.write_text("# Updated\n\n" + ("body " * 60), encoding="utf-8")
        watcher.on_modified(FileModifiedEvent(str(file_path)))
        # Force the debounce window past + run worker.
        import time as _time

        _time.sleep(0.3)
        watcher._process_due()

    assert watcher.stats.files_ingested == 1
    assert storage.count(collection) >= 1
