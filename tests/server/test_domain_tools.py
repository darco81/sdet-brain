"""Unit tests for the five domain MCP tools (T2-02).

Mirrors the integration-style approach in ``test_tools.py``: a fake
embedder, a real Qdrant collection seeded per test, and assertions
against the Markdown output the tools return to MCP clients.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
import pytest
from qdrant_client.models import PointStruct, SparseVector

from sdet_brain.config import Settings
from sdet_brain.embeddings.factory import EmbedderSelection
from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools._helpers import ToolError
from sdet_brain.server.tools.domain import (
    list_articles_by_status,
    search_decisions,
    search_smaczki,
    search_sprint_reports,
    search_voice_samples,
)
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
    name = f"sdet_brain_domain_test_{os.getpid()}_{id(storage)}"
    storage.ensure_hybrid_collection(name, VECTOR_SIZE)
    # The DatetimeRange filter on `fm_created_at` requires a datetime
    # payload index; fast-fail if that wiring breaks.
    from qdrant_client.models import PayloadSchemaType

    storage.ensure_payload_indexes(
        name,
        {
            "category": PayloadSchemaType.KEYWORD,
            "status": PayloadSchemaType.KEYWORD,
            "series": PayloadSchemaType.KEYWORD,
            "fm_created_at": PayloadSchemaType.DATETIME,
        },
    )
    yield name
    if storage.collection_exists(name):
        storage.client.delete_collection(collection_name=name)


class _FakeEmbedder:
    vector_size = VECTOR_SIZE
    model_name = "fake/deterministic"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [(((hash(text) >> i) & 0xFF) / 255.0) for i in range(VECTOR_SIZE)]
            for text in texts
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


def _seed_chunk(
    storage: QdrantStorage,
    collection: str,
    source_path: str,
    *,
    text: str,
    category: str,
    status: str = "draft",
    series: str | None = None,
    language: str = "en",
    fm_created_at: str | None = None,
    chunk_index: int = 0,
) -> None:
    payload = {
        "text": text,
        "source_path": source_path,
        "source_type": "drafts",
        "chunk_index": chunk_index,
        "total_chunks": 1,
        "heading_path": "",
        "has_code": False,
        "char_count": len(text),
        "token_estimate": len(text) // 4,
        "frontmatter": {},
        "content_hash": f"hash-{source_path}-{chunk_index}",
        "created_at": "2026-05-01T00:00:00Z",
        "category": category,
        "status": status,
        "language": language,
    }
    if series:
        payload["series"] = series
    if fm_created_at:
        payload["fm_created_at"] = fm_created_at

    embedder = _FakeEmbedder()
    dense = embedder.embed([text])[0]
    text_hash = abs(hash(text))
    storage.upsert_points(
        collection,
        [
            PointStruct(
                id=abs(hash(f"{source_path}-{chunk_index}")) % 10_000_000,
                vector={
                    "dense": dense,
                    "bm25": SparseVector(
                        indices=[text_hash % 1024, (text_hash + 1) % 1024],
                        values=[1.0, 0.5],
                    ),
                },
                payload=payload,
            )
        ],
    )


# --- search_voice_samples -------------------------------------------------


def test_voice_samples_filters_to_category(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunk(storage, collection, "/v/voice.md", text="self-deprecating opener", category="voice-sample")
    _seed_chunk(storage, collection, "/v/draft.md", text="self-deprecating opener", category="draft")
    output = search_voice_samples(state, topic="opener", limit=5, collection=collection)
    assert "voice.md" in output
    assert "draft.md" not in output


def test_voice_samples_empty_query_raises(state: AppState, collection: str) -> None:
    with pytest.raises(ToolError):
        search_voice_samples(state, topic="  ", collection=collection)


def test_voice_samples_no_matches_message(
    state: AppState, collection: str
) -> None:
    output = search_voice_samples(state, topic="anything", collection=collection)
    assert "No voice samples match" in output


# --- search_smaczki -------------------------------------------------------


def test_smaczki_filters_to_category(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunk(storage, collection, "/s/smaczki.md", text="zinger about flaky tests", category="smaczki", series="wcag-toolkit")
    _seed_chunk(storage, collection, "/s/case.md", text="zinger about flaky tests", category="case-study")
    output = search_smaczki(state, topic="flaky tests", limit=5, collection=collection)
    assert "smaczki.md" in output
    assert "case.md" not in output


# --- search_decisions -----------------------------------------------------


def test_decisions_filters_to_category(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunk(storage, collection, "/d/decision.md", text="we decided not to mock", category="decision")
    _seed_chunk(storage, collection, "/d/sprint.md", text="we decided not to mock", category="sprint-report")
    output = search_decisions(state, topic="mock", limit=5, collection=collection)
    assert "decision.md" in output
    assert "sprint.md" not in output


def test_decisions_since_filter(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunk(
        storage,
        collection,
        "/d/old.md",
        text="ancient verdict",
        category="decision",
        fm_created_at="2026-04-01",
    )
    _seed_chunk(
        storage,
        collection,
        "/d/recent.md",
        text="ancient verdict",
        category="decision",
        fm_created_at="2026-04-29",
    )
    output = search_decisions(
        state, topic="verdict", since="2026-04-25", limit=5, collection=collection
    )
    assert "recent.md" in output
    assert "old.md" not in output


def test_decisions_invalid_since_raises(state: AppState, collection: str) -> None:
    with pytest.raises(ToolError):
        search_decisions(state, topic="x", since="not-a-date", collection=collection)


# --- list_articles_by_status ---------------------------------------------


def test_articles_by_status_groups_by_path(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunk(
        storage,
        collection,
        "/a/case-1.md",
        text="case 1 chunk 1",
        category="case-study",
        status="published",
        series="wcag-toolkit",
        chunk_index=0,
    )
    _seed_chunk(
        storage,
        collection,
        "/a/case-1.md",
        text="case 1 chunk 2",
        category="case-study",
        status="published",
        series="wcag-toolkit",
        chunk_index=1,
    )
    _seed_chunk(
        storage,
        collection,
        "/a/case-2.md",
        text="case 2",
        category="case-study",
        status="draft",
    )
    output = list_articles_by_status(state, status="published", collection=collection)
    assert "case-1.md" in output
    assert "case-2.md" not in output
    assert "1 articles" in output
    assert "series=wcag-toolkit" in output


def test_articles_by_status_with_series_filter(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunk(
        storage, collection, "/a/wcag.md",
        text="wcag", category="case-study", status="draft", series="wcag-toolkit",
    )
    _seed_chunk(
        storage, collection, "/a/jarvis.md",
        text="jarvis", category="case-study", status="draft", series="jarvis-brain",
    )
    output = list_articles_by_status(
        state, status="draft", series="wcag-toolkit", collection=collection
    )
    assert "wcag.md" in output
    assert "jarvis.md" not in output


def test_articles_invalid_status_raises(state: AppState, collection: str) -> None:
    with pytest.raises(ToolError):
        list_articles_by_status(state, status="totally-bogus", collection=collection)


def test_articles_empty_returns_message(state: AppState, collection: str) -> None:
    output = list_articles_by_status(state, status="published", collection=collection)
    assert "No case-study articles" in output


# --- search_sprint_reports -----------------------------------------------


def test_sprint_reports_filters_to_category(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunk(
        storage, collection, "/sr/wed.md",
        text="deploy outcome", category="sprint-report", series="case-study-01",
    )
    _seed_chunk(
        storage, collection, "/sr/draft.md",
        text="deploy outcome", category="draft",
    )
    output = search_sprint_reports(state, query="deploy", limit=5, collection=collection)
    assert "wed.md" in output
    assert "draft.md" not in output


def test_sprint_reports_project_filter(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed_chunk(
        storage, collection, "/sr/wcag.md",
        text="wcag sprint", category="sprint-report", series="wcag-toolkit",
    )
    _seed_chunk(
        storage, collection, "/sr/example.md",
        text="wcag sprint", category="sprint-report", series="case-study-01",
    )
    output = search_sprint_reports(
        state, query="sprint", project="wcag-toolkit", limit=5, collection=collection
    )
    assert "wcag.md" in output
    assert "example.md" not in output
