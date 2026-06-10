"""Tests for the multi_query_search MCP tool (T4-04)."""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
import pytest
from qdrant_client.models import PointStruct, SparseVector

from sdet_brain.config import Settings
from sdet_brain.embeddings.factory import EmbedderSelection
from sdet_brain.embeddings.sparse_embedder import SparseVector as SparseV
from sdet_brain.llm.protocol import ChatMessage
from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools._helpers import ToolError
from sdet_brain.server.tools.multi_query import (
    _decompose,
    _extract_json,
    _rrf_merge,
    multi_query_search,
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
    name = f"sdet_brain_multi_query_test_{os.getpid()}_{id(storage)}"
    storage.ensure_hybrid_collection(name, VECTOR_SIZE)
    yield name
    if storage.collection_exists(name):
        storage.client.delete_collection(collection_name=name)


class _FakeEmbedder:
    vector_size = VECTOR_SIZE
    model_name = "fake/dense"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[(((hash(t) >> i) & 0xFF) / 255.0) for i in range(VECTOR_SIZE)] for t in texts]

    def health_check(self) -> bool:
        return True


class _FakeSparse:
    model_name = "fake/sparse"

    def embed(self, texts):  # type: ignore[no-untyped-def]
        out = []
        for t in texts:
            base = abs(hash(t))
            out.append(
                SparseV(
                    indices=[base % 1024, (base + 1) % 1024],
                    values=[1.0, 0.5],
                )
            )
        return out

    def health_check(self) -> bool:
        return True


class _FakeRouter:
    """Returns a JSON decomposition that splits the query into 2 sub-queries."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[ChatMessage] | None = None

    def chat(self, messages, *, task="chat", max_tokens=512, temperature=0.7):  # type: ignore[no-untyped-def]
        self.last_messages = list(messages)
        return self._response

    def generate(self, prompt, *, task="summarize", max_tokens=512, temperature=0.7):  # type: ignore[no-untyped-def]
        return self._response

    def chat_stream(self, messages, *, task="chat", max_tokens=512, temperature=0.7):  # type: ignore[no-untyped-def]
        yield self._response

    def get(self, task):  # type: ignore[no-untyped-def]
        return self


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


@pytest.fixture(autouse=True)
def _patch_router_and_sparse(monkeypatch: pytest.MonkeyPatch) -> _FakeRouter:
    router = _FakeRouter('{"queries": ["wcag toolkit", "portfolio deploy"]}')
    monkeypatch.setattr(
        "sdet_brain.server.tools.multi_query.get_router",
        lambda: router,
    )
    monkeypatch.setattr(
        "sdet_brain.server.tools.multi_query._sparse",
        _FakeSparse,
    )
    return router


def _seed(storage: QdrantStorage, collection: str, source: str, text: str) -> None:
    embedder = _FakeEmbedder()
    base = abs(hash(text))
    storage.upsert_points(
        collection,
        [
            PointStruct(
                id=base % 10_000_000,
                vector={
                    "dense": embedder.embed([text])[0],
                    "bm25": SparseVector(
                        indices=[base % 1024, (base + 1) % 1024],
                        values=[1.0, 0.5],
                    ),
                },
                payload={
                    "text": text,
                    "source_path": source,
                    "source_type": "drafts",
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
            )
        ],
    )


# --- _extract_json ------------------------------------------------------


def test_extract_json_handles_fenced_block() -> None:
    raw = 'Here you go:\n```json\n{"queries": ["a", "b"]}\n```'
    assert _extract_json(raw) == '{"queries": ["a", "b"]}'


def test_extract_json_handles_bare_object() -> None:
    raw = 'preamble {"queries": ["only"]} trailer'
    assert _extract_json(raw) == '{"queries": ["only"]}'


def test_extract_json_returns_none_when_no_json() -> None:
    assert _extract_json("just text, no json here") is None


# --- _decompose ---------------------------------------------------------


def test_decompose_returns_queries_array(
    _patch_router_and_sparse: _FakeRouter,
) -> None:
    out = _decompose("complex question")
    assert out == ["wcag toolkit", "portfolio deploy"]


def test_decompose_falls_back_when_llm_returns_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = _FakeRouter("not json at all")
    monkeypatch.setattr(
        "sdet_brain.server.tools.multi_query.get_router",
        lambda: bad,
    )
    out = _decompose("the original question")
    assert out == ["the original question"]


def test_decompose_caps_at_five_subqueries(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = '{"queries": ["a","b","c","d","e","f","g"]}'
    monkeypatch.setattr(
        "sdet_brain.server.tools.multi_query.get_router",
        lambda: _FakeRouter(payload),
    )
    assert _decompose("x") == ["a", "b", "c", "d", "e"]


# --- _rrf_merge ---------------------------------------------------------


class _Hit:
    def __init__(self, point_id: str, payload: dict[str, str]) -> None:
        self.id = point_id
        self.score = 0.0
        self.payload = payload


def test_rrf_merge_dedups_and_orders_by_aggregate_rank() -> None:
    list_a = [_Hit("A", {"text": "a"}), _Hit("B", {"text": "b"})]
    list_b = [_Hit("B", {"text": "b"}), _Hit("C", {"text": "c"})]
    merged = _rrf_merge([list_a, list_b], limit=3)  # type: ignore[arg-type]
    ids = [m.id for m in merged]
    # B appears in both lists, so it ranks first.
    assert ids[0] == "B"
    assert set(ids) == {"A", "B", "C"}


# --- multi_query_search end-to-end -------------------------------------


def test_multi_query_search_runs_decompose_and_returns_decomposition_in_output(
    state: AppState, collection: str, storage: QdrantStorage
) -> None:
    _seed(storage, collection, "/m/wcag.md", "wcag toolkit publication plan")
    _seed(storage, collection, "/m/portfolio.md", "portfolio deploy day timeline")
    out = multi_query_search(
        state,
        query="how does wcag toolkit publication relate to portfolio deploy",
        limit=5,
        collection=collection,
    )
    assert "Decomposed into" in out
    assert "wcag toolkit" in out
    assert "portfolio deploy" in out


def test_multi_query_search_empty_query_raises(state: AppState, collection: str) -> None:
    with pytest.raises(ToolError):
        multi_query_search(state, query="   ", collection=collection)


def test_multi_query_search_empty_corpus_returns_no_match_message(
    state: AppState, collection: str
) -> None:
    out = multi_query_search(state, query="anything complex", collection=collection)
    assert "No matches across the decomposed sub-queries" in out
