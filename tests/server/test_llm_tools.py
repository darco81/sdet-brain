"""Tests for the LLM-backed MCP tools (T2-05).

The fake ``ILLM`` injected via ``monkeypatch`` keeps the tests offline
- no Qwen weights touched. Real integration is covered by manual
smoke tests after deploy.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
import pytest
from qdrant_client.models import PointStruct, SparseVector

from sdet_brain.config import Settings
from sdet_brain.embeddings.factory import EmbedderSelection
from sdet_brain.llm.protocol import ChatMessage
from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools._helpers import ToolError
from sdet_brain.server.tools.query_rewrite import query_rewrite
from sdet_brain.server.tools.summarize_results import summarize_results
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
    name = f"sdet_brain_llm_test_{os.getpid()}_{id(storage)}"
    storage.ensure_hybrid_collection(name, VECTOR_SIZE)
    yield name
    if storage.collection_exists(name):
        storage.client.delete_collection(collection_name=name)


class _FakeEmbedder:
    vector_size = VECTOR_SIZE
    model_name = "fake/dense"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [(((hash(t) >> i) & 0xFF) / 255.0) for i in range(VECTOR_SIZE)]
            for t in texts
        ]

    def health_check(self) -> bool:
        return True


class _FakeLLM:
    model_name = "fake/llm"
    last_prompt: str | None = None

    def __init__(self, response: str = "FAKE_LLM_REPLY") -> None:
        self._response = response

    def generate(
        self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.7
    ) -> str:
        self.last_prompt = prompt
        return self._response

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.last_prompt = messages[-1].content
        return self._response

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        yield self.chat(messages)

    def health_check(self) -> bool:
        return True


class _FakeSparse:
    model_name = "fake/sparse"

    def embed(self, texts):  # type: ignore[no-untyped-def]
        from sdet_brain.embeddings.sparse_embedder import SparseVector

        out = []
        for t in texts:
            base = abs(hash(t))
            out.append(
                SparseVector(
                    indices=[base % 1024, (base + 1) % 1024], values=[1.0, 0.5]
                )
            )
        return out

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


@pytest.fixture(autouse=True)
def _patch_llm_and_sparse(monkeypatch: pytest.MonkeyPatch) -> _FakeLLM:
    fake_llm = _FakeLLM("hypothetical body about chunks")

    class _FakeRouter:
        def chat(self, messages, *, task="chat", max_tokens=512, temperature=0.7):  # type: ignore[no-untyped-def]
            return fake_llm.chat(messages, max_tokens=max_tokens, temperature=temperature)

        def generate(self, prompt, *, task="summarize", max_tokens=512, temperature=0.7):  # type: ignore[no-untyped-def]
            return fake_llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        def chat_stream(self, messages, *, task="chat", max_tokens=512, temperature=0.7):  # type: ignore[no-untyped-def]
            yield from fake_llm.chat_stream(messages, max_tokens=max_tokens, temperature=temperature)

        def get(self, task):  # type: ignore[no-untyped-def]
            return fake_llm

    fake_router = _FakeRouter()
    monkeypatch.setattr(
        "sdet_brain.server.tools.query_rewrite.get_router",
        lambda: fake_router,
    )
    monkeypatch.setattr(
        "sdet_brain.server.tools.summarize_results.get_router",
        lambda: fake_router,
    )
    monkeypatch.setattr(
        "sdet_brain.server.tools.query_rewrite._sparse",
        _FakeSparse,
    )
    monkeypatch.setattr(
        "sdet_brain.server.tools.summarize_results._sparse",
        _FakeSparse,
    )
    return fake_llm


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


# --- query_rewrite -------------------------------------------------------


def test_query_rewrite_uses_llm_hypothetical(
    state: AppState,
    collection: str,
    storage: QdrantStorage,
    _patch_llm_and_sparse: _FakeLLM,
) -> None:
    _seed(storage, collection, "/q/file.md", "answer about chunks")
    out = query_rewrite(state, query="what about chunks", limit=3, collection=collection)
    assert "Hypothetical answer used for retrieval" in out
    assert "hypothetical body about chunks" in out


def test_query_rewrite_empty_query_raises(
    state: AppState, collection: str
) -> None:
    with pytest.raises(ToolError):
        query_rewrite(state, query="  ", collection=collection)


def test_query_rewrite_no_matches_message(
    state: AppState, collection: str
) -> None:
    out = query_rewrite(state, query="nothing seeded", collection=collection)
    assert "No matches" in out


# --- summarize_results --------------------------------------------------


def test_summarize_emits_summary_and_sources(
    state: AppState,
    collection: str,
    storage: QdrantStorage,
) -> None:
    _seed(storage, collection, "/s/one.md", "deploy on Tuesday")
    _seed(storage, collection, "/s/two.md", "deploy fixes for Wednesday")
    out = summarize_results(state, topic="deploy", limit=5, collection=collection)
    assert "Summary for `deploy`" in out
    assert "## Sources" in out
    assert "/s/one.md" in out or "/s/two.md" in out


def test_summarize_empty_topic_raises(state: AppState, collection: str) -> None:
    with pytest.raises(ToolError):
        summarize_results(state, topic="", collection=collection)


def test_summarize_no_matches_message(
    state: AppState, collection: str
) -> None:
    out = summarize_results(state, topic="nothing seeded", collection=collection)
    assert "No matches" in out
