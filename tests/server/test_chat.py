"""Tests for the conversational chat endpoint (T3-01)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import PointStruct, SparseVector

from sdet_brain.config import Settings
from sdet_brain.embeddings.sparse_embedder import SparseVector as SparseV
from sdet_brain.llm.protocol import ChatMessage
from sdet_brain.server.app import create_app
from sdet_brain.server.dependencies import require_embedder, require_storage
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
    name = f"sdet_brain_chat_test_{os.getpid()}_{id(storage)}"
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


class _FakeSparse:
    model_name = "fake/sparse"

    def embed(self, texts):  # type: ignore[no-untyped-def]
        out = []
        for t in texts:
            base = abs(hash(t))
            out.append(
                SparseV(
                    indices=[base % 1024, (base + 1) % 1024], values=[1.0, 0.5]
                )
            )
        return out

    def health_check(self) -> bool:
        return True


class _FakeLLM:
    model_name = "fake/llm"
    last_messages: list[ChatMessage] | None = None

    def __init__(self, response: str = "OK_REPLY") -> None:
        self._response = response

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.7) -> str:
        return self._response

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        type(self).last_messages = list(messages)
        return self._response

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        type(self).last_messages = list(messages)
        for token in self._response.split():
            yield token + " "

    def health_check(self) -> bool:
        return True


@pytest.fixture
def client(
    storage: QdrantStorage,
    collection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """Build a FastAPI test client with the chat dependencies pre-faked."""
    embedder = _FakeEmbedder()
    fake_llm = _FakeLLM("Hello world reply")
    fake_sparse = _FakeSparse()

    # Build a ChatPipeline pinned to the test collection. Replacing
    # the route's `_build_pipeline` is the simplest way to inject the
    # collection because `ChatPipeline.__init__` defaults are bound at
    # class-definition time and resist monkeypatching.
    from sdet_brain.server.chat.pipeline import ChatPipeline

    def _test_pipeline(storage_arg, embedder_arg):  # type: ignore[no-untyped-def]
        return ChatPipeline(
            embedder=embedder_arg,
            sparse_embedder=fake_sparse,
            storage=storage_arg,
            llm=fake_llm,
            collection=collection,
        )

    monkeypatch.setattr(
        "sdet_brain.server.routes.chat._build_pipeline",
        _test_pipeline,
    )

    # Seed at least one chunk for retrieval.
    storage.upsert_points(
        collection,
        [
            PointStruct(
                id=1,
                vector={
                    "dense": embedder.embed(["seed"])[0],
                    "bm25": SparseVector(indices=[1, 2], values=[1.0, 0.5]),
                },
                payload={
                    "text": "seed chunk text",
                    "source_path": "/var/test/seed.md",
                },
            )
        ],
    )

    app = create_app(settings=Settings())
    app.dependency_overrides[require_storage] = lambda: storage
    app.dependency_overrides[require_embedder] = lambda: embedder

    with TestClient(app) as c:
        yield c


def test_chat_non_stream_returns_reply_and_sources(client: TestClient) -> None:
    resp = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "co masz w corpusie"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reply"] == "Hello world reply"
    assert body["retrieved_chunk_count"] >= 1
    assert "/var/test/seed.md" in body["sources"]


def test_chat_stream_emits_sse_data_frames(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/chat",
        json={
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode()
    assert "data: " in body
    # final frame includes the done event with sources
    assert '"event": "done"' in body or '"event":"done"' in body


def test_chat_retrieve_false_skips_corpus(client: TestClient) -> None:
    resp = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "no retrieve"}],
            "stream": False,
            "retrieve": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved_chunk_count"] == 0
    assert body["sources"] == []


def test_chat_rejects_empty_messages(client: TestClient) -> None:
    resp = client.post("/chat", json={"messages": []})
    assert resp.status_code == 422  # validation error from min_length=1


def test_chat_keeps_history_in_messages(client: TestClient) -> None:
    """The fake LLM's ``last_messages`` should reflect history + system."""
    payload = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first reply"},
            {"role": "user", "content": "second"},
        ],
        "stream": False,
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200, resp.text
    sent = _FakeLLM.last_messages
    assert sent is not None
    assert sent[0].role == "system"  # SDET Brain prompt
    # history of three turns appended after the system message
    history_roles = [m.role for m in sent[1:]]
    assert history_roles == ["user", "assistant", "user"]


def test_chat_sse_payload_is_valid_json_per_frame(client: TestClient) -> None:
    """Each SSE ``data:`` frame must parse as JSON (no concatenation bugs)."""
    with client.stream(
        "POST",
        "/chat",
        json={
            "messages": [{"role": "user", "content": "json please"}],
            "stream": True,
        },
    ) as resp:
        body = b"".join(resp.iter_bytes()).decode()
    frames = [
        line[len("data: ") :]
        for line in body.split("\n\n")
        if line.startswith("data: ")
    ]
    assert frames
    for frame in frames:
        json.loads(frame)  # raises if any frame is malformed
