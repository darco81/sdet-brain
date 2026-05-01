"""Integration tests for the QdrantStorage wrapper.

Run with a live Qdrant container (``docker compose up -d qdrant``). When
the container is down the tests skip rather than fail, but CI / pre-merge
runs must have it up - skipped tests do not count toward AC.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
)

from sdet_brain.storage.collections import (
    PAYLOAD_INDEXES,
    init_collections,
)
from sdet_brain.storage.qdrant_client import QdrantStorage

VECTOR_SIZE = 384


def _vec(seed: int) -> list[float]:
    """Deterministic dummy 384-dim vector built from an integer seed."""
    return [((seed * 31 + i) % 100) / 100.0 for i in range(VECTOR_SIZE)]


def _named_vec(seed: int) -> dict[str, object]:
    """Hybrid named-vector payload: dense + tiny sparse stub."""
    return {
        "dense": _vec(seed),
        "bm25": SparseVector(
            indices=[seed % 32], values=[1.0]
        ),
    }


def test_ensure_collection_is_idempotent(
    storage: QdrantStorage, temp_collection: str
) -> None:
    created_first = storage.ensure_hybrid_collection(temp_collection, VECTOR_SIZE)
    created_second = storage.ensure_hybrid_collection(temp_collection, VECTOR_SIZE)
    assert created_first is True
    assert created_second is False
    assert storage.collection_exists(temp_collection)


def test_delete_collection_returns_correct_flag(
    storage: QdrantStorage, temp_collection: str
) -> None:
    storage.ensure_hybrid_collection(temp_collection, VECTOR_SIZE)
    assert storage.delete_collection(temp_collection) is True
    assert storage.delete_collection(temp_collection) is False


def test_hybrid_search_returns_fused_hits(
    storage: QdrantStorage, temp_collection: str
) -> None:
    storage.ensure_hybrid_collection(temp_collection, VECTOR_SIZE)
    storage.upsert_points(
        temp_collection,
        [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": _vec(seed=i),
                    "bm25": SparseVector(indices=[i, i + 1], values=[1.0, 0.5]),
                },
                payload={"label": f"chunk-{i}"},
            )
            for i in range(5)
        ],
    )
    hits = storage.hybrid_search(
        temp_collection,
        dense_vector=_vec(seed=0),
        sparse_indices=[0, 1],
        sparse_values=[1.0, 0.5],
        limit=3,
    )
    assert len(hits) == 3
    assert hits[0].payload is not None


def test_upsert_and_search_round_trip(
    storage: QdrantStorage, temp_collection: str
) -> None:
    storage.ensure_hybrid_collection(temp_collection, VECTOR_SIZE)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=_named_vec(i),
            payload={"label": f"chunk-{i}", "source_path": f"/var/sdet-brain-fixtures/file-{i}.md"},
        )
        for i in range(5)
    ]
    written = storage.upsert_points(temp_collection, points)
    assert written == 5
    assert storage.count(temp_collection) == 5

    results = storage.search(temp_collection, query_vector=_vec(seed=0), limit=3)
    assert len(results) == 3
    assert results[0].payload is not None
    assert results[0].payload["label"] == "chunk-0"


def test_delete_by_filter_removes_only_matching_points(
    storage: QdrantStorage, temp_collection: str
) -> None:
    storage.ensure_hybrid_collection(temp_collection, VECTOR_SIZE)
    storage.upsert_points(
        temp_collection,
        [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=_named_vec(i),
                payload={"source_path": "/var/sdet-brain-fixtures/keep.md"},
            )
            for i in range(3)
        ]
        + [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=_named_vec(10 + i),
                payload={"source_path": "/var/sdet-brain-fixtures/drop.md"},
            )
            for i in range(2)
        ],
    )
    assert storage.count(temp_collection) == 5

    storage.delete_by_filter(
        temp_collection,
        Filter(
            must=[FieldCondition(key="source_path", match=MatchValue(value="/var/sdet-brain-fixtures/drop.md"))]
        ),
    )
    assert storage.count(temp_collection) == 3


def test_init_collections_creates_collection_and_indexes(
    storage: QdrantStorage, temp_collection: str
) -> None:
    """`init_collections` is idempotent and registers all payload indexes.

    Uses a disposable collection name so the production
    ``sdet_brand_v1`` is never touched - test runs are safe in any
    environment that already has real data.
    """
    created = init_collections(storage, vector_size=VECTOR_SIZE, name=temp_collection)
    assert created is True
    # Re-running must be a no-op.
    assert init_collections(storage, vector_size=VECTOR_SIZE, name=temp_collection) is False

    snapshot = storage.status(temp_collection)
    assert snapshot.name == temp_collection
    assert snapshot.vector_size == VECTOR_SIZE
    assert snapshot.distance.lower() == "cosine"

    info = storage.get_collection(temp_collection)
    indexed_fields = set(info.payload_schema or {})
    assert set(PAYLOAD_INDEXES).issubset(indexed_fields)


@pytest.mark.parametrize("count", [0, 1, 7])
def test_upsert_points_returns_written_count(
    storage: QdrantStorage, temp_collection: str, count: int
) -> None:
    storage.ensure_hybrid_collection(temp_collection, VECTOR_SIZE)
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=_named_vec(i), payload={})
        for i in range(count)
    ]
    assert storage.upsert_points(temp_collection, points) == count
