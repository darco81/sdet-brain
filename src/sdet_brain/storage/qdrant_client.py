"""Thin typed wrapper around the official `qdrant-client` SDK.

The wrapper centralises the operations the rest of the codebase needs and
keeps a single configured `QdrantClient` instance. We expose only what is
required today; T2-03 will add hybrid (sparse + dense) variants on top.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    CollectionInfo,
    Distance,
    Filter,
    Fusion,
    FusionQuery,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    QueryResponse,
    ScoredPoint,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S: Final[int] = 10


@dataclass(frozen=True)
class CollectionStatus:
    """Lightweight summary of a collection used by the CLI."""

    name: str
    vector_size: int
    distance: str
    points_count: int


class QdrantStorage:
    """Application-facing facade for Qdrant operations.

    Parameters
    ----------
    url:
        Full Qdrant HTTP URL (e.g. ``http://localhost:6333``).
    api_key:
        Optional API key forwarded as ``api-key`` header. Required for
        production deploys (Tier 3).
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        *,
        timeout: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._url = url
        self._client = QdrantClient(url=url, api_key=api_key, timeout=timeout)

    @property
    def url(self) -> str:
        return self._url

    @property
    def client(self) -> QdrantClient:
        """Escape hatch for callers that need raw access (e.g. tests)."""
        return self._client

    def collection_exists(self, name: str) -> bool:
        return self._client.collection_exists(collection_name=name)

    def ensure_collection(
        self,
        name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """Create a single-vector collection if it does not exist.

        Returns ``True`` when the collection was created in this call,
        ``False`` if it was already present (idempotent). Hybrid
        collections (T2-03) should call :meth:`ensure_hybrid_collection`
        instead.
        """
        if self.collection_exists(name):
            return False
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        return True

    def ensure_hybrid_collection(
        self,
        name: str,
        dense_vector_size: int,
        *,
        dense_distance: Distance = Distance.COSINE,
        dense_name: str = "dense",
        sparse_name: str = "bm25",
    ) -> bool:
        """Create a hybrid (named-vectors) collection if missing.

        The collection has one named dense vector (cosine, configurable
        size) and one named sparse vector (BM25 with IDF modifier).
        Returns ``True`` when this call created the collection.
        """
        if self.collection_exists(name):
            return False
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                dense_name: VectorParams(size=dense_vector_size, distance=dense_distance)
            },
            sparse_vectors_config={sparse_name: SparseVectorParams(modifier=Modifier.IDF)},
        )
        return True

    def delete_collection(self, name: str) -> bool:
        """Drop ``name`` if present. Returns ``True`` on actual delete."""
        if not self.collection_exists(name):
            return False
        self._client.delete_collection(collection_name=name)
        return True

    def ensure_payload_indexes(
        self,
        collection: str,
        indexes: Mapping[str, PayloadSchemaType],
    ) -> None:
        """Create payload indexes idempotently.

        Qdrant raises 409 when the index already exists; we swallow that
        specific case so the call is safe to repeat at startup.
        """
        for field_name, schema in indexes.items():
            try:
                self._client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            except UnexpectedResponse as exc:
                if exc.status_code == 409:
                    continue
                raise

    def upsert_points(
        self,
        collection: str,
        points: Sequence[PointStruct],
        wait: bool = True,
    ) -> int:
        """Upsert a batch of points and return the count written."""
        if not points:
            return 0
        self._client.upsert(
            collection_name=collection,
            points=list(points),
            wait=wait,
        )
        return len(points)

    def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        limit: int = 10,
        query_filter: Filter | None = None,
        score_threshold: float | None = None,
        vector_name: str = "dense",
    ) -> list[ScoredPoint]:
        """Run a dense-vector similarity search.

        Uses ``query_points`` under the hood (the post-1.10 API). The
        ``vector_name`` argument names the dense vector inside a hybrid
        collection (default ``"dense"``); single-vector legacy
        collections accept the same call as long as their vector name
        is ``"dense"`` (we recreate them that way in T2-03).
        """
        response: QueryResponse = self._client.query_points(
            collection_name=collection,
            query=list(query_vector),
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return list(response.points)

    def hybrid_search(
        self,
        collection: str,
        *,
        dense_vector: Sequence[float],
        sparse_indices: Sequence[int],
        sparse_values: Sequence[float],
        limit: int = 10,
        prefetch_limit: int = 30,
        query_filter: Filter | None = None,
        dense_name: str = "dense",
        sparse_name: str = "bm25",
    ) -> list[ScoredPoint]:
        """Hybrid search: dense + sparse fused via Reciprocal Rank Fusion.

        Each prefetch fetches ``prefetch_limit`` candidates against its
        own vector index; the outer ``query=FusionQuery(RRF)`` merges
        them and trims to ``limit``. ``query_filter`` (if given) is
        applied to *both* prefetches so the same payload constraint
        scopes both legs.
        """
        sparse_payload = QdrantSparseVector(
            indices=list(sparse_indices), values=list(sparse_values)
        )
        prefetches = [
            Prefetch(
                query=list(dense_vector),
                using=dense_name,
                limit=prefetch_limit,
                filter=query_filter,
            ),
            Prefetch(
                query=sparse_payload,
                using=sparse_name,
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ]
        response: QueryResponse = self._client.query_points(
            collection_name=collection,
            prefetch=prefetches,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return list(response.points)

    def delete_by_filter(
        self,
        collection: str,
        query_filter: Filter,
        wait: bool = True,
    ) -> None:
        """Delete every point matching ``query_filter`` (idempotent)."""
        self._client.delete(
            collection_name=collection,
            points_selector=query_filter,
            wait=wait,
        )

    def count(self, collection: str, *, exact: bool = True) -> int:
        result = self._client.count(collection_name=collection, exact=exact)
        return int(result.count)

    def get_collection(self, name: str) -> CollectionInfo:
        return self._client.get_collection(collection_name=name)

    def status(self, collection: str) -> CollectionStatus:
        """Return a flat snapshot suitable for logs and CLI output."""
        info = self.get_collection(collection)
        params = info.config.params.vectors
        if isinstance(params, VectorParams):
            vector_size = params.size
            distance = params.distance.value
        elif params is None:
            raise RuntimeError(f"Collection {collection!r} has no vector configuration")
        else:
            # Multi-named-vector collection (Tier 2). Pick the first vector.
            first_name, first_params = next(iter(params.items()))
            vector_size = first_params.size
            distance = first_params.distance.value
            logger.debug(
                "Collection %s uses named vectors; reporting %s",
                collection,
                first_name,
            )
        return CollectionStatus(
            name=collection,
            vector_size=vector_size,
            distance=distance,
            points_count=self.count(collection),
        )

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.get_collections().collections]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> QdrantStorage:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
