"""Collection definitions and payload schema for SDET Brain.

The brand corpus lives in a single Qdrant collection (`sdet_brand_v1`).
Re-creating the collection requires renaming the constant - the version
suffix lets us migrate without overwriting old data in disaster scenarios.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Final, Literal, TypedDict

from qdrant_client.models import Distance, PayloadSchemaType

from sdet_brain.storage.qdrant_client import QdrantStorage

logger = logging.getLogger(__name__)

COLLECTION_NAME: Literal["sdet_brand_v1"] = "sdet_brand_v1"
DEFAULT_DISTANCE: Distance = Distance.COSINE

# Named-vector identifiers (T2-03). The collection now stores both a
# dense semantic vector ("dense") and a sparse BM25 vector ("bm25").
# Hybrid search fuses the two via RRF at query time.
DENSE_VECTOR_NAME: Final[str] = "dense"
SPARSE_VECTOR_NAME: Final[str] = "bm25"

SourceType = Literal[
    "project-knowledge",
    "drafts",
    "articles",
    "sprint-reports",
    "brief",
]

PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "source_type": PayloadSchemaType.KEYWORD,
    "source_path": PayloadSchemaType.KEYWORD,
    "content_hash": PayloadSchemaType.KEYWORD,
    "chunk_index": PayloadSchemaType.INTEGER,
    # Structured frontmatter fields lifted to top-level payload (T2-01).
    # Each key has at most a few hundred distinct values across the
    # corpus, so KEYWORD indexes are cheap and let server-side filters
    # short-circuit before scanning vectors.
    "category": PayloadSchemaType.KEYWORD,
    "status": PayloadSchemaType.KEYWORD,
    "tags": PayloadSchemaType.KEYWORD,
    "series": PayloadSchemaType.KEYWORD,
    "language": PayloadSchemaType.KEYWORD,
    # Datetime indexes powering the `since` filter on `search_decisions`.
    "fm_created_at": PayloadSchemaType.DATETIME,
    "fm_updated_at": PayloadSchemaType.DATETIME,
}


class ChunkPayload(TypedDict):
    """Schema persisted alongside every embedding vector.

    The shape is enforced at the application layer (Qdrant payloads are
    untyped JSON). `frontmatter` is whatever YAML header the source file
    declared - we keep it as a free-form dict for now and refine the
    schema in T2-01.
    """

    source_path: str
    source_type: SourceType
    chunk_index: int
    total_chunks: int
    content_hash: str
    created_at: str
    frontmatter: dict[str, object]


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 with `Z` suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_collections(
    storage: QdrantStorage,
    vector_size: int,
    *,
    name: str = COLLECTION_NAME,
) -> bool:
    """Create the primary hybrid collection and payload indexes if missing.

    The collection carries two named vectors:
      * ``dense`` (cosine, ``vector_size``) - semantic embedding.
      * ``bm25`` (sparse, IDF modifier) - BM25 term vector.

    Parameters
    ----------
    storage:
        Configured `QdrantStorage` instance.
    vector_size:
        Dense embedding dimensionality. Must match the producer
        (MLX = 1024, Gemini = 768).
    name:
        Collection name override. Defaults to the production
        ``sdet_brand_v1`` constant; tests pass a disposable name so
        they do not stomp on real data.

    Returns ``True`` when the collection was created during this call,
    ``False`` when it already existed (idempotent re-init).
    """
    created = storage.ensure_hybrid_collection(
        name=name,
        dense_vector_size=vector_size,
        dense_distance=DEFAULT_DISTANCE,
    )
    storage.ensure_payload_indexes(name, PAYLOAD_INDEXES)
    if created:
        logger.info(
            "Created hybrid Qdrant collection %s (dense=%d, sparse=bm25)",
            name,
            vector_size,
        )
    else:
        logger.info("Qdrant collection %s already exists", name)
    return created
