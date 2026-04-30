"""Collection definitions and payload schema for SDET Brain.

The brand corpus lives in a single Qdrant collection (`sdet_brand_v1`).
Re-creating the collection requires renaming the constant - the version
suffix lets us migrate without overwriting old data in disaster scenarios.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Literal, TypedDict

from qdrant_client.models import Distance, PayloadSchemaType

from sdet_brain.storage.qdrant_client import QdrantStorage

logger = logging.getLogger(__name__)

COLLECTION_NAME: Literal["sdet_brand_v1"] = "sdet_brand_v1"
DEFAULT_DISTANCE: Distance = Distance.COSINE

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
    """Create the primary collection and payload indexes if missing.

    Parameters
    ----------
    storage:
        Configured `QdrantStorage` instance.
    vector_size:
        Embedding dimensionality. Must match the producer (MLX = 1024,
        Gemini = 768).
    name:
        Collection name override. Defaults to the production
        ``sdet_brand_v1`` constant; tests pass a disposable name so they
        do not stomp on real data.

    Returns ``True`` when the collection was created during this call,
    ``False`` when it already existed (idempotent re-init).
    """
    created = storage.ensure_collection(
        name=name,
        vector_size=vector_size,
        distance=DEFAULT_DISTANCE,
    )
    storage.ensure_payload_indexes(name, PAYLOAD_INDEXES)
    if created:
        logger.info("Created Qdrant collection %s (vector_size=%d)", name, vector_size)
    else:
        logger.info("Qdrant collection %s already exists", name)
    return created
