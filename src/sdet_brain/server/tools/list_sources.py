"""`list_sources` MCP tool implementation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Final, NamedTuple

from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools._helpers import (
    collection_or_default,
    require_storage,
    safe_str,
    source_type_filter,
)
from sdet_brain.storage.qdrant_client import QdrantStorage

SCROLL_PAGE_SIZE: Final[int] = 256


class _Aggregate(NamedTuple):
    source_path: str
    source_type: str
    chunks: int
    last_ingestion_at: str


def list_sources(
    state: AppState,
    *,
    source_type: str | None = None,
    collection: str | None = None,
) -> str:
    """Group every indexed chunk by source path and render as Markdown."""
    storage = require_storage(state)
    collection_name = collection_or_default(collection)

    aggregates = _scroll_and_group(storage, collection_name, source_type)
    if not aggregates:
        suffix = f" (source_type={source_type})" if source_type else ""
        return f"No sources indexed{suffix}.\n"

    total_chunks = sum(item.chunks for item in aggregates)
    title_filter = f" (filter source_type={source_type})" if source_type else ""
    lines = [
        f"# Indexed sources{title_filter}",
        "",
        f"_{len(aggregates)} files / {total_chunks} chunks_",
        "",
    ]
    aggregates_sorted = sorted(aggregates, key=lambda item: item.source_path)
    for item in aggregates_sorted:
        ingested = item.last_ingestion_at or "unknown"
        lines.append(
            f"- **[{item.source_type}]** `{item.source_path}` "
            f"({item.chunks} chunks, last ingested {ingested})"
        )
    lines.append("")
    return "\n".join(lines)


def _scroll_and_group(
    storage: QdrantStorage,
    collection: str,
    source_type: str | None,
) -> list[_Aggregate]:
    chunk_counts: dict[str, int] = defaultdict(int)
    type_per_path: dict[str, str] = {}
    last_seen: dict[str, str] = {}

    offset: Any = None
    query_filter = source_type_filter(source_type)
    while True:
        page, offset = storage.client.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in page:
            payload = dict(point.payload or {})
            path = safe_str(payload, "source_path")
            if not path:
                continue
            chunk_counts[path] += 1
            type_per_path[path] = safe_str(payload, "source_type", default="unknown")
            created_at = safe_str(payload, "created_at")
            if created_at and (path not in last_seen or created_at > last_seen[path]):
                last_seen[path] = created_at
        if offset is None:
            break

    return [
        _Aggregate(
            source_path=path,
            source_type=type_per_path.get(path, "unknown"),
            chunks=chunk_counts[path],
            last_ingestion_at=last_seen.get(path, ""),
        )
        for path in chunk_counts
    ]
