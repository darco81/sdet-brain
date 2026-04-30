"""`get_chunk_neighbors` MCP tool implementation."""

from __future__ import annotations

from typing import Any

from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools._helpers import (
    ToolError,
    collection_or_default,
    require_storage,
    safe_int,
    safe_str,
    source_path_filter,
)
from sdet_brain.storage.qdrant_client import QdrantStorage

SCROLL_PAGE_SIZE = 64


def get_chunk_neighbors(
    state: AppState,
    *,
    source_path: str,
    chunk_index: int,
    window: int = 2,
    collection: str | None = None,
) -> str:
    """Return chunks ``[chunk_index - window, chunk_index + window]`` clamped to the file."""
    if not source_path:
        raise ToolError("source_path must not be empty")
    if chunk_index < 0:
        raise ToolError("chunk_index must be non-negative")
    if window < 0:
        raise ToolError("window must be non-negative")

    storage = require_storage(state)
    collection_name = collection_or_default(collection)
    chunks = _scroll_chunks(storage, collection_name, source_path)

    if not chunks:
        return f"No chunks indexed for `{source_path}`.\n"

    by_index = {chunk["chunk_index"]: chunk for chunk in chunks}
    total = chunks[-1]["total_chunks"] or len(chunks)
    start = max(0, chunk_index - window)
    end = min(total - 1, chunk_index + window)

    lines = [
        f"# Neighbours for `{source_path}` "
        f"chunk {chunk_index} (window={window}, range {start}..{end})",
        "",
    ]
    for index in range(start, end + 1):
        chunk = by_index.get(index)
        if chunk is None:
            lines.append(f"## chunk {index} _(missing)_\n")
            continue
        heading = chunk["heading_path"]
        text = chunk["text"]
        marker = " (target)" if index == chunk_index else ""
        lines.append(f"## chunk {index}/{total}{marker}")
        if heading:
            lines.append(f"_{heading}_")
        lines.append("")
        lines.append(text or "_(no text stored)_")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _scroll_chunks(
    storage: QdrantStorage, collection: str, source_path: str
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        page, offset = storage.client.scroll(
            collection_name=collection,
            scroll_filter=source_path_filter(source_path),
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in page:
            payload = dict(point.payload or {})
            chunk_index = safe_int(payload, "chunk_index")
            if chunk_index is None:
                continue
            items.append(
                {
                    "chunk_index": chunk_index,
                    "total_chunks": safe_int(payload, "total_chunks"),
                    "heading_path": safe_str(payload, "heading_path"),
                    "text": safe_str(payload, "text"),
                }
            )
        if offset is None:
            break
    items.sort(key=lambda item: item["chunk_index"])
    return items
