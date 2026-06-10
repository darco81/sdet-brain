"""`list_articles_by_status` MCP tool."""

from __future__ import annotations

from typing import Any, Final, cast

from qdrant_client.models import FieldCondition, Filter, MatchValue

from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools._helpers import (
    ToolError,
    collection_or_default,
    require_storage,
    safe_int,
    safe_str,
)
from sdet_brain.storage.qdrant_client import QdrantStorage

_ALLOWED_STATUSES: Final[frozenset[str]] = frozenset({"draft", "review", "published", "archive"})
_SCROLL_PAGE: Final[int] = 256


def list_articles_by_status(
    state: AppState,
    *,
    status: str,
    series: str | None = None,
    collection: str | None = None,
) -> str:
    """List case-study chunks filtered by ``status`` (and optional ``series``).

    Unlike the search tools this one does NOT take a query - it scrolls
    every chunk that matches the keyword filter. The result is grouped
    by file so the caller sees one row per article rather than one per
    chunk.
    """
    if status not in _ALLOWED_STATUSES:
        raise ToolError(f"status must be one of {sorted(_ALLOWED_STATUSES)}, got {status!r}")

    storage = require_storage(state)
    collection_name = collection_or_default(collection)
    must: list[Any] = [
        FieldCondition(key="category", match=MatchValue(value="case-study")),
        FieldCondition(key="status", match=MatchValue(value=status)),
    ]
    if series:
        must.append(FieldCondition(key="series", match=MatchValue(value=series)))

    return _scroll_and_group(
        storage,
        collection_name,
        Filter(must=cast(Any, must)),
        status=status,
        series=series,
    )


def _scroll_and_group(
    storage: QdrantStorage,
    collection: str,
    query_filter: Filter,
    *,
    status: str,
    series: str | None,
) -> str:
    by_path: dict[str, dict[str, Any]] = {}
    offset: Any = None
    while True:
        page, offset = storage.client.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=_SCROLL_PAGE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in page:
            payload = dict(point.payload or {})
            path = safe_str(payload, "source_path")
            if not path:
                continue
            entry = by_path.setdefault(
                path,
                {
                    "chunks": 0,
                    "series": safe_str(payload, "series", default=""),
                    "language": safe_str(payload, "language", default=""),
                    "total_chunks": safe_int(payload, "total_chunks"),
                },
            )
            entry["chunks"] = entry["chunks"] + 1
        if offset is None:
            break

    return _format(status, series, by_path)


def _format(status: str, series: str | None, by_path: dict[str, dict[str, Any]]) -> str:
    series_suffix = f", series={series}" if series else ""
    if not by_path:
        return f"No case-study articles with status={status}{series_suffix}.\n"
    lines = [
        f"# Case-study articles (status={status}{series_suffix})",
        "",
        f"_{len(by_path)} articles_",
        "",
    ]
    for path in sorted(by_path):
        entry = by_path[path]
        meta_parts: list[str] = []
        if entry["series"]:
            meta_parts.append(f"series={entry['series']}")
        if entry["language"]:
            meta_parts.append(f"lang={entry['language']}")
        meta_parts.append(f"chunks={entry['chunks']}")
        meta = ", ".join(meta_parts)
        lines.append(f"- `{path}` ({meta})")
    lines.append("")
    return "\n".join(lines)
