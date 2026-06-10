"""Shared helpers for the five domain tools."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, cast

from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchValue,
    ScoredPoint,
)

from sdet_brain.embeddings.sparse_embedder import (
    FastembedBM25,
    get_sparse_embedder,
)
from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools._helpers import (
    ToolError,
    collection_or_default,
    require_embedder,
    require_storage,
    safe_int,
    safe_str,
)

# Reuse the same lazy BM25 across all domain tools.
_SPARSE: FastembedBM25 | None = None


def _sparse_embedder() -> FastembedBM25:
    global _SPARSE
    if _SPARSE is None:
        _SPARSE = get_sparse_embedder()
    return _SPARSE


_DEFAULT_LIMIT = 5
_MAX_LIMIT = 50


def run_category_search(
    state: AppState,
    *,
    category: str,
    query: str,
    limit: int,
    extra_keyword_filters: dict[str, str] | None = None,
    since: str | None = None,
    collection: str | None = None,
) -> list[ScoredPoint]:
    """Embed ``query`` and semantic-search within a single category.

    Parameters
    ----------
    category:
        Required ``BrandFrontmatter.category`` value to filter on.
    extra_keyword_filters:
        Additional keyword constraints AND-combined with ``category``.
        Empty values are dropped so callers can blindly pass optionals.
    since:
        ISO date (``YYYY-MM-DD``) lower bound on ``fm_created_at``.
    """
    if not query.strip():
        raise ToolError("query must not be empty")
    if limit <= 0 or limit > _MAX_LIMIT:
        raise ToolError(f"limit must be between 1 and {_MAX_LIMIT}")

    embedder = require_embedder(state)
    storage = require_storage(state)
    collection_name = collection_or_default(collection)

    must: list[Any] = [FieldCondition(key="category", match=MatchValue(value=category))]
    for key, value in (extra_keyword_filters or {}).items():
        if value:
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    if since:
        must.append(
            FieldCondition(
                key="fm_created_at",
                range=DatetimeRange(gte=date.fromisoformat(since)),
            )
        )

    vectors = embedder.embed([query])
    if not vectors:
        return []

    sparse_vec = _sparse_embedder().embed([query])[0]
    return storage.hybrid_search(
        collection=collection_name,
        dense_vector=vectors[0],
        sparse_indices=sparse_vec.indices,
        sparse_values=sparse_vec.values,
        limit=limit,
        query_filter=Filter(must=cast(Any, must)),
    )


def format_hits_markdown(
    *,
    title: str,
    empty_message: str,
    hits: Sequence[ScoredPoint],
    extra_payload_keys: Sequence[str] = (),
) -> str:
    """Render search hits as Markdown with the canonical layout.

    ``extra_payload_keys`` adds a small "key: value" line under the
    score header so domain tools can surface fields like ``status``
    or ``fm_created_at`` without redefining the renderer.
    """
    if not hits:
        return f"{empty_message}\n"
    lines = [f"# {title}", ""]
    for index, hit in enumerate(hits, start=1):
        payload = dict(hit.payload or {})
        source_path = safe_str(payload, "source_path") or "(unknown source)"
        chunk_index = safe_int(payload, "chunk_index")
        total_chunks = safe_int(payload, "total_chunks")
        chunk_label = (
            f" chunk {chunk_index + 1}/{total_chunks}"
            if chunk_index is not None and total_chunks is not None
            else ""
        )
        lines.append(f"## {index}. [{source_path}]{chunk_label} (score: {float(hit.score):.3f})")
        for key in extra_payload_keys:
            value = payload.get(key)
            if value is not None and value != "":
                lines.append(f"_{key}: {value}_")
        heading_path = safe_str(payload, "heading_path")
        if heading_path:
            lines.append(f"_{heading_path}_")
        lines.append("")
        lines.append(safe_str(payload, "text") or "_(no text stored on this chunk)_")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_limit(limit: int | None) -> int:
    return _DEFAULT_LIMIT if limit is None else limit
