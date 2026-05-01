"""`search` MCP tool implementation."""

from __future__ import annotations

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
    source_type_filter,
)

# Single shared lazy BM25 instance per process so the MCP path
# matches the FastAPI route's caching behaviour.
_SPARSE: FastembedBM25 | None = None


def _sparse() -> FastembedBM25:
    global _SPARSE
    if _SPARSE is None:
        _SPARSE = get_sparse_embedder()
    return _SPARSE


def search(
    state: AppState,
    *,
    query: str,
    limit: int = 5,
    source_type: str | None = None,
    min_score: float = 0.0,
    collection: str | None = None,
    hybrid: bool = True,
) -> str:
    """Run a hybrid (dense + BM25 RRF) search and format hits as Markdown.

    ``hybrid=False`` falls back to dense-only, retained for benchmark
    parity with v0.1.x. The default is hybrid because exact-keyword
    queries (``"WCAG 2.2 AA"``) get materially better recall.
    """
    if not query.strip():
        raise ToolError("query must not be empty")
    if limit <= 0 or limit > 50:
        raise ToolError("limit must be between 1 and 50")

    embedder = require_embedder(state)
    storage = require_storage(state)
    collection_name = collection_or_default(collection)

    vectors = embedder.embed([query])
    if not vectors:
        return _format_empty(query, source_type)

    if hybrid:
        sparse_vec = _sparse().embed([query])[0]
        results = storage.hybrid_search(
            collection=collection_name,
            dense_vector=vectors[0],
            sparse_indices=sparse_vec.indices,
            sparse_values=sparse_vec.values,
            limit=limit,
            query_filter=source_type_filter(source_type),
        )
    else:
        results = storage.search(
            collection=collection_name,
            query_vector=vectors[0],
            limit=limit,
            query_filter=source_type_filter(source_type),
            score_threshold=min_score if min_score > 0 else None,
        )
    if not results:
        return _format_empty(query, source_type)

    return _format_hits(query, source_type, results)


def _format_empty(query: str, source_type: str | None) -> str:
    suffix = f" (filter source_type={source_type})" if source_type else ""
    return f"No matches for `{query}`{suffix}."


def _format_hits(query: str, source_type: str | None, hits: list) -> str:  # type: ignore[type-arg]
    header_filter = f" (filter source_type={source_type})" if source_type else ""
    lines = [f"# Search results for `{query}`{header_filter}", ""]
    for index, hit in enumerate(hits, start=1):
        payload = dict(hit.payload or {})
        source_path = safe_str(payload, "source_path")
        heading_path = safe_str(payload, "heading_path")
        chunk_index = safe_int(payload, "chunk_index")
        total_chunks = safe_int(payload, "total_chunks")
        text = safe_str(payload, "text")
        location = source_path or "(unknown source)"
        chunk_label = (
            f" chunk {chunk_index + 1}/{total_chunks}"
            if chunk_index is not None and total_chunks is not None
            else ""
        )
        score = float(hit.score)
        lines.append(f"## {index}. [{location}]{chunk_label} (score: {score:.3f})")
        if heading_path:
            lines.append(f"_{heading_path}_")
        lines.append("")
        lines.append(text or "_(no text stored on this chunk)_")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
