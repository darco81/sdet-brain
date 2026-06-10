"""`search` MCP tool implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sdet_brain.embeddings.reranker import (
    FastembedReranker,
    RerankCandidate,
    get_reranker,
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
    source_type_filter,
)

if TYPE_CHECKING:
    from sdet_brain.config import Settings

# Single shared lazy BM25 instance per process so the MCP path
# matches the FastAPI route's caching behaviour.
_SPARSE: FastembedBM25 | None = None

# Single shared lazy reranker, rebuilt only when the configured model changes,
# mirroring the FastAPI /search route so the two paths behave identically.
_RERANKER: FastembedReranker | None = None
_RERANKER_MODEL: str | None = None


def _sparse() -> FastembedBM25:
    global _SPARSE
    if _SPARSE is None:
        _SPARSE = get_sparse_embedder()
    return _SPARSE


def _get_reranker(settings: Settings) -> FastembedReranker:
    global _RERANKER, _RERANKER_MODEL
    if _RERANKER is None or settings.rerank_model != _RERANKER_MODEL:
        _RERANKER = get_reranker(model_name=settings.rerank_model)
        _RERANKER_MODEL = settings.rerank_model
    return _RERANKER


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

    When ``RERANK_ENABLED`` is set the tool over-fetches
    ``rerank_top_k_retrieve`` candidates and re-orders them with the
    cross-encoder before returning the top ``limit`` — matching the
    ``/search`` route.

    ``min_score`` is applied to the **final** result score, in the scale
    of the active pipeline: cosine for ``hybrid=False``, RRF fusion score
    for hybrid (small, ~0.01-0.05), or cross-encoder score when reranking.
    Leave it at 0.0 unless you know which scale you are thresholding.
    """
    if not query.strip():
        raise ToolError("query must not be empty")
    if limit <= 0 or limit > 50:
        raise ToolError("limit must be between 1 and 50")

    embedder = require_embedder(state)
    storage = require_storage(state)
    collection_name = collection_or_default(collection)
    settings = state.settings

    rerank_active = settings.rerank_enabled
    # Over-fetch when reranking so the cross-encoder has a broad candidate set.
    fetch_limit = max(limit, settings.rerank_top_k_retrieve) if rerank_active else limit

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
            limit=fetch_limit,
            query_filter=source_type_filter(source_type),
        )
    else:
        results = storage.search(
            collection=collection_name,
            query_vector=vectors[0],
            limit=fetch_limit,
            query_filter=source_type_filter(source_type),
        )

    if rerank_active and results:
        reranker = _get_reranker(settings)
        candidates = [
            RerankCandidate(text=safe_str(dict(point.payload or {}), "text"), payload=point)
            for point in results
        ]
        results = [
            # Replace the retrieval score with the cross-encoder score so the
            # rendered scores and any min_score threshold are consistent.
            ranked.payload.model_copy(update={"score": ranked.score})
            for ranked in reranker.rerank(query, candidates, top_k=limit)
        ]
    else:
        results = results[:limit]

    if min_score > 0:
        results = [hit for hit in results if float(hit.score) >= min_score]

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
