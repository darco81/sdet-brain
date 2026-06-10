"""`multi_query_search` MCP tool - agentic retrieval (T4-04).

Pattern: hand a complex / multi-hop query to the Thinking model, ask
it to decompose into 3-5 focused sub-queries, hybrid-search each in
turn, fuse the ranked lists with Reciprocal Rank Fusion, and return
the merged top-K. The caller sees both the decomposition and the
matched chunks so audits don't lose the LLM's reasoning trace.

The fusion is intentionally simple - RRF on rank, no score weighting.
The dense+BM25 hybrid path each sub-query takes already does the
heavy lifting; this tool's job is just to make sure we ask the right
questions.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

from qdrant_client.models import ScoredPoint

from sdet_brain.embeddings.sparse_embedder import (
    FastembedBM25,
    get_sparse_embedder,
)
from sdet_brain.llm import LLMError
from sdet_brain.llm.factory import get_router
from sdet_brain.llm.protocol import ChatMessage
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

logger = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a brand-aware RAG system. "
    "Given a complex or multi-hop question, return JSON with a single key "
    "`queries` whose value is an array of 3 to 5 short, focused sub-queries "
    "that together cover the original question. Each sub-query should be a "
    "phrase a search engine would handle well (3-12 words). Use the "
    "language of the input question. Reply with the JSON only."
)

_DEFAULT_PER_QUERY_LIMIT = 8
_DEFAULT_FINAL_LIMIT = 5
_RRF_K = 60  # Standard RRF constant; smaller = more weight on top ranks.

_SPARSE: FastembedBM25 | None = None


def _sparse() -> FastembedBM25:
    global _SPARSE
    if _SPARSE is None:
        _SPARSE = get_sparse_embedder()
    return _SPARSE


def _decompose(query: str) -> list[str]:
    """Ask the reasoning model to split ``query`` into sub-queries.

    Returns the original query as a single-element list when the LLM
    output isn't parseable JSON or doesn't contain a `queries` array,
    so the tool always falls back to the bare-query path instead of
    failing.
    """
    try:
        raw = (
            get_router()
            .chat(
                [
                    ChatMessage(role="system", content=_DECOMPOSE_SYSTEM),
                    ChatMessage(role="user", content=query),
                ],
                task="decompose",
                max_tokens=512,
                temperature=0.3,
            )
            .strip()
        )
    except LLMError as exc:
        logger.warning("decomposition failed, falling back to bare query: %s", exc)
        return [query]

    json_payload = _extract_json(raw)
    if json_payload is None:
        logger.warning("decomposition produced no parseable JSON: %r", raw[:200])
        return [query]
    try:
        data = json.loads(json_payload)
    except json.JSONDecodeError:
        return [query]
    queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(queries, list):
        return [query]
    cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    return cleaned[:5] if cleaned else [query]


def _extract_json(text: str) -> str | None:
    """Pull the first ``{...}`` JSON block out of a possibly-chatty LLM reply."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    bare = re.search(r"\{.*\}", text, re.DOTALL)
    if bare:
        return bare.group(0)
    return None


def _rrf_merge(
    ranked_lists: Sequence[Sequence[ScoredPoint]],
    *,
    limit: int,
) -> list[ScoredPoint]:
    """Reciprocal Rank Fusion across multiple sub-query result lists."""
    scores: dict[str, float] = {}
    points_by_id: dict[str, ScoredPoint] = {}
    for results in ranked_lists:
        for rank, point in enumerate(results, start=1):
            point_id = str(point.id)
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (_RRF_K + rank)
            points_by_id.setdefault(point_id, point)
    ordered_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)
    merged: list[ScoredPoint] = []
    for point_id in ordered_ids[:limit]:
        point = points_by_id[point_id]
        # Surface the RRF score on a copy so downstream renderers can use it.
        point.score = float(scores[point_id])
        merged.append(point)
    return merged


def multi_query_search(
    state: AppState,
    *,
    query: str,
    limit: int = _DEFAULT_FINAL_LIMIT,
    per_query_limit: int = _DEFAULT_PER_QUERY_LIMIT,
    source_type: str | None = None,
    collection: str | None = None,
) -> str:
    """Decompose ``query``, hybrid-search each sub-query, RRF-fuse results."""
    if not query.strip():
        raise ToolError("query must not be empty")
    if limit <= 0 or limit > 30:
        raise ToolError("limit must be between 1 and 30")
    if per_query_limit <= 0 or per_query_limit > 30:
        raise ToolError("per_query_limit must be between 1 and 30")

    embedder = require_embedder(state)
    storage = require_storage(state)
    collection_name = collection_or_default(collection)
    payload_filter = source_type_filter(source_type)

    sub_queries = _decompose(query)

    sparse = _sparse()
    ranked_lists: list[list[ScoredPoint]] = []
    for sub in sub_queries:
        dense = embedder.embed([sub])
        if not dense:
            continue
        sparse_vec = sparse.embed([sub])[0]
        hits = storage.hybrid_search(
            collection=collection_name,
            dense_vector=dense[0],
            sparse_indices=sparse_vec.indices,
            sparse_values=sparse_vec.values,
            limit=per_query_limit,
            query_filter=payload_filter,
        )
        ranked_lists.append(hits)

    fused = _rrf_merge(ranked_lists, limit=limit)
    return _format(query, sub_queries, fused)


def _format(original: str, sub_queries: list[str], hits: list[ScoredPoint]) -> str:
    lines = [
        f"# multi_query_search for `{original}`",
        "",
        f"_Decomposed into {len(sub_queries)} sub-{'query' if len(sub_queries) == 1 else 'queries'}:_",
        "",
    ]
    lines.extend(f"- `{sq}`" for sq in sub_queries)
    lines.append("")
    lines.append("---")
    lines.append("")
    if not hits:
        lines.append("No matches across the decomposed sub-queries.")
        lines.append("")
        return "\n".join(lines)
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
        lines.append(f"## {index}. [{source_path}]{chunk_label} (rrf: {float(hit.score):.4f})")
        heading = safe_str(payload, "heading_path")
        if heading:
            lines.append(f"_{heading}_")
        lines.append("")
        lines.append(safe_str(payload, "text") or "_(no text on chunk)_")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
