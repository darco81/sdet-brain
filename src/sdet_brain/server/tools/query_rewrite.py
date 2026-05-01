"""`query_rewrite` MCP tool - HyDE-style query expansion (T2-05).

Pattern: feed the user's short query to a local LLM, ask it to draft
a hypothetical answer paragraph, then run the dense+sparse retrieval
against that paragraph instead of the bare query. The LLM-written
hypothetical pulls the embedding closer to the kind of text the
relevant chunks contain, so recall on terse or under-specified queries
goes up materially.

Shape of the result mirrors the regular ``search`` tool's Markdown
output, with a header line documenting the hypothetical so the caller
can audit what the LLM rewrote to.
"""

from __future__ import annotations

import logging

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

_HYDE_SYSTEM = (
    "You are a brand-aware retrieval assistant. Given a short query, "
    "write ONE concise paragraph (3-5 sentences) of the kind of "
    "passage that would best answer it. Reply only with the paragraph, "
    "no preamble. Use the language of the query."
)

# Sparse encoder lazy singleton; the LLM goes through the shared
# router so it picks the fast tier (gemma-4) instead of the 80B
# instruct model - HyDE wants speed, not depth.
_SPARSE: FastembedBM25 | None = None


def _sparse() -> FastembedBM25:
    global _SPARSE
    if _SPARSE is None:
        _SPARSE = get_sparse_embedder()
    return _SPARSE


def query_rewrite(
    state: AppState,
    *,
    query: str,
    limit: int = 5,
    source_type: str | None = None,
    collection: str | None = None,
) -> str:
    """Run HyDE: LLM-rewrite the query, then hybrid-search the rewrite."""
    if not query.strip():
        raise ToolError("query must not be empty")
    if limit <= 0 or limit > 50:
        raise ToolError("limit must be between 1 and 50")

    embedder = require_embedder(state)
    storage = require_storage(state)
    collection_name = collection_or_default(collection)

    try:
        hypothetical = (
            get_router()
            .chat(
                [
                    ChatMessage(role="system", content=_HYDE_SYSTEM),
                    ChatMessage(role="user", content=query),
                ],
                task="fast",
                max_tokens=256,
                temperature=0.5,
            )
            .strip()
        )
    except LLMError as exc:
        raise ToolError(f"query rewrite failed: {exc}") from exc
    if not hypothetical:
        # Fall back to the bare query so retrieval still works.
        hypothetical = query

    dense = embedder.embed([hypothetical])
    if not dense:
        return f"_(empty embedding for rewrite of `{query}`)_\n"
    sparse_vec = _sparse().embed([hypothetical])[0]
    hits = storage.hybrid_search(
        collection=collection_name,
        dense_vector=dense[0],
        sparse_indices=sparse_vec.indices,
        sparse_values=sparse_vec.values,
        limit=limit,
        query_filter=source_type_filter(source_type),
    )
    return _format(query, hypothetical, hits)


def _format(query: str, hypothetical: str, hits: list) -> str:  # type: ignore[type-arg]
    if not hits:
        return (
            f"# query_rewrite for `{query}`\n\n"
            f"_Hypothetical answer used for retrieval:_\n\n"
            f"> {hypothetical}\n\n"
            f"No matches.\n"
        )
    lines = [
        f"# query_rewrite for `{query}`",
        "",
        "_Hypothetical answer used for retrieval (HyDE):_",
        "",
        f"> {hypothetical}",
        "",
        "---",
        "",
    ]
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
        lines.append(
            f"## {index}. [{source_path}]{chunk_label} (score: {float(hit.score):.3f})"
        )
        heading = safe_str(payload, "heading_path")
        if heading:
            lines.append(f"_{heading}_")
        lines.append("")
        lines.append(safe_str(payload, "text") or "_(no text on chunk)_")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
