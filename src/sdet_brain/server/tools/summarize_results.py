"""`summarize_results` MCP tool - LLM summary over retrieved chunks.

Runs the standard hybrid search, then asks the local LLM to produce
a single concise summary that cites the source files. Use this when
the user wants the *answer* rather than a list of chunks - e.g.
"summarize my decisions about CI from last week" instead of "show
me the decision chunks".
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
    safe_str,
    source_type_filter,
)

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = (
    "You are a brand-aware summarizer for the SDET Brain corpus. Given "
    "a topic and a numbered list of retrieved passages with [source] "
    "citations, write a concise summary in the language of the topic "
    "(Polish if the topic is Polish, English otherwise). Cite sources "
    "inline as [n] referring to the passage numbers. Be direct, "
    "honest, no marketing fluff. If the passages don't actually answer "
    "the topic, say so explicitly."
)

_SPARSE: FastembedBM25 | None = None


def _sparse() -> FastembedBM25:
    global _SPARSE
    if _SPARSE is None:
        _SPARSE = get_sparse_embedder()
    return _SPARSE


def summarize_results(
    state: AppState,
    *,
    topic: str,
    limit: int = 8,
    source_type: str | None = None,
    collection: str | None = None,
) -> str:
    """Hybrid-search ``topic``, then LLM-summarize the top chunks."""
    if not topic.strip():
        raise ToolError("topic must not be empty")
    if limit <= 0 or limit > 30:
        raise ToolError("limit must be between 1 and 30")

    embedder = require_embedder(state)
    storage = require_storage(state)
    collection_name = collection_or_default(collection)

    dense = embedder.embed([topic])
    if not dense:
        return f"No matches for `{topic}`.\n"
    sparse_vec = _sparse().embed([topic])[0]
    hits = storage.hybrid_search(
        collection=collection_name,
        dense_vector=dense[0],
        sparse_indices=sparse_vec.indices,
        sparse_values=sparse_vec.values,
        limit=limit,
        query_filter=source_type_filter(source_type),
    )
    if not hits:
        return f"No matches for `{topic}`.\n"

    passages = [
        f"[{idx}] [{safe_str(dict(hit.payload or {}), 'source_path') or '(unknown)'}] "
        f"{safe_str(dict(hit.payload or {}), 'text')}"
        for idx, hit in enumerate(hits, start=1)
    ]
    user_payload = (
        f"Topic: {topic}\n\nPassages:\n" + "\n\n".join(passages)
    )
    try:
        summary = (
            get_router()
            .chat(
                [
                    ChatMessage(role="system", content=_SUMMARY_SYSTEM),
                    ChatMessage(role="user", content=user_payload),
                ],
                task="summarize",
                max_tokens=512,
                temperature=0.4,
            )
            .strip()
        )
    except LLMError as exc:
        raise ToolError(f"summarize failed: {exc}") from exc
    if not summary:
        summary = "_(LLM returned empty summary)_"

    sources = sorted(
        {
            safe_str(dict(hit.payload or {}), "source_path")
            for hit in hits
            if safe_str(dict(hit.payload or {}), "source_path")
        }
    )
    lines = [
        f"# Summary for `{topic}`",
        "",
        summary,
        "",
        "## Sources",
        "",
    ]
    lines.extend(f"- `{src}`" for src in sources)
    lines.append("")
    return "\n".join(lines)
