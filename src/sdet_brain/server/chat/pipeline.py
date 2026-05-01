"""End-to-end chat pipeline (T3-01).

Stateless on the server: every request carries the full conversation
history. The pipeline:
  1. Embeds the latest user turn (dense + BM25).
  2. Hybrid-searches the brain corpus for context (when ``retrieve`` is
     true).
  3. Stitches a system prompt + retrieved context + the conversation
     and asks the local LLM for a reply.
  4. Returns either the full reply (`respond`) or an iterator of
     incremental tokens (`respond_stream`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from sdet_brain.embeddings.protocol import IEmbedder
from sdet_brain.embeddings.sparse_embedder import ISparseEmbedder
from sdet_brain.llm.protocol import ILLM, ChatMessage
from sdet_brain.server.chat.models import ChatRequest, Source
from sdet_brain.server.chat.prompt_template import SYSTEM_PROMPT, format_context
from sdet_brain.storage.collections import COLLECTION_NAME
from sdet_brain.storage.qdrant_client import QdrantStorage

_SNIPPET_CHARS = 200

logger = logging.getLogger(__name__)


class ChatPipeline:
    """Build chat replies on top of hybrid retrieval + local LLM."""

    def __init__(
        self,
        *,
        embedder: IEmbedder,
        sparse_embedder: ISparseEmbedder,
        storage: QdrantStorage,
        llm: ILLM,
        collection: str = COLLECTION_NAME,
    ) -> None:
        self._embedder = embedder
        self._sparse = sparse_embedder
        self._storage = storage
        self._llm = llm
        self._collection = collection

    def _retrieve_context(
        self, query: str, top_k: int
    ) -> tuple[list[tuple[str, str]], list[Source]]:
        """Return ``(numbered_passages, sources)`` for ``query``.

        ``sources`` is a list of :class:`Source` objects in the same
        order as ``passages`` so the index in either list maps onto
        the 1-based ``[N]`` citation marker the LLM is asked to use.
        """
        if not query.strip():
            return [], []
        dense = self._embedder.embed([query])
        if not dense:
            return [], []
        sparse_vec = self._sparse.embed([query])[0]
        hits = self._storage.hybrid_search(
            collection=self._collection,
            dense_vector=dense[0],
            sparse_indices=sparse_vec.indices,
            sparse_values=sparse_vec.values,
            limit=top_k,
        )
        passages: list[tuple[str, str]] = []
        sources: list[Source] = []
        for n, hit in enumerate(hits, start=1):
            payload = dict(hit.payload or {})
            raw_source = payload.get("source_path")
            raw_text = payload.get("text")
            raw_chunk = payload.get("chunk_index")
            source_path = raw_source if isinstance(raw_source, str) else "(unknown)"
            text = raw_text if isinstance(raw_text, str) else ""
            chunk_index = raw_chunk if isinstance(raw_chunk, int) else None
            passages.append((source_path, text))
            sources.append(
                Source(
                    n=n,
                    source_path=source_path,
                    chunk_index=chunk_index,
                    score=float(hit.score),
                    snippet=text[:_SNIPPET_CHARS].strip(),
                )
            )
        return passages, sources

    def _build_messages(
        self, request: ChatRequest
    ) -> tuple[list[ChatMessage], list[Source], int]:
        """Stitch system prompt + retrieved context + history."""
        latest_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        passages: list[tuple[str, str]] = []
        sources: list[Source] = []
        if request.retrieve:
            passages, sources = self._retrieve_context(latest_user, request.top_k)

        system_payload = SYSTEM_PROMPT
        context_block = format_context(passages)
        if context_block:
            system_payload = f"{SYSTEM_PROMPT}\n{context_block}"

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_payload)
        ]
        for turn in request.messages:
            messages.append(ChatMessage(role=turn.role, content=turn.content))
        return messages, sources, len(passages)

    def respond(self, request: ChatRequest) -> tuple[str, list[Source], int]:
        """Single-shot reply."""
        messages, sources, retrieved_count = self._build_messages(request)
        reply = self._llm.chat(messages, max_tokens=request.max_tokens)
        return reply, sources, retrieved_count

    def respond_stream(
        self, request: ChatRequest
    ) -> tuple[Iterator[str], list[Source], int]:
        """Token-by-token streaming variant.

        Returns the iterator plus the citations / count so the caller
        can emit a final SSE frame with the source list.
        """
        messages, sources, retrieved_count = self._build_messages(request)
        stream = self._llm.chat_stream(messages, max_tokens=request.max_tokens)
        return stream, sources, retrieved_count


def _coerce_storage(state: Any) -> QdrantStorage:
    """Helper used by the route to fetch a storage instance from app state."""
    storage = getattr(state, "storage", None)
    if storage is None:
        raise RuntimeError("Storage not available")
    if not isinstance(storage, QdrantStorage):
        raise RuntimeError(f"Unexpected storage type: {type(storage).__name__}")
    return storage
