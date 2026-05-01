"""POST /chat - multi-turn chat with optional SSE streaming."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse

from sdet_brain.embeddings.protocol import IEmbedder
from sdet_brain.embeddings.sparse_embedder import (
    FastembedBM25,
    get_sparse_embedder,
)
from sdet_brain.llm import get_llm
from sdet_brain.llm.protocol import ILLM
from sdet_brain.server.chat.models import ChatRequest, ChatResponse, Source
from sdet_brain.server.chat.pipeline import ChatPipeline
from sdet_brain.server.dependencies import require_embedder, require_storage
from sdet_brain.storage.qdrant_client import QdrantStorage

logger = logging.getLogger(__name__)

router = APIRouter()

# Single shared lazy instances per process so chat doesn't reload BM25
# or the LLM on every request. The MLX cold start is paid once.
_LLM: ILLM | None = None
_SPARSE: FastembedBM25 | None = None


def _llm() -> ILLM:
    global _LLM
    if _LLM is None:
        _LLM = get_llm()
    return _LLM


def _sparse() -> FastembedBM25:
    global _SPARSE
    if _SPARSE is None:
        _SPARSE = get_sparse_embedder()
    return _SPARSE


def _build_pipeline(
    storage: QdrantStorage,
    embedder: IEmbedder,
) -> ChatPipeline:
    return ChatPipeline(
        embedder=embedder,
        sparse_embedder=_sparse(),
        storage=storage,
        llm=_llm(),
    )


def _sse_iter(
    stream: Iterator[str], sources: list[Source], retrieved_count: int
) -> Iterator[str]:
    """Wrap the LLM token stream in SSE frames + a final metadata frame.

    The terminal frame ships the structured ``Source`` list so clients
    can render footnote-style citations next to the inline ``[N]``
    markers the LLM produces.
    """
    for chunk in stream:
        yield f"data: {json.dumps({'text': chunk})}\n\n"
    yield (
        "data: "
        + json.dumps(
            {
                "event": "done",
                "sources": [s.model_dump(mode="json") for s in sources],
                "retrieved": retrieved_count,
            }
        )
        + "\n\n"
    )


@router.post("/chat", tags=["chat"], response_model=None)
def post_chat(
    body: Annotated[ChatRequest, Body()],
    storage: QdrantStorage = Depends(require_storage),
    embedder: IEmbedder = Depends(require_embedder),
) -> ChatResponse | StreamingResponse:
    """Multi-turn chat with optional SSE streaming.

    The latest ``user`` turn is hybrid-searched against the brain
    corpus (unless ``retrieve=false``); retrieved chunks are injected
    into the system prompt so the model can cite them. ``stream=true``
    flips the response to SSE; ``data:`` frames carry incremental
    tokens, with a final ``{"event": "done", ...}`` frame listing
    sources.
    """
    pipeline = _build_pipeline(storage, embedder)
    if body.stream:
        stream, sources, retrieved = pipeline.respond_stream(body)
        return StreamingResponse(
            _sse_iter(stream, sources, retrieved),
            media_type="text/event-stream",
        )
    reply, sources, retrieved = pipeline.respond(body)
    return ChatResponse(
        reply=reply, sources=sources, retrieved_chunk_count=retrieved
    )
