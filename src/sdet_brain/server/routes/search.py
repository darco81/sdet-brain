"""Dense-vector search endpoint with optional cross-encoder reranking."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, Field
from qdrant_client.models import FieldCondition, Filter, MatchValue

from sdet_brain.config import Settings, get_settings
from sdet_brain.embeddings.protocol import IEmbedder
from sdet_brain.embeddings.reranker import (
    FastembedReranker,
    RerankCandidate,
)
from sdet_brain.embeddings.sparse_embedder import (
    FastembedBM25,
    get_sparse_embedder,
)
from sdet_brain.server.dependencies import require_embedder, require_storage
from sdet_brain.storage.collections import COLLECTION_NAME
from sdet_brain.storage.qdrant_client import QdrantStorage

router = APIRouter()

# Single shared lazy reranker instance per process. Built on first
# request that opts in; subsequent requests reuse the warm encoder.
_RERANKER: FastembedReranker | None = None
_RERANKER_MODEL: str | None = None

# Single shared lazy BM25 sparse embedder. Built once per process so
# hybrid queries don't reload the tokenizer.
_SPARSE: FastembedBM25 | None = None


def _get_reranker(settings: Settings) -> FastembedReranker:
    global _RERANKER, _RERANKER_MODEL
    if _RERANKER is None or settings.rerank_model != _RERANKER_MODEL:
        _RERANKER = FastembedReranker(
            model_name=settings.rerank_model,
            top_k_default=settings.rerank_top_k_return,
        )
        _RERANKER_MODEL = settings.rerank_model
    return _RERANKER


def _get_sparse_embedder() -> FastembedBM25:
    global _SPARSE
    if _SPARSE is None:
        _SPARSE = get_sparse_embedder()
    return _SPARSE


class SearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1)]
    limit: Annotated[int, Field(ge=1, le=50)] = 10
    source_type_filter: str | None = None
    score_threshold: float | None = None
    rerank: bool | None = Field(
        default=None,
        description=(
            "Override RERANK_ENABLED for this request. None = use settings default."
        ),
    )
    hybrid: bool = Field(
        default=True,
        description=(
            "Hybrid (dense + BM25 RRF) is the default. Set false to "
            "run a dense-only query (used by benchmarks)."
        ),
    )


class SearchResultItem(BaseModel):
    id: str
    score: float
    source_path: str
    source_type: str | None
    heading_path: str | None
    text: str
    chunk_index: int | None
    total_chunks: int | None


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchResultItem]
    reranked: bool = False
    hybrid: bool = False


def _build_filter(source_type_filter: str | None) -> Filter | None:
    if not source_type_filter:
        return None
    return Filter(
        must=[
            FieldCondition(key="source_type", match=MatchValue(value=source_type_filter))
        ]
    )


def _point_to_item(point: object) -> SearchResultItem:
    payload = getattr(point, "payload", None) or {}
    return SearchResultItem(
        id=str(getattr(point, "id", "")),
        score=float(getattr(point, "score", 0.0)),
        source_path=str(payload.get("source_path", "")),
        source_type=payload.get("source_type") if isinstance(payload.get("source_type"), str) else None,
        heading_path=payload.get("heading_path") if isinstance(payload.get("heading_path"), str) else None,
        text=str(payload.get("text", "")),
        chunk_index=payload.get("chunk_index") if isinstance(payload.get("chunk_index"), int) else None,
        total_chunks=payload.get("total_chunks") if isinstance(payload.get("total_chunks"), int) else None,
    )


@router.post("/search", response_model=SearchResponse, tags=["search"])
def post_search(
    body: Annotated[SearchRequest, Body()],
    storage: QdrantStorage = Depends(require_storage),
    embedder: IEmbedder = Depends(require_embedder),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    rerank_active = body.rerank if body.rerank is not None else settings.rerank_enabled
    # When reranking, over-fetch from Qdrant so the cross-encoder has a
    # broader candidate set to re-rank from. Else honour the request limit.
    fetch_limit = (
        max(body.limit, settings.rerank_top_k_retrieve) if rerank_active else body.limit
    )

    vectors = embedder.embed([body.query])
    if not vectors:
        return SearchResponse(
            query=body.query, count=0, results=[], reranked=False, hybrid=body.hybrid
        )

    query_filter = _build_filter(body.source_type_filter)
    if body.hybrid:
        sparse = _get_sparse_embedder().embed([body.query])
        sparse_vec = sparse[0]
        points = storage.hybrid_search(
            collection=COLLECTION_NAME,
            dense_vector=vectors[0],
            sparse_indices=sparse_vec.indices,
            sparse_values=sparse_vec.values,
            limit=fetch_limit,
            query_filter=query_filter,
        )
    else:
        points = storage.search(
            collection=COLLECTION_NAME,
            query_vector=vectors[0],
            limit=fetch_limit,
            query_filter=query_filter,
            score_threshold=body.score_threshold,
        )

    items = [_point_to_item(p) for p in points]

    if rerank_active and items:
        reranker = _get_reranker(settings)
        candidates = [RerankCandidate(text=item.text or "", payload=item) for item in items]
        rerank_results = reranker.rerank(body.query, candidates, top_k=body.limit)
        items = [
            # Replace the dense-vector score with the rerank score so
            # downstream consumers can sort / threshold consistently.
            r.payload.model_copy(update={"score": r.score})
            for r in rerank_results
        ]
        return SearchResponse(
            query=body.query,
            count=len(items),
            results=items,
            reranked=True,
            hybrid=body.hybrid,
        )

    return SearchResponse(
        query=body.query,
        count=len(items[: body.limit]),
        results=items[: body.limit],
        reranked=False,
        hybrid=body.hybrid,
    )
