"""Ingest endpoint - thin wrapper around `sdet_brain.ingestion.pipeline`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field

from sdet_brain.embeddings.protocol import IEmbedder
from sdet_brain.ingestion.pipeline import IngestStats, ingest_path
from sdet_brain.server.dependencies import require_embedder, require_storage
from sdet_brain.storage.collections import COLLECTION_NAME
from sdet_brain.storage.qdrant_client import QdrantStorage

router = APIRouter()


class IngestRequest(BaseModel):
    path: Annotated[str, Field(min_length=1)]
    force: bool = False
    batch_size: Annotated[int, Field(ge=1, le=128)] = 32
    exclude_dirs: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    files_processed: int
    files_skipped: int
    chunks_created: int
    chunks_replaced: int
    errors: list[tuple[str, str]]
    summary: str


def _to_response(stats: IngestStats) -> IngestResponse:
    return IngestResponse(
        files_processed=stats.files_processed,
        files_skipped=stats.files_skipped,
        chunks_created=stats.chunks_created,
        chunks_replaced=stats.chunks_replaced,
        errors=list(stats.errors),
        summary=stats.summary(),
    )


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
def post_ingest(
    body: Annotated[IngestRequest, Body()],
    storage: QdrantStorage = Depends(require_storage),
    embedder: IEmbedder = Depends(require_embedder),
) -> IngestResponse:
    target = Path(body.path)
    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path does not exist: {target}",
        )
    stats = ingest_path(
        target,
        storage,
        embedder,
        collection=COLLECTION_NAME,
        batch_size=body.batch_size,
        force_reindex=body.force,
        exclude_dirs=tuple(Path(d) for d in body.exclude_dirs),
    )
    return _to_response(stats)
