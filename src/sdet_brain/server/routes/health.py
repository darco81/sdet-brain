"""Liveness + readiness endpoint."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from sdet_brain.server.dependencies import AppState, get_state
from sdet_brain.storage.collections import COLLECTION_NAME

logger = logging.getLogger(__name__)
router = APIRouter()

HealthStatus = Literal["ok", "degraded", "unavailable"]


class HealthResponse(BaseModel):
    status: HealthStatus
    qdrant_ok: bool
    embedder_ok: bool
    embedder_provider: str | None
    embedder_fell_back: bool
    vector_size: int | None
    collection_count: int | None
    qdrant_error: str | None = None
    embedder_error: str | None = None


def _qdrant_summary(state: AppState) -> tuple[bool, int | None, str | None]:
    if state.storage is None:
        return False, None, state.qdrant_error or "Qdrant client not initialised."
    try:
        count = state.storage.count(COLLECTION_NAME, exact=False)
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return False, None, str(exc)
    return True, count, None


def _embedder_summary(state: AppState) -> tuple[bool, str | None]:
    if state.selection is None:
        return False, state.embedder_error or "Embedder not initialised."
    try:
        ok = state.selection.embedder.health_check()
    except Exception as exc:
        logger.warning("Embedder health check failed: %s", exc)
        return False, str(exc)
    return ok, None if ok else "Embedder health probe returned False."


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def get_health(state: AppState = Depends(get_state)) -> HealthResponse:
    qdrant_ok, collection_count, qdrant_err = _qdrant_summary(state)
    embedder_ok, embedder_err = _embedder_summary(state)
    overall: HealthStatus = (
        "ok"
        if (qdrant_ok and embedder_ok)
        else "degraded"
        if (qdrant_ok or embedder_ok)
        else "unavailable"
    )
    selection = state.selection
    return HealthResponse(
        status=overall,
        qdrant_ok=qdrant_ok,
        embedder_ok=embedder_ok,
        embedder_provider=None if selection is None else selection.provider,
        embedder_fell_back=False if selection is None else selection.fell_back,
        vector_size=None if selection is None else selection.embedder.vector_size,
        collection_count=collection_count,
        qdrant_error=qdrant_err,
        embedder_error=embedder_err,
    )
