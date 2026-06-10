"""Shared FastAPI / FastMCP dependency helpers.

The application keeps a single `AppState` attached to the FastAPI
instance for the duration of its lifespan. Routes (and MCP tools, in
T1-07) reach into the state through the helpers in this module rather
than importing `app.state` directly so they remain testable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request, status

from sdet_brain.ingestion.source_classifier import SourceConfig

if TYPE_CHECKING:
    from sdet_brain.config import Settings
    from sdet_brain.embeddings.factory import EmbedderSelection
    from sdet_brain.embeddings.protocol import IEmbedder
    from sdet_brain.storage.qdrant_client import QdrantStorage


@dataclass
class AppState:
    """Aggregate of long-lived components used by routes and tools."""

    settings: Settings
    storage: QdrantStorage | None
    selection: EmbedderSelection | None
    qdrant_error: str | None = None
    embedder_error: str | None = None
    # Built once from settings so server ingest (MCP tool + /ingest route)
    # classifies re-ingested files instead of tagging them "unknown".
    source_config: SourceConfig = field(default_factory=SourceConfig)

    @property
    def embedder(self) -> IEmbedder | None:
        return None if self.selection is None else self.selection.embedder


def get_state(request: Request) -> AppState:
    state = getattr(request.app.state, "app_state", None)
    if not isinstance(state, AppState):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application state is not initialised.",
        )
    return state


def require_storage(state: AppState = Depends(get_state)) -> QdrantStorage:
    if state.storage is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.qdrant_error or "Qdrant is not available.",
        )
    return state.storage


def require_embedder(state: AppState = Depends(get_state)) -> IEmbedder:
    if state.embedder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.embedder_error or "Embedder is not available.",
        )
    return state.embedder


def iter_dependencies(state: AppState) -> Iterator[object]:
    """Yield the long-lived components for shutdown logging."""
    if state.storage is not None:
        yield state.storage
    if state.selection is not None:
        yield state.selection.embedder
