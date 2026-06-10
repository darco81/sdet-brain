"""Shared state-construction helpers for the FastAPI app and MCP entrypoints.

`build_default_state` produces a freshly-wired `AppState` from the
runtime settings. The FastAPI app uses it inside its lifespan, while
`mcp_stdio` / `mcp_sse` use it once at process start and then thread
the result into the FastMCP instance.
"""

from __future__ import annotations

import logging

from sdet_brain.config import Settings, get_settings
from sdet_brain.embeddings.factory import get_embedder
from sdet_brain.embeddings.protocol import EmbeddingError
from sdet_brain.ingestion.source_classifier import build_source_config
from sdet_brain.server.dependencies import AppState
from sdet_brain.storage.qdrant_client import QdrantStorage

logger = logging.getLogger(__name__)


def build_default_state(settings: Settings | None = None) -> AppState:
    """Construct an `AppState` honouring the runtime settings.

    Each backend is constructed independently so a missing one (e.g.
    Gemini API key, MLX unavailable) only disables that capability and
    the surviving routes / tools keep serving.
    """
    settings = settings or get_settings()
    state = AppState(
        settings=settings,
        storage=None,
        selection=None,
        source_config=build_source_config(settings),
    )
    try:
        state.storage = QdrantStorage(
            settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    except Exception as exc:
        state.qdrant_error = str(exc)
        logger.warning("Qdrant unavailable at startup: %s", exc)

    try:
        state.selection = get_embedder(settings)
    except EmbeddingError as exc:
        state.embedder_error = str(exc)
        logger.warning("Embedder unavailable at startup: %s", exc)
    return state
