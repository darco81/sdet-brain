"""Internal helpers shared by the MCP tool implementations."""

from __future__ import annotations

from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from sdet_brain.embeddings.protocol import IEmbedder
from sdet_brain.server.dependencies import AppState
from sdet_brain.storage.collections import COLLECTION_NAME
from sdet_brain.storage.qdrant_client import QdrantStorage


class ToolError(RuntimeError):
    """Surface-level error message returned to the MCP client."""


def require_storage(state: AppState) -> QdrantStorage:
    if state.storage is None:
        raise ToolError(state.qdrant_error or "Qdrant is not available; ingest cannot proceed.")
    return state.storage


def require_embedder(state: AppState) -> IEmbedder:
    if state.embedder is None:
        raise ToolError(state.embedder_error or "Embedder is not available; search cannot proceed.")
    return state.embedder


def source_type_filter(value: str | None) -> Filter | None:
    if not value:
        return None
    return Filter(must=[FieldCondition(key="source_type", match=MatchValue(value=value))])


def source_path_filter(value: str) -> Filter:
    return Filter(must=[FieldCondition(key="source_path", match=MatchValue(value=value))])


def collection_or_default(name: str | None = None) -> str:
    return name or COLLECTION_NAME


def safe_str(payload: dict[str, Any] | None, key: str, default: str = "") -> str:
    if not payload:
        return default
    value = payload.get(key)
    return value if isinstance(value, str) else default


def safe_int(payload: dict[str, Any] | None, key: str) -> int | None:
    if not payload:
        return None
    value = payload.get(key)
    return value if isinstance(value, int) else None
