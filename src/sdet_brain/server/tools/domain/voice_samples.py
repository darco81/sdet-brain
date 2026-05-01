"""`search_voice_samples` MCP tool."""

from __future__ import annotations

from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools.domain._common import (
    format_hits_markdown,
    parse_limit,
    run_category_search,
)


def search_voice_samples(
    state: AppState,
    *,
    topic: str,
    limit: int | None = None,
    collection: str | None = None,
) -> str:
    """Search for chunks tagged ``category=voice-sample``."""
    effective_limit = parse_limit(limit)
    hits = run_category_search(
        state,
        category="voice-sample",
        query=topic,
        limit=effective_limit,
        collection=collection,
    )
    return format_hits_markdown(
        title=f"Voice samples matching `{topic}`",
        empty_message=f"No voice samples match `{topic}`.",
        hits=hits,
        extra_payload_keys=("language",),
    )
