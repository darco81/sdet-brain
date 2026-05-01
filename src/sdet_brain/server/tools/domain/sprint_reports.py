"""`search_sprint_reports` MCP tool."""

from __future__ import annotations

from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools.domain._common import (
    format_hits_markdown,
    parse_limit,
    run_category_search,
)


def search_sprint_reports(
    state: AppState,
    *,
    query: str,
    project: str | None = None,
    limit: int | None = None,
    collection: str | None = None,
) -> str:
    """Search for chunks tagged ``category=sprint-report``.

    ``project`` filters by the ``series`` payload (one of
    ``wcag-toolkit``, ``sdet-brain``, ``portfolio-v2``,
    ``jarvis-brain``).
    """
    effective_limit = parse_limit(limit)
    hits = run_category_search(
        state,
        category="sprint-report",
        query=query,
        limit=effective_limit,
        extra_keyword_filters={"series": project} if project else None,
        collection=collection,
    )
    title_suffix = f" (project={project})" if project else ""
    return format_hits_markdown(
        title=f"Sprint reports matching `{query}`{title_suffix}",
        empty_message=f"No sprint reports match `{query}`{title_suffix}.",
        hits=hits,
        extra_payload_keys=("series", "fm_created_at"),
    )
