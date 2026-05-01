"""`search_decisions` MCP tool."""

from __future__ import annotations

import re

from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools._helpers import ToolError
from sdet_brain.server.tools.domain._common import (
    format_hits_markdown,
    parse_limit,
    run_category_search,
)

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def search_decisions(
    state: AppState,
    *,
    topic: str,
    since: str | None = None,
    limit: int | None = None,
    collection: str | None = None,
) -> str:
    """Search for chunks tagged ``category=decision``.

    ``since`` is an optional ``YYYY-MM-DD`` lower bound on the
    decision's `fm_created_at` payload field.
    """
    if since is not None and not _ISO_DATE.match(since):
        raise ToolError("`since` must be in YYYY-MM-DD format")

    effective_limit = parse_limit(limit)
    hits = run_category_search(
        state,
        category="decision",
        query=topic,
        limit=effective_limit,
        since=since,
        collection=collection,
    )
    title_suffix = f" since {since}" if since else ""
    return format_hits_markdown(
        title=f"Decisions matching `{topic}`{title_suffix}",
        empty_message=f"No decisions match `{topic}`{title_suffix}.",
        hits=hits,
        extra_payload_keys=("fm_created_at", "status"),
    )
