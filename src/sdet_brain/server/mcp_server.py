"""FastMCP wrapper exposing the SDET Brain server as MCP tools.

The four core tools (`search`, `ingest_path`, `list_sources`,
`get_chunk_neighbors`) close over a `state_getter` callable so the
same tool implementations work across the FastAPI mount, the stdio
entrypoint, and the SSE entrypoint without any of them having to share
mutable state.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastmcp import FastMCP

from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools.domain import (
    list_articles_by_status as list_articles_by_status_tool,
)
from sdet_brain.server.tools.domain import (
    search_decisions as search_decisions_tool,
)
from sdet_brain.server.tools.domain import (
    search_smaczki as search_smaczki_tool,
)
from sdet_brain.server.tools.domain import (
    search_sprint_reports as search_sprint_reports_tool,
)
from sdet_brain.server.tools.domain import (
    search_voice_samples as search_voice_samples_tool,
)
from sdet_brain.server.tools.get_chunk_neighbors import (
    get_chunk_neighbors as get_chunk_neighbors_tool,
)
from sdet_brain.server.tools.ingest import ingest_path as ingest_path_tool
from sdet_brain.server.tools.list_sources import list_sources as list_sources_tool
from sdet_brain.server.tools.search import search as search_tool

logger = logging.getLogger(__name__)

StateGetter = Callable[[], AppState | None]


def build_mcp(state_getter: StateGetter | None = None) -> FastMCP:
    """Construct the FastMCP instance and register the four core tools.

    Pass `state_getter=None` for use cases that only want a configured
    `FastMCP` instance (e.g. unit tests that don't need to invoke
    tools). Tool calls in that mode raise a clear runtime error.
    """
    if state_getter is None:
        def state_getter() -> AppState | None:  # pragma: no cover - default fallback
            return None

    mcp: FastMCP = FastMCP(
        name="sdet-brain",
        instructions=(
            "Persistent RAG for the SDET brand domain. Tools query a Qdrant "
            "collection of Markdown chunks (drafts, articles, sprint reports, "
            "project knowledge). Prefer narrowing with `source_type` when the "
            "user signals a category."
        ),
    )

    @mcp.tool
    def ping() -> dict[str, str]:
        """Cheap liveness probe confirming the MCP transport works."""
        return {"status": "ok", "service": "sdet-brain"}

    @mcp.tool
    def search(
        query: str,
        limit: int = 5,
        source_type: str | None = None,
        min_score: float = 0.0,
    ) -> str:
        """Semantic search across the SDET brand corpus.

        Use this when the user wants to find passages that talk about a
        topic, voice sample, decision, or sprint outcome. Optional
        `source_type` filter accepts one of ``project-knowledge``,
        ``drafts``, ``articles``, ``sprint-reports``. Lower
        ``min_score`` (0.0-0.4) for exploratory queries; raise it
        (0.6+) when the user wants only highly relevant chunks.
        """
        state = _require_state(state_getter())
        return search_tool(
            state, query=query, limit=limit, source_type=source_type, min_score=min_score
        )

    @mcp.tool
    def ingest_path(path: str, force: bool = False) -> str:
        """Re-ingest a Markdown file or directory into the brain.

        Use this when the user wants to refresh the index after editing
        a file by hand or after dropping a new note in the corpus. Set
        ``force=true`` to bypass the content-hash cache.
        """
        state = _require_state(state_getter())
        return ingest_path_tool(state, path=path, force=force)

    @mcp.tool
    def list_sources(source_type: str | None = None) -> str:
        """List every Markdown file currently indexed in the brain.

        Use this when the user asks "what's in the brain?" or wants to
        narrow a follow-up search. Optional ``source_type`` filter
        scopes the listing to one category.
        """
        state = _require_state(state_getter())
        return list_sources_tool(state, source_type=source_type)

    @mcp.tool
    def get_chunk_neighbors(
        source_path: str,
        chunk_index: int,
        window: int = 2,
    ) -> str:
        """Return the neighbouring chunks around a given chunk in a file.

        Use this after `search` when the user wants more context. The
        function returns chunks in the closed range
        ``[chunk_index - window, chunk_index + window]`` clamped to
        the file's bounds.
        """
        state = _require_state(state_getter())
        return get_chunk_neighbors_tool(
            state,
            source_path=source_path,
            chunk_index=chunk_index,
            window=window,
        )

    @mcp.tool
    def search_voice_samples(topic: str, limit: int = 5) -> str:
        """Find Dariusz's authentic voice samples for a given topic.

        Use this when the user wants quotable phrasing in his style:
        openers, closers, transitions, structural variety, hooks. The
        tool filters to chunks tagged ``category=voice-sample`` so the
        result is voice material only - never strategy docs or
        sprint reports. Prefer this over `search` whenever the user
        asks "how does Dariusz say X" or "find me a self-deprecating
        opener".
        """
        state = _require_state(state_getter())
        return search_voice_samples_tool(state, topic=topic, limit=limit)

    @mcp.tool
    def search_smaczki(topic: str, limit: int = 5) -> str:
        """Find vivid sentence-level "smaczki" (zingers) about a topic.

        "Smaczki" are the bite-sized, quotable beats that land in
        articles - one-liners, sharp metaphors, recurring motifs. Use
        this when the user wants colour for an article: ``"smaczki
        about flaky tests"`` or ``"give me the smaczki about my
        keyboard-trap detector"``. Filters to ``category=smaczki``
        only - not the case study, not the raw notes.
        """
        state = _require_state(state_getter())
        return search_smaczki_tool(state, topic=topic, limit=limit)

    @mcp.tool
    def search_decisions(
        topic: str,
        since: str | None = None,
        limit: int = 5,
    ) -> str:
        """Find prior decisions / verdicts / policies on a topic.

        Use this when the user asks "what did we decide about X?",
        "have we resolved Y?", or "is there a policy on Z?". Filters
        to ``category=decision``. Optional ``since`` (``YYYY-MM-DD``)
        scopes the results to decisions made on or after that date,
        so you can answer "what decisions did we ship this week?".
        """
        state = _require_state(state_getter())
        return search_decisions_tool(state, topic=topic, since=since, limit=limit)

    @mcp.tool
    def list_articles_by_status(status: str, series: str | None = None) -> str:
        """List case-study articles in a given workflow ``status``.

        Use this when the user wants a stocktake: "what's still in
        draft?", "show me the published case studies", "what's in
        review for the WCAG toolkit series?". ``status`` must be one
        of ``draft``, ``review``, ``published``, ``archive``. Optional
        ``series`` (e.g. ``wcag-toolkit``) narrows the listing. The
        result is grouped by file - one row per article, not per
        chunk.
        """
        state = _require_state(state_getter())
        return list_articles_by_status_tool(state, status=status, series=series)

    @mcp.tool
    def search_sprint_reports(
        query: str,
        project: str | None = None,
        limit: int = 5,
    ) -> str:
        """Find sprint reports about a topic, optionally per-project.

        Use this when the user asks "what shipped in last week's
        sprint?", "how did the deploy sprint go?", or "summarize
        sprint outcomes for the WCAG toolkit". Filters to
        ``category=sprint-report``. ``project`` matches the ``series``
        payload (``wcag-toolkit``, ``sdet-brain``, ``portfolio-v2``,
        ``jarvis-brain``) so cross-project sprint queries are easy.
        """
        state = _require_state(state_getter())
        return search_sprint_reports_tool(
            state, query=query, project=project, limit=limit
        )

    return mcp


def _require_state(state: AppState | None) -> AppState:
    if state is None:
        raise RuntimeError(
            "SDET Brain MCP tools have no AppState - check the server lifespan."
        )
    return state
