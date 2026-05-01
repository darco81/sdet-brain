"""Domain-specific MCP tools (T2-02).

Each tool wraps :func:`sdet_brain.server.tools.search.search` with a
preset payload filter so LLMs can pick the right tool for the user's
intent (``"find a voice sample"`` -> ``search_voice_samples``,
``"what did we decide about X"`` -> ``search_decisions``, etc.) without
needing to remember which payload key carries the category tag.
"""

from sdet_brain.server.tools.domain.articles import list_articles_by_status
from sdet_brain.server.tools.domain.decisions import search_decisions
from sdet_brain.server.tools.domain.smaczki import search_smaczki
from sdet_brain.server.tools.domain.sprint_reports import search_sprint_reports
from sdet_brain.server.tools.domain.voice_samples import search_voice_samples

__all__ = [
    "list_articles_by_status",
    "search_decisions",
    "search_smaczki",
    "search_sprint_reports",
    "search_voice_samples",
]
