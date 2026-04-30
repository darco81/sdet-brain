"""Map a Markdown file path onto a source-type tag.

The classifier returns one of the values declared in
:class:`sdet_brain.storage.collections.SourceType` plus the open-ended
``"unknown"`` for files outside any registered source. The tag lands in
the chunk payload so downstream tools can filter by category.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

PROJECT_KNOWLEDGE_FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(\d{2}-PROJECT-CONTEXT|\d{2}-BRAND-STRATEGY|EXECUTION-PLAN|LINEAR-ISSUE-TEMPLATE)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceConfig:
    """Path heuristics that drive `classify_source`.

    Each field is a sequence of absolute paths or path-prefix strings.
    The classifier picks the *most specific* match (longest prefix
    wins) so a file inside ``drafts/`` that also matches the
    ``project-knowledge`` filename pattern is tagged as
    ``project-knowledge`` rather than ``drafts``.
    """

    project_knowledge_dirs: tuple[Path, ...] = field(default_factory=tuple)
    drafts_dirs: tuple[Path, ...] = field(default_factory=tuple)
    articles_dirs: tuple[Path, ...] = field(default_factory=tuple)
    sprint_reports_dirs: tuple[Path, ...] = field(default_factory=tuple)
    brief_dirs: tuple[Path, ...] = field(default_factory=tuple)


def _is_under(path: Path, parents: Iterable[Path]) -> bool:
    resolved = path.resolve()
    for parent in parents:
        try:
            resolved.relative_to(parent.resolve())
            return True
        except ValueError:
            continue
    return False


def classify_source(path: Path, config: SourceConfig) -> str:
    """Return the source-type tag for ``path``.

    Resolution order:
    1. project-knowledge filename pattern under any project-knowledge
       directory wins outright.
    2. Otherwise, exact directory containment is checked
       (sprint-reports, brief, articles, drafts).
    3. Default to ``"unknown"``.
    """
    if PROJECT_KNOWLEDGE_FILENAME_PATTERN.match(path.name) and _is_under(
        path, config.project_knowledge_dirs
    ):
        return "project-knowledge"
    if _is_under(path, config.sprint_reports_dirs):
        return "sprint-reports"
    if _is_under(path, config.brief_dirs):
        return "brief"
    if _is_under(path, config.articles_dirs):
        return "articles"
    if _is_under(path, config.drafts_dirs):
        return "drafts"
    return "unknown"


def default_source_config_from_mapping(mapping: Mapping[str, list[str]]) -> SourceConfig:
    """Build a `SourceConfig` from a serialisable mapping.

    The mapping uses the source-type tag as the key. Unknown keys are
    ignored so callers can tag their own categories without breaking.
    """
    return SourceConfig(
        project_knowledge_dirs=tuple(Path(p) for p in mapping.get("project-knowledge", [])),
        drafts_dirs=tuple(Path(p) for p in mapping.get("drafts", [])),
        articles_dirs=tuple(Path(p) for p in mapping.get("articles", [])),
        sprint_reports_dirs=tuple(Path(p) for p in mapping.get("sprint-reports", [])),
        brief_dirs=tuple(Path(p) for p in mapping.get("brief", [])),
    )
