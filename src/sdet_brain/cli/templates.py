"""Saved query templates (T5-03).

YAML files declare a saved query: which MCP-style tool to invoke, the
default tool args, and a Jinja2-substituted ``query_template``. The
loader scans the user's `~/.sdet-brain/templates/` directory plus the
shipped `examples/templates/` directory so a fresh checkout of the
repo gives you something to run on day one.

Tools are dispatched against the locally configured app state (no
HTTP) so a CLI invocation reuses the same lazy MLX caches the live
daemon uses; running the brain server is not required for templates.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from jinja2 import StrictUndefined, Template, UndefinedError
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Project-shipped templates ride along with the repo so a clone has
# something to demonstrate.
EXAMPLES_DIR_NAME = "examples/templates"
USER_DIR_NAME = ".sdet-brain/templates"

OutputFormat = Literal["markdown", "json", "text"]
ToolName = Literal[
    "search",
    "search_voice_samples",
    "search_smaczki",
    "search_decisions",
    "search_sprint_reports",
    "list_articles_by_status",
    "summarize_results",
    "query_rewrite",
    "multi_query_search",
]


class TemplateError(RuntimeError):
    """Raised when a template can't be loaded, rendered, or executed."""


class QueryTemplate(BaseModel):
    """Pydantic schema for a YAML query template."""

    model_config = {"extra": "ignore"}

    name: str = Field(min_length=1)
    description: str = ""
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    query_template: str = Field(min_length=1)
    default_args: dict[str, Any] = Field(default_factory=dict)
    output_format: OutputFormat = "markdown"


@dataclass(frozen=True)
class TemplateLocation:
    """Where a template was loaded from. Used for ``template show`` UX."""

    template: QueryTemplate
    path: Path


def _project_root() -> Path:
    """Return the sdet-brain repo root.

    `pyproject.toml` is the marker. Walking up from this file is the
    most reliable way to find the repo whether we run from the repo
    or from a `pip install`-style entry point.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parent


def _candidate_dirs() -> list[Path]:
    """Directories scanned for `*.yaml` templates, in priority order."""
    user_dir = Path.home() / USER_DIR_NAME
    examples_dir = _project_root() / EXAMPLES_DIR_NAME
    return [user_dir, examples_dir]


def _load_yaml(path: Path) -> QueryTemplate:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TemplateError(f"Template {path} did not parse as a mapping.")
    try:
        return QueryTemplate.model_validate(raw)
    except ValidationError as exc:
        raise TemplateError(f"Template {path} failed schema: {exc}") from exc


def discover_templates(extra_dirs: Iterable[Path] = ()) -> list[TemplateLocation]:
    """Return every template YAML reachable on the search path.

    User templates win when the same ``name`` exists in both
    `~/.sdet-brain/templates/` and `examples/templates/`. ``extra_dirs``
    is honoured first (test isolation hook).
    """
    seen: dict[str, TemplateLocation] = {}
    for directory in [*extra_dirs, *_candidate_dirs()]:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.yaml")):
            try:
                tpl = _load_yaml(path)
            except TemplateError as exc:
                logger.warning("skipping malformed template %s: %s", path, exc)
                continue
            seen.setdefault(tpl.name, TemplateLocation(template=tpl, path=path))
    return list(seen.values())


def find_template(name: str, extra_dirs: Iterable[Path] = ()) -> TemplateLocation:
    """Locate a template by ``name`` or raise :class:`TemplateError`."""
    for loc in discover_templates(extra_dirs=extra_dirs):
        if loc.template.name == name:
            return loc
    raise TemplateError(f"No template named {name!r} found.")


def render_query(template: QueryTemplate, **variables: Any) -> str:
    """Apply Jinja2 substitution to ``template.query_template``.

    Variables fall back to ``template.default_args``; unrecognised
    variables in the template raise :class:`TemplateError` (StrictUndefined).
    """
    merged = {**template.default_args, **variables}
    try:
        return Template(template.query_template, undefined=StrictUndefined).render(**merged)
    except UndefinedError as exc:
        raise TemplateError(
            f"Missing required variable for template {template.name!r}: {exc}"
        ) from exc


def format_output(fmt: OutputFormat, body: str, sources: list[str] | None = None) -> str:
    """Render the tool's Markdown body in the requested ``output_format``.

    The brain's MCP tools all return Markdown. ``json`` and ``text``
    are convenience pass-throughs for shell pipelines that want a
    minimal envelope or no formatting at all.
    """
    sources = sources or []
    if fmt == "markdown":
        return body
    if fmt == "text":
        return body
    if fmt == "json":
        import json

        return json.dumps({"body": body, "sources": sources}, ensure_ascii=False)
    raise TemplateError(f"unknown output_format {fmt!r}")
