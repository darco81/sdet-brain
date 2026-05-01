"""Structured frontmatter schema for the SDET brand corpus (T2-01).

`BrandFrontmatter` is the typed view we want every Markdown file in the
brand corpus to declare in its YAML header. Validation is *graceful* -
files that don't match the schema are still ingested, but only the
matching subset of fields gets lifted into top-level Qdrant payload
keys for fast filtering. The raw header is always preserved verbatim
on the chunk payload so nothing is lost.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

Category = Literal[
    "brand-strategy",
    "execution-plan",
    "voice-sample",
    "smaczki",
    "draft",
    "sprint-report",
    "case-study",
    "decision",
    "outline",
    "raw-notes",
    "prompt",
    "other",
]

Status = Literal["draft", "review", "published", "archive"]

Language = Literal["en", "pl", "mixed"]


class BrandFrontmatter(BaseModel):
    """Pydantic schema for SDET brand corpus YAML headers."""

    model_config = {"extra": "ignore"}

    category: Category
    tags: list[str] = Field(default_factory=list)
    status: Status = "draft"
    series: str | None = None
    episode: int | None = None
    part: int | None = None
    language: Language = "en"
    created_at: date | None = None
    updated_at: date | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: Any) -> Any:
        """Allow ``tags: foo`` (single string) as a stand-in for ``[foo]``."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


def parse_brand_frontmatter(raw: dict[str, object]) -> BrandFrontmatter | None:
    """Validate ``raw`` against :class:`BrandFrontmatter`.

    Returns the validated model on success, ``None`` on failure. The
    validation error is logged at WARNING so a malformed header is
    visible in operator logs without aborting the ingest.
    """
    if not raw:
        return None
    try:
        return BrandFrontmatter.model_validate(raw)
    except ValidationError as exc:
        logger.warning(
            "Frontmatter does not match BrandFrontmatter schema: %s",
            exc.errors(include_url=False),
        )
        return None


def to_payload_fields(model: BrandFrontmatter) -> dict[str, object]:
    """Flatten a validated frontmatter into top-level payload keys.

    Only non-default values are returned so payloads stay compact. Date
    fields are serialised to ISO-8601 strings to match the rest of the
    payload (``created_at`` is already a string elsewhere).
    """
    out: dict[str, object] = {
        "category": model.category,
        "status": model.status,
        "language": model.language,
    }
    if model.tags:
        out["tags"] = list(model.tags)
    if model.series is not None:
        out["series"] = model.series
    if model.episode is not None:
        out["episode"] = model.episode
    if model.part is not None:
        out["part"] = model.part
    if model.created_at is not None:
        out["fm_created_at"] = model.created_at.isoformat()
    if model.updated_at is not None:
        out["fm_updated_at"] = model.updated_at.isoformat()
    return out
