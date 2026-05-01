"""Suggest a :class:`BrandFrontmatter` from a Markdown file's path.

The migration tool runs this classifier on every file in the brand
corpus that lacks structured frontmatter and writes the suggestion
back as a YAML header. The heuristics are deliberately conservative:
when in doubt the classifier picks ``category=other`` and
``status=draft`` so the human can refine later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sdet_brain.ingestion.frontmatter_schema import (
    BrandFrontmatter,
    Category,
    Language,
    Status,
)

_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}-")
_CASE_STUDY = re.compile(r"^case-study-(\d+)-", re.IGNORECASE)
_EPISODE = re.compile(r"^episode-(\d+)-", re.IGNORECASE)
_PART_SUFFIX = re.compile(r"part[-_]?(\d+)", re.IGNORECASE)
_VERSION_TAG = re.compile(r"^v\d+(?:\.\d+)*", re.IGNORECASE)
# Polish-only diacritics. Five or more in the first 1k chars => Polish.
_PL_CHARS = re.compile(r"[ąćęłńóśźż]")

# Tokens that we drop from the auto-derived tag list because they're
# noise rather than signal (file-naming conventions, dates, etc.).
_TAG_STOPLIST: frozenset[str] = frozenset(
    {
        "md",
        "draft",
        "report",
        "prompt",
        "sprint",
        "fix",
        "round",
        "block",
        "phase",
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "alpha",
        "beta",
        "rc",
        "new",
        "old",
        "final",
        "wip",
        "tmp",
        "the",
        "for",
        "and",
        "with",
    }
)


@dataclass(frozen=True)
class ClassificationResult:
    """Outcome of running the classifier on a single path."""

    frontmatter: BrandFrontmatter
    confidence: str  # "high" | "medium" | "low"
    rationale: str


def _strip_date_prefix(stem: str) -> str:
    return _DATE_PREFIX.sub("", stem)


def _detect_language(text: str) -> Language:
    """Polish vs English detection from the first ~1k characters.

    The brand corpus is bilingual (PL drafts, EN articles, mixed
    decisions). We sample the head of the file because that's where
    titles and intros live, and they are the most reliable signal.
    """
    head = text[:1000]
    pl_hits = len(_PL_CHARS.findall(head))
    if pl_hits >= 5:
        return "pl"
    return "en"


def _classify_category(stem_upper: str, stem: str) -> tuple[Category, str]:
    """Return ``(category, rationale)`` for a filename stem.

    Order matters: the most specific patterns win. The first match
    short-circuits - that's why ``case-study-NN-OUTLINE`` lands on
    ``outline`` rather than ``case-study``.
    """
    if "OUTLINE" in stem_upper:
        return "outline", "filename contains OUTLINE"
    if "RAW-NOTES" in stem_upper or stem_upper.endswith("-NOTES"):
        return "raw-notes", "filename contains RAW-NOTES or ends with -NOTES"
    if "SMACZKI" in stem_upper:
        return "smaczki", "filename contains SMACZKI"
    if (
        "VERDICT" in stem_upper
        or "POLICY" in stem_upper
        or "DECISIONS" in stem_upper
        or "DECISION" in stem_upper
    ):
        return "decision", "filename indicates a verdict/policy/decision"
    if "SPRINT-REPORT" in stem_upper or stem_upper.endswith("REPORT"):
        return "sprint-report", "filename indicates a sprint report"
    if (
        "PROMPT" in stem_upper
        or stem_upper.startswith("CC-")
        or stem_upper.startswith("WRITE-UP")
    ):
        return "prompt", "filename indicates a CC prompt"
    if _CASE_STUDY.match(stem):
        return "case-study", "filename matches case-study-NN-*"
    if "BRAND-STRATEGY" in stem_upper:
        return "brand-strategy", "filename matches *BRAND-STRATEGY*"
    if "EXECUTION-PLAN" in stem_upper:
        return "execution-plan", "filename matches *EXECUTION-PLAN*"
    if "DRAFT" in stem_upper or _EPISODE.match(stem):
        return "draft", "filename indicates an article draft"
    return "other", "no specific pattern matched"


def _derive_tags(stem: str) -> list[str]:
    """Turn a filename stem into a small, deduped list of tags.

    Stripping happens in stages:
      1. Date prefix (``YYYY-MM-DD-``) is dropped.
      2. The remainder is lowercased and split on ``[-_]``.
      3. Tokens that are pure digits, single chars, or stoplist words
         are filtered out.
      4. A ``vX.Y[.Z]`` token is preserved because version markers are
         useful tags.
    """
    stripped = _strip_date_prefix(stem).lower()
    tokens = re.split(r"[-_]+", stripped)
    seen: list[str] = []
    for tok in tokens:
        if not tok or len(tok) <= 1:
            continue
        if tok.isdigit():
            continue
        if tok in _TAG_STOPLIST:
            continue
        if tok in seen:
            continue
        seen.append(tok)
    return seen[:8]


def _detect_series(stem_upper: str, stem: str) -> tuple[str | None, int | None, int | None]:
    """Return ``(series, episode, part)`` derived from the filename."""
    series: str | None = None
    episode: int | None = None
    part: int | None = None

    cs_match = _CASE_STUDY.match(stem)
    if cs_match:
        series = "wcag-toolkit"
        episode = int(cs_match.group(1))

    ep_match = _EPISODE.match(stem)
    if ep_match:
        episode = int(ep_match.group(1))

    if "WCAG-TOOLKIT" in stem_upper or "WCAG-PUBLIC" in stem_upper or "WCAG-PRO" in stem_upper:
        series = series or "wcag-toolkit"

    if "SDET-BRAIN" in stem_upper:
        series = series or "sdet-brain"

    if "PORTFOLIO" in stem_upper:
        series = series or "portfolio-v2"

    if "JARVIS" in stem_upper:
        series = series or "jarvis-brain"

    part_match = _PART_SUFFIX.search(stem)
    if part_match:
        part = int(part_match.group(1))

    return series, episode, part


def _detect_status(stem_upper: str) -> Status:
    if "PUBLISHED" in stem_upper:
        return "published"
    if "REVIEW" in stem_upper or "FINAL" in stem_upper:
        return "review"
    if "ARCHIVE" in stem_upper or "OLD" in stem_upper:
        return "archive"
    return "draft"


def classify_path(path: Path, body_sample: str = "") -> ClassificationResult:
    """Suggest a :class:`BrandFrontmatter` for ``path``.

    Parameters
    ----------
    path:
        Markdown file. Only the filename stem is inspected; directory
        location is ignored on purpose - classification should not flip
        because a file moved between drafts/ and articles/.
    body_sample:
        The first \\~1k characters of the file body. Used only for
        language detection. Pass an empty string when the body is not
        cheaply available; language defaults to ``"en"`` then.
    """
    stem = path.stem
    stem_upper = stem.upper()
    category, rationale = _classify_category(stem_upper, stem)
    series, episode, part = _detect_series(stem_upper, stem)
    status = _detect_status(stem_upper)

    if stem_upper.endswith("-EN"):
        language: Language = "en"
    elif stem_upper.endswith("-PL"):
        language = "pl"
    else:
        language = _detect_language(body_sample) if body_sample else "en"

    tags = _derive_tags(stem)
    confidence = "high" if category != "other" else "low"

    fm = BrandFrontmatter(
        category=category,
        tags=tags,
        status=status,
        series=series,
        episode=episode,
        part=part,
        language=language,
    )
    return ClassificationResult(
        frontmatter=fm,
        confidence=confidence,
        rationale=rationale,
    )
