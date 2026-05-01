"""Tests for path-based frontmatter classification (T2-01)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sdet_brain.ingestion.frontmatter_classifier import classify_path


@pytest.mark.parametrize(
    ("filename", "expected_category"),
    [
        ("case-study-01-OUTLINE.md", "outline"),
        ("case-study-01-RAW-NOTES.md", "raw-notes"),
        ("case-study-01-SMACZKI.md", "smaczki"),
        ("case-study-01-wcag-DRAFT.md", "case-study"),
        ("episode-05-multipage-NOTES.md", "raw-notes"),
        ("EXECUTION-PLAN.md", "execution-plan"),
        ("SDET-BRAND-STRATEGY.md", "brand-strategy"),
        ("WEDNESDAY-SPRINT-REPORT.md", "sprint-report"),
        ("THURSDAY-DEPLOY-SPRINT-PROMPT.md", "prompt"),
        ("CC-PROMPT-WCAG-PRO-MAINTENANCE.md", "prompt"),
        ("WRITE-UP-PART-1-PROMPT.md", "prompt"),
        ("2026-05-01-keyboard-trap-runtime-verdict.md", "decision"),
        ("2026-04-30-force-push-policy-clarification.md", "decision"),
        ("2026-04-30-linear-workspace-setup-decisions.md", "decision"),
        ("2026-04-30-wcag-pro-state.md", "other"),
        ("v0.3.0-build-report.md", "sprint-report"),
        ("portfolio-v2-fix-summary.md", "other"),
    ],
)
def test_category_classification(filename: str, expected_category: str) -> None:
    result = classify_path(Path(filename))
    assert result.frontmatter.category == expected_category


def test_case_study_extracts_episode() -> None:
    result = classify_path(Path("case-study-01-wcag-DRAFT.md"))
    assert result.frontmatter.series == "wcag-toolkit"
    assert result.frontmatter.episode == 1


def test_episode_filename_extracts_episode() -> None:
    result = classify_path(Path("episode-05-multipage-NOTES.md"))
    assert result.frontmatter.episode == 5


def test_part_suffix_extracted() -> None:
    result = classify_path(Path("WRITE-UP-PART-2-PROMPT.md"))
    assert result.frontmatter.part == 2


def test_polish_body_detected() -> None:
    body = "Witaj, to jest test po polsku z dużą liczbą polskich znaków: ąćęłńóśźż."
    result = classify_path(Path("some-draft.md"), body_sample=body)
    assert result.frontmatter.language == "pl"


def test_english_body_detected() -> None:
    body = "Hello world, this is a perfectly normal English sentence with no diacritics."
    result = classify_path(Path("some-draft.md"), body_sample=body)
    assert result.frontmatter.language == "en"


def test_filename_suffix_overrides_body_language() -> None:
    body = "Witaj świecie, ąćęłńóśźż wszystkie znaki polskie."
    result = classify_path(
        Path("2026-04-29-dm-template-fullstack-to-qa-EN.md"), body_sample=body
    )
    assert result.frontmatter.language == "en"


def test_tags_strip_dates_and_stoplist() -> None:
    result = classify_path(Path("2026-04-30-portfolio-v2-state.md"))
    tags = result.frontmatter.tags
    # Date prefix is dropped, stoplist words removed, but "portfolio" survives.
    assert "portfolio" in tags
    assert "2026" not in tags
    assert "v0" not in tags  # stoplist


def test_tags_capped_at_eight() -> None:
    result = classify_path(
        Path(
            "2026-04-30-very-long-filename-with-many-distinct-tokens-here-please.md"
        )
    )
    assert len(result.frontmatter.tags) <= 8


def test_default_status_is_draft() -> None:
    result = classify_path(Path("some-random-file.md"))
    assert result.frontmatter.status == "draft"


def test_unrecognised_filename_falls_back_to_other() -> None:
    result = classify_path(Path("totally-random-thing.md"))
    assert result.frontmatter.category == "other"
    assert result.confidence == "low"
