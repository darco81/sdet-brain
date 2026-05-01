"""Tests for the structured frontmatter schema (T2-01)."""

from __future__ import annotations

from datetime import date

from sdet_brain.ingestion.frontmatter_schema import (
    BrandFrontmatter,
    parse_brand_frontmatter,
    to_payload_fields,
)


def test_parse_minimal_valid_header() -> None:
    raw: dict[str, object] = {"category": "draft"}
    fm = parse_brand_frontmatter(raw)
    assert fm is not None
    assert fm.category == "draft"
    assert fm.status == "draft"  # default
    assert fm.language == "en"  # default
    assert fm.tags == []


def test_parse_full_header() -> None:
    raw: dict[str, object] = {
        "category": "case-study",
        "tags": ["wcag", "toolkit"],
        "status": "published",
        "series": "wcag-toolkit",
        "episode": 1,
        "part": 2,
        "language": "en",
        "created_at": "2026-04-21",
        "updated_at": "2026-04-30",
    }
    fm = parse_brand_frontmatter(raw)
    assert fm is not None
    assert fm.category == "case-study"
    assert fm.tags == ["wcag", "toolkit"]
    assert fm.status == "published"
    assert fm.series == "wcag-toolkit"
    assert fm.episode == 1
    assert fm.part == 2
    assert fm.created_at == date(2026, 4, 21)
    assert fm.updated_at == date(2026, 4, 30)


def test_parse_invalid_category_returns_none() -> None:
    raw: dict[str, object] = {"category": "totally-bogus"}
    assert parse_brand_frontmatter(raw) is None


def test_parse_missing_required_returns_none() -> None:
    raw: dict[str, object] = {"tags": ["foo"]}
    assert parse_brand_frontmatter(raw) is None


def test_parse_empty_dict_returns_none() -> None:
    assert parse_brand_frontmatter({}) is None


def test_tags_string_coerced_to_list() -> None:
    raw: dict[str, object] = {"category": "draft", "tags": "single-tag"}
    fm = parse_brand_frontmatter(raw)
    assert fm is not None
    assert fm.tags == ["single-tag"]


def test_extra_fields_ignored() -> None:
    raw: dict[str, object] = {
        "category": "draft",
        "title": "ignored, has its own home",
        "author": "ignored too",
    }
    fm = parse_brand_frontmatter(raw)
    assert fm is not None  # extra fields don't cause validation failure


def test_to_payload_fields_compact() -> None:
    fm = BrandFrontmatter(category="draft")
    out = to_payload_fields(fm)
    # Defaults are still emitted because they're useful filter keys.
    assert out == {"category": "draft", "status": "draft", "language": "en"}


def test_to_payload_fields_full() -> None:
    fm = BrandFrontmatter(
        category="case-study",
        tags=["a", "b"],
        status="published",
        series="wcag-toolkit",
        episode=2,
        part=1,
        language="pl",
        created_at=date(2026, 5, 1),
    )
    out = to_payload_fields(fm)
    assert out["category"] == "case-study"
    assert out["status"] == "published"
    assert out["language"] == "pl"
    assert out["tags"] == ["a", "b"]
    assert out["series"] == "wcag-toolkit"
    assert out["episode"] == 2
    assert out["part"] == 1
    assert out["fm_created_at"] == "2026-05-01"
    assert "fm_updated_at" not in out  # absent => not emitted
