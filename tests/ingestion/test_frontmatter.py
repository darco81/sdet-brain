"""Frontmatter parser unit tests."""

from __future__ import annotations

from sdet_brain.ingestion.frontmatter_parser import parse_frontmatter

YAML_FRONTMATTER = """---
title: Voice sample - opener
status: draft
tags:
  - voice-sample
  - opener
---

# Opening hook

Body content lives below the YAML header.
"""


def test_yaml_frontmatter_is_extracted() -> None:
    metadata, body = parse_frontmatter(YAML_FRONTMATTER)
    assert metadata["title"] == "Voice sample - opener"
    assert metadata["status"] == "draft"
    assert metadata["tags"] == ["voice-sample", "opener"]
    assert body.startswith("# Opening hook")
    assert "---" not in body


def test_no_frontmatter_returns_empty_metadata() -> None:
    text = "# Heading only\n\nNo YAML up top here.\n"
    metadata, body = parse_frontmatter(text)
    assert metadata == {}
    # python-frontmatter strips a single trailing newline; the body
    # otherwise round-trips intact.
    assert body.rstrip("\n") == text.rstrip("\n")
    assert body.startswith("# Heading only")


def test_malformed_frontmatter_falls_back_to_raw_body() -> None:
    malformed = "---\nthis: : is not yaml\n  - dangling\n---\n\nBody.\n"
    metadata, body = parse_frontmatter(malformed)
    assert metadata == {}
    # The raw text is preserved when YAML parsing fails.
    assert body == malformed
