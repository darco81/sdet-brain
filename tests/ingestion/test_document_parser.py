"""End-to-end parse_markdown tests using committed fixtures."""

from __future__ import annotations

from pathlib import Path

from sdet_brain.ingestion.document_parser import compute_content_hash, parse_markdown

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_markdown_returns_full_document() -> None:
    document = parse_markdown(FIXTURES / "voice-sample.md")
    assert document.source_path.endswith("voice-sample.md")
    assert document.frontmatter["status"] == "draft"
    assert document.frontmatter["tags"] == ["voice-sample", "opener"]
    assert document.chunks
    # The fixture is short enough that both H1 and H2 sections collapse
    # into the same first chunk; the heading path therefore starts with
    # the H1 title.
    assert document.chunks[0].heading_path.startswith("Opening hook")


def test_parse_markdown_simple_yields_single_chunk_no_frontmatter() -> None:
    document = parse_markdown(FIXTURES / "simple.md")
    assert document.frontmatter == {}
    assert len(document.chunks) == 1
    assert document.chunks[0].heading_path == ""


def test_content_hash_is_deterministic_for_same_text() -> None:
    text = "## Heading\n\nBody paragraph.\n"
    assert compute_content_hash(text) == compute_content_hash(text)
    assert compute_content_hash(text) != compute_content_hash(text + "tail")


def test_chunk_indices_are_sequential() -> None:
    document = parse_markdown(FIXTURES / "complex.md")
    indices = [c.chunk_index for c in document.chunks]
    assert indices == list(range(len(document.chunks)))
    assert all(c.total_chunks == len(document.chunks) for c in document.chunks)
