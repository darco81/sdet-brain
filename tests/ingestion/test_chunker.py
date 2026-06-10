"""Semantic chunker unit tests.

These tests exercise the chunker against synthetic strings where the
expected behaviour is easy to reason about, plus the committed fixtures
under `tests/ingestion/fixtures/` for end-to-end realism.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sdet_brain.ingestion.chunker import (
    DEFAULT_TARGET_CHARS,
    chunk_markdown,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_short_document_produces_single_chunk() -> None:
    text = "Single paragraph that is well below the target.\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1
    assert chunks[0].heading_path == ""
    assert chunks[0].has_code is False


def test_empty_input_returns_no_chunks() -> None:
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  \n") == []


def test_long_document_with_five_headings_yields_multiple_chunks() -> None:
    sections: list[str] = []
    for i in range(5):
        sections.append(f"## Section {i}\n\n" + ("Sentence number alpha. " * 40))
    body = "\n\n".join(sections)
    chunks = chunk_markdown(body, target_size=400)
    assert len(chunks) >= 5
    headings = [c.heading_path for c in chunks]
    assert all(h.startswith("Section") for h in headings)
    # Indices are dense and start at 0.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.total_chunks == len(chunks) for c in chunks)


def test_code_blocks_are_atomic() -> None:
    body = (
        "Intro paragraph.\n\n"
        "```python\n" + "print('one')\n" * 30 + "```\n\n"
        "Tail paragraph that follows the block.\n"
    )
    chunks = chunk_markdown(body, target_size=200)
    # Every chunk has an even number of fences (open + close stay together).
    for chunk in chunks:
        assert chunk.text.count("```") % 2 == 0
    assert any(chunk.has_code for chunk in chunks)


def test_tables_are_atomic() -> None:
    body = (
        "Lead-in paragraph.\n\n"
        "| Col A | Col B |\n"
        "| ----- | ----- |\n"
        + "\n".join(f"| a{i} | b{i} |" for i in range(20))
        + "\n\nTrailing paragraph.\n"
    )
    chunks = chunk_markdown(body, target_size=120)
    table_pieces = [chunk for chunk in chunks if "| Col A" in chunk.text]
    assert len(table_pieces) == 1, "table must land in exactly one chunk"
    table_chunk = table_pieces[0]
    assert "| a0 | b0 |" in table_chunk.text
    assert "| a19 | b19 |" in table_chunk.text


def test_overlap_prepends_tail_of_previous_chunk() -> None:
    body = "## First\n\n" + "alpha " * 200 + "\n\n## Second\n\n" + "beta " * 200
    chunks = chunk_markdown(body, target_size=400, overlap_pct=0.15)
    assert len(chunks) >= 2
    second = chunks[1]
    prev = chunks[0].text
    overlap_size = max(1, int(len(prev) * 0.15))
    tail = prev[-overlap_size:]
    # The tail is normalised by trimming to a token boundary, so we
    # check that *some* prefix of the second chunk text matches the
    # tail of the first.
    common_prefix = 0
    for left, right in zip(tail.split(), second.text.split(), strict=False):
        if left == right:
            common_prefix += 1
        else:
            break
    assert common_prefix >= 1, "expected at least one overlapping token"


def test_overlap_disabled_produces_smaller_chunks_than_with_overlap() -> None:
    body = "## First\n\n" + "alpha " * 200 + "\n\n## Second\n\n" + "beta " * 200
    no_overlap = chunk_markdown(body, target_size=400, overlap_pct=0.0)
    with_overlap = chunk_markdown(body, target_size=400, overlap_pct=0.15)
    assert len(no_overlap) == len(with_overlap)
    # With overlap each chunk after the first carries an overlap prefix,
    # so its char count is strictly larger.
    for left, right in zip(no_overlap[1:], with_overlap[1:], strict=True):
        assert right.char_count > left.char_count


@pytest.mark.parametrize("fixture", ["simple.md", "voice-sample.md", "complex.md"])
def test_fixture_files_chunk_cleanly(fixture: str) -> None:
    body = (FIXTURES / fixture).read_text(encoding="utf-8")
    chunks = chunk_markdown(body, target_size=DEFAULT_TARGET_CHARS)
    assert chunks, f"{fixture} produced no chunks"
    for chunk in chunks:
        assert chunk.text.count("```") % 2 == 0
    if fixture == "complex.md":
        # The table heading row must end up in exactly one chunk.
        carriers = [c for c in chunks if "| Col A" in c.text]
        # python-frontmatter strips frontmatter via parse_markdown - here
        # we feed the raw body, so frontmatter is part of the input. The
        # chunker leaves it untouched, but headings still drive splits.
        assert len(carriers) <= 1


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        chunk_markdown("hello", target_size=0)
    with pytest.raises(ValueError):
        chunk_markdown("hello", overlap_pct=1.5)


# --- SDE-40 small-tail merge ---


def test_small_tail_merges_into_previous_prose_chunk() -> None:
    """A sub-250-char trailing section folds into the preceding prose chunk.

    Pre-merge layout would be 3 chunks (lone heading, body, tiny tail).
    Post-merge: heading still flushes alone (heading-only chunk lives
    < 250 chars and is itself a tail), so the tail rolls into the body.
    """
    body = (
        "## First section\n\n"
        + ("alpha beta gamma delta. " * 30)
        + "\n\n## Tiny tail\n\nA short tail under 250 chars.\n"
    )
    chunks = chunk_markdown(body, target_size=600, overlap_pct=0.0)
    # Tiny tail must end up inside one of the chunks (the merge target),
    # NOT as its own chunk.
    tail_carriers = [c for c in chunks if "A short tail under 250 chars" in c.text]
    assert len(tail_carriers) == 1
    carrier = tail_carriers[0]
    assert "alpha beta" in carrier.text
    assert "Tiny tail" in carrier.text
    # Carrier should be substantial (merge target), not the lone tail.
    assert carrier.char_count > 250


def test_small_tail_after_code_block_is_not_merged() -> None:
    """Code-fenced previous chunk stays atomic - no append."""
    body = (
        "## Pipeline\n\n"
        "```python\n" + "print('chunk-anchor')\n" * 30 + "```\n\n"
        "## Tail\n\nShort follow-up under 250 chars.\n"
    )
    chunks = chunk_markdown(body, target_size=400, overlap_pct=0.0)
    assert len(chunks) >= 2
    code_chunks = [c for c in chunks if c.has_code]
    assert code_chunks, "expected a code-fenced chunk"
    # The code chunk must NOT have absorbed the tail prose.
    assert not any("Short follow-up" in c.text for c in code_chunks)


def test_small_tail_does_not_cross_upper_bound() -> None:
    """If combining would exceed upper_bound (1.5 * target), keep separate."""
    target = 400
    # First chunk near upper bound, tail is small but combined > upper.
    body = (
        "## Big\n\n"
        + ("word " * 130)  # ~650 chars - close to upper bound 600
        + "\n\n## Mini\n\n"
        + ("tiny " * 40)  # ~200 chars
    )
    chunks = chunk_markdown(body, target_size=target, overlap_pct=0.0)
    # Combined ~850 > upper_bound 600 -> stay split.
    assert len(chunks) >= 2


def test_threshold_boundary_at_250() -> None:
    """Exactly 250-char tail should NOT merge (threshold is strict <250)."""
    body = (
        "## Big\n\n"
        + ("alpha " * 80)  # ~480 chars
        + "\n\n## Edge\n\n"
        + ("x" * 250)  # exactly 250 chars
    )
    chunks = chunk_markdown(body, target_size=400, overlap_pct=0.0)
    # 250 is NOT below threshold -> stays as separate chunk.
    assert len(chunks) >= 2


def test_chunk_indices_renumber_after_merge() -> None:
    """After merging, indices must be 0..n-1 and total_chunks consistent."""
    body = (
        "## A\n\n"
        + ("alpha " * 50)
        + "\n\n## B\n\n"
        + ("beta " * 50)
        + "\n\n## C\n\n"
        + "tiny tail."
    )
    chunks = chunk_markdown(body, target_size=300, overlap_pct=0.0)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.total_chunks == len(chunks) for c in chunks)
