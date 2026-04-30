"""Semantic chunker for Markdown bodies.

The chunker walks the body line-by-line, classifying runs of lines into
typed *blocks* (heading / code / table / paragraph), then greedily packs
those blocks into chunks. Code blocks and tables are atomic - the
chunker never splits one mid-fence even if it overflows the target.

Adjacent chunks share a 15% (configurable) overlap to retain context
across boundaries. The overlap is appended only when the previous chunk
is a paragraph-shaped block - we never duplicate code fences.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from sdet_brain.ingestion.models import Chunk

DEFAULT_TARGET_CHARS: Final[int] = 800
DEFAULT_OVERLAP_RATIO: Final[float] = 0.15
HEADING_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(#{1,4})\s+(.*?)\s*$")
CODE_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^```")
TABLE_SEPARATOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\s*\|?[\s|:-]+\|[\s|:-]+\s*$")


class BlockType(StrEnum):
    HEADING = "heading"
    CODE = "code"
    TABLE = "table"
    PARAGRAPH = "paragraph"


@dataclass(frozen=True)
class _Block:
    """An atomic unit consumed by the packer."""

    type: BlockType
    text: str
    heading_level: int = 0  # only meaningful for HEADING
    heading_title: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_atomic(self) -> bool:
        """True when the block must never be split mid-content."""
        return self.type in (BlockType.CODE, BlockType.TABLE)


def _looks_like_table(buffer: list[str], next_line: str) -> bool:
    """Detect a markdown table starting at ``buffer[-1]``.

    Heuristic: a header row of ``|...|`` followed by a separator of
    pipes/dashes/colons is a table; without the separator we treat it
    as a paragraph.
    """
    if not buffer:
        return False
    return buffer[-1].lstrip().startswith("|") and bool(
        TABLE_SEPARATOR_PATTERN.match(next_line)
    )


def _flush_paragraph(buffer: list[str]) -> _Block | None:
    if not buffer:
        return None
    text = "\n".join(buffer).strip("\n")
    buffer.clear()
    if not text.strip():
        return None
    return _Block(type=BlockType.PARAGRAPH, text=text)


def _iter_blocks(body: str) -> Iterator[_Block]:
    """Yield typed blocks for the chunker."""
    lines = body.splitlines()
    paragraph: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Code fence: capture until matching closing fence.
        if CODE_FENCE_PATTERN.match(line):
            flushed = _flush_paragraph(paragraph)
            if flushed is not None:
                yield flushed
            code_lines = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                if CODE_FENCE_PATTERN.match(lines[i]):
                    i += 1
                    break
                i += 1
            yield _Block(type=BlockType.CODE, text="\n".join(code_lines))
            continue

        # Heading.
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            flushed = _flush_paragraph(paragraph)
            if flushed is not None:
                yield flushed
            level = len(heading_match.group(1))
            title = heading_match.group(2)
            yield _Block(
                type=BlockType.HEADING,
                text=line.rstrip(),
                heading_level=level,
                heading_title=title,
            )
            i += 1
            continue

        # Table: pipe row followed (next line) by separator row.
        if line.lstrip().startswith("|") and i + 1 < len(lines) and TABLE_SEPARATOR_PATTERN.match(
            lines[i + 1]
        ):
            flushed = _flush_paragraph(paragraph)
            if flushed is not None:
                yield flushed
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            yield _Block(type=BlockType.TABLE, text="\n".join(table_lines))
            continue

        # Blank line ends the current paragraph.
        if not line.strip():
            flushed = _flush_paragraph(paragraph)
            if flushed is not None:
                yield flushed
            i += 1
            continue

        paragraph.append(line)
        i += 1

    flushed = _flush_paragraph(paragraph)
    if flushed is not None:
        yield flushed


def _heading_path(stack: list[tuple[int, str]]) -> str:
    return " / ".join(title for _, title in stack)


def _push_heading(stack: list[tuple[int, str]], level: int, title: str) -> None:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


def _split_long_paragraph(text: str, target: int) -> list[str]:
    """Hard-split an oversized paragraph on sentence-ish boundaries."""
    if len(text) <= target:
        return [text]
    pieces: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + target, len(text))
        if end < len(text):
            # Try to back off to the previous sentence boundary.
            window = text.rfind(". ", cursor, end)
            if window != -1 and window > cursor + int(target * 0.5):
                end = window + 1  # keep the period
        pieces.append(text[cursor:end].strip())
        cursor = end
    return [piece for piece in pieces if piece]


def _build_overlap(prev_chunk_text: str, ratio: float) -> str:
    if ratio <= 0 or not prev_chunk_text:
        return ""
    overlap_chars = max(1, int(len(prev_chunk_text) * ratio))
    snippet = prev_chunk_text[-overlap_chars:]
    # Start at a token boundary so we don't begin mid-word.
    space = snippet.find(" ")
    if space != -1 and space < len(snippet) - 1:
        snippet = snippet[space + 1 :]
    return snippet.strip()


def chunk_markdown(
    body: str,
    target_size: int = DEFAULT_TARGET_CHARS,
    overlap_pct: float = DEFAULT_OVERLAP_RATIO,
) -> list[Chunk]:
    """Split a Markdown body into a list of chunks.

    Parameters
    ----------
    body:
        The body text (frontmatter must have been stripped beforehand).
    target_size:
        Soft target character budget per chunk. Atomic blocks (code,
        tables) may push a chunk past this ceiling.
    overlap_pct:
        Fraction of the previous chunk's text to prepend to the next
        chunk. Set to ``0`` to disable.
    """
    if not body.strip():
        return []
    if not 0 <= overlap_pct < 1:
        raise ValueError("overlap_pct must be in [0, 1)")
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    blocks = list(_iter_blocks(body))
    if not blocks:
        return []

    # Build chunks as buffers of block text + heading state.
    chunk_texts: list[str] = []
    chunk_headings: list[str] = []
    chunk_has_code: list[bool] = []

    buffer_parts: list[str] = []
    buffer_size = 0
    has_code = False
    heading_stack: list[tuple[int, str]] = []
    chunk_heading_path = ""

    def flush(force: bool = False) -> None:
        nonlocal buffer_size, has_code
        if not buffer_parts:
            return
        if not force and buffer_size == 0:
            return
        chunk_texts.append("\n\n".join(buffer_parts).strip())
        chunk_headings.append(chunk_heading_path)
        chunk_has_code.append(has_code)
        buffer_parts.clear()
        buffer_size = 0
        has_code = False

    for block in blocks:
        if block.type == BlockType.HEADING:
            # A heading is a soft chunk boundary when the buffer is
            # already substantial.
            if buffer_size >= target_size * 0.5:
                flush()
            _push_heading(heading_stack, block.heading_level, block.heading_title)
            chunk_heading_path = _heading_path(heading_stack)
            buffer_parts.append(block.text)
            buffer_size += block.char_count
            continue

        if block.is_atomic:
            if buffer_size and buffer_size + block.char_count > target_size:
                flush()
            buffer_parts.append(block.text)
            buffer_size += block.char_count
            if block.type == BlockType.CODE:
                has_code = True
            # Atomic blocks bigger than the target ride alone.
            if buffer_size >= target_size:
                flush()
            continue

        # Paragraph - may itself need hard-splitting if it dwarfs the
        # target, otherwise it joins the current buffer greedily.
        for piece in _split_long_paragraph(block.text, target_size):
            piece_size = len(piece)
            if buffer_size and buffer_size + piece_size > target_size:
                flush()
            buffer_parts.append(piece)
            buffer_size += piece_size
            if buffer_size >= target_size:
                flush()

    flush(force=True)

    if not chunk_texts:
        return []

    # Apply overlap. We never duplicate code fences, so when the
    # previous chunk ended inside a code fence we skip the overlap.
    chunks: list[Chunk] = []
    total = len(chunk_texts)
    for index, text in enumerate(chunk_texts):
        prefixed = text
        if index > 0 and overlap_pct > 0 and not chunk_has_code[index - 1]:
            overlap_text = _build_overlap(chunk_texts[index - 1], overlap_pct)
            if overlap_text:
                prefixed = f"{overlap_text}\n\n{text}"
        chunks.append(
            Chunk(
                text=prefixed,
                chunk_index=index,
                total_chunks=total,
                heading_path=chunk_headings[index],
                has_code=chunk_has_code[index],
                char_count=len(prefixed),
                token_estimate=len(prefixed) // 4,
            )
        )
    return chunks
