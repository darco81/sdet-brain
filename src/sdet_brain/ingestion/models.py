"""Plain dataclasses describing parsed Markdown content."""

from __future__ import annotations

from dataclasses import dataclass, field

from sdet_brain.ingestion.frontmatter_schema import BrandFrontmatter


@dataclass(frozen=True)
class Chunk:
    """A single chunk produced by the semantic chunker.

    Fields
    ------
    text:
        The chunk body. Includes the overlap prefix from the prior chunk
        (when present) so each chunk is self-contained.
    chunk_index:
        Zero-based ordinal within the parent document.
    total_chunks:
        Total number of chunks produced for the parent document.
    heading_path:
        Slash-separated heading breadcrumb (e.g. ``"Sec / Subsec"``).
        Empty string when the chunk lives before any heading.
    has_code:
        True when the chunk contains at least one fenced code block.
    char_count:
        Length of ``text`` in characters.
    token_estimate:
        Rough ``char_count // 4`` token estimate (no tokenizer
        dependency yet).
    """

    text: str
    chunk_index: int
    total_chunks: int
    heading_path: str
    has_code: bool
    char_count: int
    token_estimate: int


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed Markdown document with its metadata and chunks.

    ``frontmatter`` is the verbatim YAML header for backwards-compat;
    ``brand_frontmatter`` is the validated :class:`BrandFrontmatter`
    when the header matched the schema, else ``None``. Both can be
    populated simultaneously (a header may carry extra keys).
    """

    source_path: str
    content_hash: str
    frontmatter: dict[str, object] = field(default_factory=dict)
    chunks: tuple[Chunk, ...] = ()
    brand_frontmatter: BrandFrontmatter | None = None
