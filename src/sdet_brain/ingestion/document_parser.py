"""End-to-end Markdown ingestion: file -> ParsedDocument."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from sdet_brain.ingestion.chunker import (
    DEFAULT_OVERLAP_RATIO,
    DEFAULT_TARGET_CHARS,
    chunk_markdown,
)
from sdet_brain.ingestion.frontmatter_parser import parse_frontmatter
from sdet_brain.ingestion.models import ParsedDocument

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """Return a deterministic SHA-256 hex digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_markdown(
    path: Path,
    target_size: int = DEFAULT_TARGET_CHARS,
    overlap_pct: float = DEFAULT_OVERLAP_RATIO,
) -> ParsedDocument:
    """Read ``path`` and return a fully-chunked ``ParsedDocument``.

    The hash is computed over the raw file contents so a re-ingest can
    short-circuit when nothing changed on disk.
    """
    raw = path.read_text(encoding="utf-8")
    content_hash = compute_content_hash(raw)
    metadata, body = parse_frontmatter(raw)
    chunks = chunk_markdown(body, target_size=target_size, overlap_pct=overlap_pct)
    logger.debug("Parsed %s: %d chunks (hash=%s)", path, len(chunks), content_hash[:12])
    return ParsedDocument(
        source_path=str(path),
        content_hash=content_hash,
        frontmatter=metadata,
        chunks=tuple(chunks),
    )
