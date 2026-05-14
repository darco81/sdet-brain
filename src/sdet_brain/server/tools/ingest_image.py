"""`ingest_image` MCP tool — explicit image / PDF ingestion path.

Sister tool to ``ingest_path`` that exists solely to make Claude pick
the right one (edge case #15 in the v0.6.0 plan). The docstring on the
registered tool lists supported formats explicitly so the agent doesn't
shove a JPEG into the markdown-only ``ingest_path`` slot.
"""

from __future__ import annotations

from pathlib import Path

from sdet_brain.ingestion.image_parser import is_image_path, is_pdf_path
from sdet_brain.ingestion.pipeline import (
    ingest_path as run_ingest,
)
from sdet_brain.ingestion.pipeline import (
    maybe_build_ocr_engine,
)
from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools._helpers import (
    ToolError,
    collection_or_default,
    require_embedder,
    require_storage,
)


def ingest_image(
    state: AppState,
    *,
    path: str,
    force: bool = False,
    collection: str | None = None,
) -> str:
    """Ingest a single image / PDF file (or a directory of them).

    For a single-file target, the path MUST be an image or PDF —
    pointing at a ``.md`` raises ``ToolError`` (use ``ingest_path``
    for markdown). For a directory target, only image and PDF
    descendants are processed; any markdown alongside is left to
    ``ingest_path``. A directory with NO image/PDF files raises
    ``ToolError`` ("nothing to OCR") so callers don't burn the
    OCR-engine boot for a no-op walk.
    """
    target = Path(path)
    if not target.exists():
        raise ToolError(f"path does not exist: {target}")

    if target.is_file() and not (is_image_path(target) or is_pdf_path(target)):
        raise ToolError(
            f"`{target}` is not an image or PDF — use `ingest_path` for "
            "markdown files.",
        )

    # Guard against the wrong-tool-for-the-job mistake before paying for
    # infra checks — user-error diagnostics first, then dependencies.
    ocr_engine = maybe_build_ocr_engine(target, state.settings)
    if ocr_engine is None:
        raise ToolError(
            f"No image or PDF files under `{target}` — nothing to OCR.",
        )

    storage = require_storage(state)
    embedder = require_embedder(state)

    stats = run_ingest(
        target,
        storage,
        embedder,
        collection=collection_or_default(collection),
        force_reindex=force,
        ocr_engine=ocr_engine,
        settings=state.settings,
    )

    lines = [
        f"# Image/PDF ingest summary for `{target}`",
        "",
        f"- Files processed: **{stats.files_processed}**",
        f"- Files skipped (cache): **{stats.files_skipped}**",
        f"- Chunks created: **{stats.chunks_created}**",
        f"- Chunks replaced: **{stats.chunks_replaced}**",
    ]
    if stats.errors:
        lines.append(f"- Errors: **{len(stats.errors)}**")
        for src, message in stats.errors[:5]:
            lines.append(f"  - `{src}`: {message}")
    return "\n".join(lines) + "\n"
