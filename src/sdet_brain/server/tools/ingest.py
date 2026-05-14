"""`ingest_path` MCP tool implementation."""

from __future__ import annotations

from pathlib import Path

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


def ingest_path(
    state: AppState,
    *,
    path: str,
    force: bool = False,
    collection: str | None = None,
) -> str:
    """Re-ingest a file or directory and return a summary string.

    Supports markdown (``.md``), images (``.jpg``/``.png``/``.heic``/...)
    and PDFs (``.pdf``). Image and PDF files trigger OCR via the
    configured provider chain (MLX-VLM → Ollama DeepSeek → Ollama Qwen
    on Mac; Ollama-only on Win).
    """
    target = Path(path)
    if not target.exists():
        raise ToolError(f"path does not exist: {target}")
    storage = require_storage(state)
    embedder = require_embedder(state)
    ocr_engine = maybe_build_ocr_engine(target, state.settings)
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
        f"# Ingest summary for `{target}`",
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
