"""CLI entrypoint for the ingestion pipeline.

Examples
--------

    uv run python -m sdet_brain.cli.ingest_cli /path/to/dir
    uv run python -m sdet_brain.cli.ingest_cli /path/to/file.md --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from sdet_brain.config import Settings, get_settings, parse_path_list
from sdet_brain.embeddings.factory import get_embedder
from sdet_brain.ingestion.pipeline import (
    DEFAULT_BATCH_SIZE,
    ingest_path,
)
from sdet_brain.ingestion.source_classifier import (
    SourceConfig,
    default_source_config_from_mapping,
)
from sdet_brain.storage.collections import COLLECTION_NAME, init_collections
from sdet_brain.storage.qdrant_client import QdrantStorage

# Per-source-type root paths come from env vars only
# (PROJECT_KNOWLEDGE_PATHS, DRAFTS_PATHS, ARTICLES_PATHS,
# SPRINT_REPORTS_PATHS, BRIEF_PATHS). Empty env var means "no roots for
# this source_type" - files outside all roots fall through to
# source_type=unknown.
LOCAL_DEFAULT_PATHS: dict[str, list[str]] = {
    "project-knowledge": [],
    "drafts": [],
    "articles": [],
    "sprint-reports": [],
    "brief": [],
}

logger = logging.getLogger("sdet_brain.cli.ingest")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdet-brain-ingest",
        description="Walk a Markdown source and (re-)ingest it into Qdrant.",
    )
    parser.add_argument("path", type=Path, help="File or directory to ingest.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed even when content_hash already matches.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks embedded per API/MLX call.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        type=Path,
        metavar="DIR",
        help="Directory subtree to skip (repeatable).",
    )
    return parser


def _build_source_config(settings: Settings) -> SourceConfig:
    """Return the source-classifier config built from runtime settings.

    Each source_type reads its env var (``DRAFTS_PATHS`` etc.). Empty
    env var falls back to `LOCAL_DEFAULT_PATHS` (an empty list per
    source_type by default).
    """
    mapping: dict[str, list[str]] = {}
    overrides = {
        "project-knowledge": settings.project_knowledge_paths,
        "drafts": settings.drafts_paths,
        "articles": settings.articles_paths,
        "sprint-reports": settings.sprint_reports_paths,
        "brief": settings.brief_paths,
    }
    for source_type, raw in overrides.items():
        configured = parse_path_list(raw)
        mapping[source_type] = configured or LOCAL_DEFAULT_PATHS.get(source_type, [])
    return default_source_config_from_mapping(mapping)


def _run(args: argparse.Namespace, settings: Settings) -> int:
    if not args.path.exists():
        print(f"ERROR: {args.path} does not exist", file=sys.stderr)
        return 1

    selection = get_embedder(settings)
    if selection.fell_back:
        print(
            f"warning: primary provider unavailable, using {selection.provider}",
            file=sys.stderr,
        )

    with QdrantStorage(settings.qdrant_url, api_key=settings.qdrant_api_key) as storage:
        init_collections(storage, vector_size=selection.embedder.vector_size)
        files = list(args.path.rglob("*.md")) if args.path.is_dir() else [args.path]
        progress = tqdm(files, unit="file", desc="ingest")
        stats = ingest_path(
            args.path,
            storage,
            selection.embedder,
            source_config=_build_source_config(settings),
            collection=COLLECTION_NAME,
            batch_size=args.batch_size,
            force_reindex=args.force,
            exclude_dirs=tuple(args.exclude),
            progress=iter(progress),
        )
        progress.close()
    print(stats.summary())
    if stats.errors:
        for src, message in stats.errors:
            print(f"  ERROR {src}: {message}", file=sys.stderr)
    return 0 if not stats.errors else 2


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    return _run(args, settings)


if __name__ == "__main__":
    sys.exit(main())
