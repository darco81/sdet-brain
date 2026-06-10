"""`sdet-brain-watcher` CLI - daemon process that auto-reindexes Markdown sources.

Reads the watch paths from `WATCH_PATHS` (comma-separated absolute
paths) and runs until SIGTERM / SIGINT. The signal handler triggers a
graceful shutdown that drains any pending debounced events.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from sdet_brain.cli.ingest_cli import LOCAL_DEFAULT_PATHS
from sdet_brain.config import Settings, get_settings, parse_path_list
from sdet_brain.embeddings.factory import get_embedder
from sdet_brain.ingestion.source_classifier import SourceConfig, default_source_config_from_mapping
from sdet_brain.ingestion.watcher import BrainWatcher
from sdet_brain.storage.collections import COLLECTION_NAME, init_collections
from sdet_brain.storage.qdrant_client import QdrantStorage

logger = logging.getLogger("sdet_brain.cli.watcher")


def _split_paths(value: str) -> list[Path]:
    return [Path(part.strip()).expanduser() for part in value.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdet-brain-watcher",
        description="Auto-reindex Markdown sources on disk into Qdrant.",
    )
    parser.add_argument(
        "--paths",
        type=str,
        default=None,
        help="Comma-separated paths to watch. Falls back to WATCH_PATHS env.",
    )
    parser.add_argument(
        "--debounce-ms",
        type=int,
        default=None,
        help="Debounce window in milliseconds (overrides WATCHER_DEBOUNCE_MS).",
    )
    return parser


def _resolve_paths(args: argparse.Namespace, settings: Settings) -> list[Path]:
    raw = args.paths if args.paths is not None else settings.watch_paths
    paths = _split_paths(raw)
    if not paths:
        raise SystemExit("no watch paths configured (pass --paths or set WATCH_PATHS in .env)")
    return paths


def _default_source_config(settings: Settings) -> SourceConfig:
    """Build the source classifier config, env-var overrides win.

    Mirrors the logic in ``cli/ingest_cli._build_source_config`` so a
    single ``.env`` controls both the watcher and the one-shot ingest
    CLI. Empty env vars fall back to ``LOCAL_DEFAULT_PATHS``.
    """
    overrides = {
        "project-knowledge": settings.project_knowledge_paths,
        "drafts": settings.drafts_paths,
        "articles": settings.articles_paths,
        "sprint-reports": settings.sprint_reports_paths,
        "brief": settings.brief_paths,
    }
    mapping: dict[str, list[str]] = {}
    for source_type, raw in overrides.items():
        configured = parse_path_list(raw)
        mapping[source_type] = configured or LOCAL_DEFAULT_PATHS.get(source_type, [])
    return default_source_config_from_mapping(mapping)


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    watch_paths = _resolve_paths(args, settings)
    debounce_ms = args.debounce_ms or settings.watcher_debounce_ms

    selection = get_embedder(settings)
    if selection.fell_back:
        logger.warning("Primary embedder unavailable, watcher running on %s", selection.provider)

    with QdrantStorage(settings.qdrant_url, api_key=settings.qdrant_api_key) as storage:
        init_collections(storage, vector_size=selection.embedder.vector_size)
        watcher = BrainWatcher(
            watch_paths,
            storage,
            selection.embedder,
            source_config=_default_source_config(settings),
            collection=COLLECTION_NAME,
            debounce_ms=debounce_ms,
        )
        stop_event = threading.Event()

        def _handle_signal(signum: int, _: object) -> None:
            logger.info("Caught signal %s - shutting watcher down", signum)
            stop_event.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        watcher.start()
        try:
            while not stop_event.is_set():
                stop_event.wait(0.5)
        finally:
            watcher.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
