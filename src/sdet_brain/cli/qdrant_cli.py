"""CLI for inspecting / initialising the Qdrant collection.

Usage examples
--------------

    uv run python -m sdet_brain.cli.qdrant_cli init
    uv run python -m sdet_brain.cli.qdrant_cli status
    uv run python -m sdet_brain.cli.qdrant_cli ping
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from sdet_brain.config import Settings, get_settings
from sdet_brain.storage.collections import COLLECTION_NAME, init_collections
from sdet_brain.storage.qdrant_client import QdrantStorage

CommandHandler = Callable[[Settings], int]

logger = logging.getLogger("sdet_brain.cli.qdrant")


def _vector_size_for(settings: Settings) -> int:
    if settings.embedding_provider == "mlx":
        return settings.mlx_vector_size
    return settings.gemini_vector_size


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdet-brain-qdrant",
        description="Initialise or inspect the SDET Brain Qdrant collection.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create the collection if it does not exist (idempotent).")
    sub.add_parser("status", help="Print collection metadata and point count.")
    sub.add_parser("ping", help="Health-check the Qdrant endpoint.")
    return parser


def _cmd_init(settings: Settings) -> int:
    vector_size = _vector_size_for(settings)
    with QdrantStorage(settings.qdrant_url, api_key=settings.qdrant_api_key) as storage:
        created = init_collections(storage, vector_size=vector_size)
    if created:
        print(f"Created collection {COLLECTION_NAME} (vector_size={vector_size}).")
    else:
        print(f"Collection {COLLECTION_NAME} already exists - no-op.")
    return 0


def _cmd_status(settings: Settings) -> int:
    with QdrantStorage(settings.qdrant_url, api_key=settings.qdrant_api_key) as storage:
        if not storage.collection_exists(COLLECTION_NAME):
            print(f"Collection {COLLECTION_NAME} does not exist. Run `init` first.")
            return 1
        snapshot = storage.status(COLLECTION_NAME)
    print(f"name:         {snapshot.name}")
    print(f"vector_size:  {snapshot.vector_size}")
    print(f"distance:     {snapshot.distance}")
    print(f"points_count: {snapshot.points_count}")
    return 0


def _cmd_ping(settings: Settings) -> int:
    with QdrantStorage(settings.qdrant_url, api_key=settings.qdrant_api_key) as storage:
        collections = storage.list_collections()
    print(f"OK - {settings.qdrant_url} reachable, collections={collections}")
    return 0


_HANDLERS: dict[str, CommandHandler] = {
    "init": _cmd_init,
    "status": _cmd_status,
    "ping": _cmd_ping,
}


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
    return handler(get_settings())


if __name__ == "__main__":
    sys.exit(main())
