"""CLI for the embedding service.

Examples
--------

    uv run python -m sdet_brain.cli.embed_cli health
    uv run python -m sdet_brain.cli.embed_cli encode "Hello SDET Brain"
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from sdet_brain.config import Settings, get_settings
from sdet_brain.embeddings.factory import EmbedderSelection, get_embedder
from sdet_brain.embeddings.protocol import EmbeddingError

CommandHandler = Callable[[argparse.Namespace, Settings], int]

logger = logging.getLogger("sdet_brain.cli.embed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdet-brain-embed",
        description="Inspect and exercise the embedding service.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    encode = sub.add_parser("encode", help="Encode a string and print its vector summary.")
    encode.add_argument("text", help="Text to embed.")
    encode.add_argument(
        "--full",
        action="store_true",
        help="Print the full vector instead of a head/tail preview.",
    )

    sub.add_parser("health", help="Run the configured provider's health check.")
    return parser


def _select_embedder(settings: Settings) -> EmbedderSelection:
    return get_embedder(settings)


def _cmd_encode(args: argparse.Namespace, settings: Settings) -> int:
    selection = _select_embedder(settings)
    try:
        vectors = selection.embedder.embed([args.text])
    except EmbeddingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not vectors:
        print("ERROR: provider returned no vectors", file=sys.stderr)
        return 2
    vector = vectors[0]
    print(f"provider:    {selection.provider}{' (fallback)' if selection.fell_back else ''}")
    print(f"model:       {selection.embedder.model_name}")
    print(f"vector_size: {len(vector)}")
    if args.full:
        print(f"vector:      {vector}")
    else:
        head = ", ".join(f"{v:.4f}" for v in vector[:3])
        tail = ", ".join(f"{v:.4f}" for v in vector[-3:])
        print(f"vector head: [{head}, ...]")
        print(f"vector tail: [..., {tail}]")
    return 0


def _cmd_health(_: argparse.Namespace, settings: Settings) -> int:
    try:
        selection = _select_embedder(settings)
    except EmbeddingError as exc:
        print(f"unhealthy: {exc}", file=sys.stderr)
        return 1
    print(f"primary:     {settings.embedding_provider}")
    print(f"active:      {selection.provider}")
    print(f"fell_back:   {selection.fell_back}")
    print(f"attempted:   {', '.join(selection.attempted)}")
    print(f"model:       {selection.embedder.model_name}")
    print(f"vector_size: {selection.embedder.vector_size}")
    return 0


_HANDLERS: dict[str, CommandHandler] = {
    "encode": _cmd_encode,
    "health": _cmd_health,
}


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    handler = _HANDLERS[args.command]
    return handler(args, settings)


if __name__ == "__main__":
    sys.exit(main())
