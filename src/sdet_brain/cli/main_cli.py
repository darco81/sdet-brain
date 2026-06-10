"""Unified sdet-brain-cli dispatcher.

Dispatches to subcommands:
  ingest   — Ingest Markdown files into Qdrant (formerly sdet-brain-cli default)
  search   — Search the knowledge base and emit structured output

Examples
--------

    sdet-brain-cli ingest /path/to/dir
    sdet-brain-cli search --query "umysl pieciu" --format json
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]

    if not args or (args[0] in ("-h", "--help") and len(args) == 1):
        _print_help()
        return 0

    subcommand = args[0]

    if subcommand == "ingest":
        from sdet_brain.cli.ingest_cli import main as ingest_main

        return ingest_main(args[1:])

    if subcommand == "search":
        from sdet_brain.cli.search_cli import main as search_main

        return search_main(args[1:])

    # Legacy fallback: if first argument looks like a path (not a flag / subcommand),
    # treat the whole argv as an ingest call for backwards compatibility.
    if args[0].startswith("-") or _looks_like_path(args[0]):
        from sdet_brain.cli.ingest_cli import main as ingest_main

        return ingest_main(args)

    print(f"Unknown subcommand: {args[0]!r}", file=sys.stderr)
    _print_help()
    return 1


def _looks_like_path(token: str) -> bool:
    """Heuristic: does the token look like a filesystem path?"""
    return token.startswith("/") or token.startswith("./") or token.startswith("../")


def _print_help() -> None:
    print(
        """\
usage: sdet-brain-cli <subcommand> [options]

Subcommands:
  ingest   Ingest Markdown files into the Qdrant knowledge base.
           (all flags forwarded to sdet-brain-ingest)
  search   Search the knowledge base.
           --query TEXT         Search query (required)
           --source-type TYPE   Filter by source_type (e.g. councils)
           --limit N            Max results (default: 5)
           --min-score FLOAT    Minimum score threshold (default: 0.0)
           --format {json,text} Output format (default: text)

Examples:
  sdet-brain-cli ingest ~/docs/
  sdet-brain-cli search --query "umysl pieciu" --format json
  sdet-brain-cli search --query "council decision" --source-type councils --limit 3
"""
    )


if __name__ == "__main__":
    sys.exit(main())
