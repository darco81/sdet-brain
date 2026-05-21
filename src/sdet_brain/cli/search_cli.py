"""CLI entrypoint for the search subcommand.

Runs a hybrid (dense + BM25 RRF) search against the local Qdrant collection
and emits JSON suitable for consumption by external tools like umysl-pieciu.

Examples
--------

    sdet-brain-cli search --query "umysl pieciu publish" --format json
    sdet-brain-cli search --query "WCAG accessibility" --source-type articles --limit 5
    sdet-brain-cli search --query "council decision" --source-type councils --min-score 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchValue

from sdet_brain.config import get_settings
from sdet_brain.embeddings.factory import get_embedder
from sdet_brain.embeddings.sparse_embedder import FastembedBM25, get_sparse_embedder
from sdet_brain.storage.collections import COLLECTION_NAME
from sdet_brain.storage.qdrant_client import QdrantStorage

logger = logging.getLogger("sdet_brain.cli.search")

# Single lazy BM25 instance per process.
_SPARSE: FastembedBM25 | None = None


def _sparse() -> FastembedBM25:
    global _SPARSE
    if _SPARSE is None:
        _SPARSE = get_sparse_embedder()
    return _SPARSE


def _source_type_filter(value: str | None) -> Filter | None:
    if not value:
        return None
    return Filter(
        must=[FieldCondition(key="source_type", match=MatchValue(value=value))]
    )


def _extract_slug(source_path: str) -> str:
    """Derive the council slug from a path like .../councils/<slug>/verdict.md."""
    # Match councils/<slug>/... pattern
    m = re.search(r"councils[/\\]([^/\\]+)[/\\]", source_path)
    if m:
        return m.group(1)
    # Fallback: stem of the file
    return Path(source_path).stem or "unknown"


def _extract_topic(text: str, source_path: str) -> str:
    """Extract topic from the first H1 heading in the text, or fall back."""
    # Look for # Topic at line start
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    # Fall back to slug
    return _extract_slug(source_path).replace("-", " ").title()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdet-brain-cli search",
        description="Search the sdet-brain knowledge base and emit structured output.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Search query string.",
    )
    parser.add_argument(
        "--source-type",
        default=None,
        help="Filter by source_type (e.g. councils, articles, drafts).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 5).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum similarity score threshold (default: 0.0 = no filter).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format: json (machine-readable) or text (human-readable).",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Use dense-only search instead of hybrid BM25+dense (slower but simpler).",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    selection = get_embedder(settings)
    if selection.fell_back:
        print(
            f"warning: primary provider unavailable, using {selection.provider}",
            file=sys.stderr,
        )

    embedder = selection.embedder
    vectors = embedder.embed([args.query])
    if not vectors:
        _emit_empty(args)
        return 0

    query_filter = _source_type_filter(args.source_type)
    hybrid = not args.no_hybrid

    with QdrantStorage(settings.qdrant_url, api_key=settings.qdrant_api_key) as storage:
        if hybrid:
            sparse_vec = _sparse().embed([args.query])[0]
            points = storage.hybrid_search(
                collection=COLLECTION_NAME,
                dense_vector=vectors[0],
                sparse_indices=sparse_vec.indices,
                sparse_values=sparse_vec.values,
                limit=args.limit,
                query_filter=query_filter,
            )
        else:
            points = storage.search(
                collection=COLLECTION_NAME,
                query_vector=vectors[0],
                limit=args.limit,
                query_filter=query_filter,
                score_threshold=args.min_score if args.min_score > 0 else None,
            )

    # Apply min_score filter (hybrid search doesn't support score_threshold directly)
    if args.min_score > 0:
        points = [p for p in points if float(p.score) >= args.min_score]

    if args.format == "json":
        _emit_json(args, points)
    else:
        _emit_text(args, points)

    return 0


def _emit_empty(args: argparse.Namespace) -> None:
    if args.format == "json":
        print(json.dumps({"results": []}, ensure_ascii=False))
    else:
        print(f"No results for: {args.query}")


def _emit_json(args: argparse.Namespace, points: list) -> None:  # type: ignore[type-arg]
    results = []
    for point in points:
        payload = dict(point.payload or {})
        source_path = str(payload.get("source_path", ""))
        text = str(payload.get("text", ""))
        score = float(point.score)
        results.append(
            {
                "slug": _extract_slug(source_path),
                "topic": _extract_topic(text, source_path),
                "snippet": text[:500],
                "score": round(score, 6),
                "path": source_path,
            }
        )
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


def _emit_text(args: argparse.Namespace, points: list) -> None:  # type: ignore[type-arg]
    if not points:
        print(f"No results for: {args.query!r}")
        return
    filter_label = f" [source_type={args.source_type}]" if args.source_type else ""
    print(f"Search results for: {args.query!r}{filter_label}\n")
    for idx, point in enumerate(points, start=1):
        payload = dict(point.payload or {})
        source_path = str(payload.get("source_path", "(unknown)"))
        text = str(payload.get("text", ""))
        score = float(point.score)
        slug = _extract_slug(source_path)
        topic = _extract_topic(text, source_path)
        print(f"{idx}. [{slug}] {topic} (score: {score:.4f})")
        print(f"   path: {source_path}")
        snippet = text[:200].replace("\n", " ")
        print(f"   {snippet}")
        print()


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
