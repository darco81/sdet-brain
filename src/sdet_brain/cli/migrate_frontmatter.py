"""Apply structured frontmatter to brand corpus files in-place (T2-01).

Walks one or more directories, classifies every ``.md`` file using
:func:`sdet_brain.ingestion.frontmatter_classifier.classify_path`, and
prepends a YAML header for any file that lacks one. Files that already
declare a valid :class:`BrandFrontmatter` are skipped.

Usage
-----

    uv run python -m sdet_brain.cli.migrate_frontmatter PATH [PATH...] \\
        --apply  # without --apply we only print suggestions

Always run with ``--dry-run`` first (the default) and review the diff
before re-running with ``--apply``. The tool writes a migration log to
``migrations/<timestamp>-frontmatter-migration.md`` so the changes are
traceable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import yaml

from sdet_brain.ingestion.frontmatter_classifier import (
    ClassificationResult,
    classify_path,
)
from sdet_brain.ingestion.frontmatter_parser import parse_frontmatter
from sdet_brain.ingestion.frontmatter_schema import (
    BrandFrontmatter,
    parse_brand_frontmatter,
)

# Map free-form status values that already live in the corpus onto the
# strict :class:`Status` literal. Values not covered fall back to the
# classifier's default ("draft") rather than failing the migration.
_STATUS_REMAP: dict[str, str] = {
    "in-progress": "draft",
    "wip": "draft",
    "completed": "published",
    "done": "published",
    "decided": "published",
    "decision-log": "published",
    "log": "published",
    "final": "published",
    "archived": "archive",
    "old": "archive",
}

logger = logging.getLogger("sdet_brain.cli.migrate_frontmatter")

# Body sample size for language detection; matches the classifier's
# internal cutoff so the tool and the runtime see the same signal.
_BODY_SAMPLE_CHARS = 1000


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdet-brain-migrate-frontmatter",
        description=(
            "Walk Markdown files under PATH and prepend structured "
            "frontmatter where missing. Existing valid headers are kept."
        ),
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="One or more directories (or files) to walk.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Without this flag we only report.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("migrations"),
        help="Directory the migration log is written to (created if missing).",
    )
    return parser


def _iter_md(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".md" else []
    return sorted(
        p for p in root.rglob("*.md") if not any(part.startswith(".") for part in p.parts)
    )


def _read_body_sample(text: str, body: str) -> str:
    """Pick the right slice of body text for language detection.

    ``parse_frontmatter`` returns the body without the YAML header. If
    the file has no header, ``body`` already equals ``text`` and we
    just take its head. Either way the slice is bounded so we don't
    pull megabytes off disk.
    """
    return body[:_BODY_SAMPLE_CHARS] if body else text[:_BODY_SAMPLE_CHARS]


def _format_yaml_header(model: BrandFrontmatter) -> str:
    """Render a `BrandFrontmatter` as a YAML header block."""
    payload = model.model_dump(mode="json", exclude_none=True)
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{body}\n---\n\n"


def _apply_in_place(path: Path, raw_text: str, header: str) -> None:
    path.write_text(header + raw_text, encoding="utf-8")


def _classify_file(
    path: Path,
) -> tuple[ClassificationResult, dict[str, object], str, str]:
    raw_text = path.read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(raw_text)
    sample = _read_body_sample(raw_text, body)
    return classify_path(path, body_sample=sample), metadata, raw_text, body


def _merge_existing_with_classifier(
    metadata: dict[str, object],
    classification: ClassificationResult,
) -> BrandFrontmatter:
    """Build a :class:`BrandFrontmatter` from a partial existing header.

    The strategy is conservative: every field on the classifier's
    suggestion is the default; we *override* with the user's value
    when the user supplied one and it's of the right shape. Tags from
    both sources are unioned (existing tags first, classifier tags
    appended).
    """
    fm = classification.frontmatter
    overrides: dict[str, object] = {}

    raw_status = metadata.get("status")
    if isinstance(raw_status, str):
        mapped = _STATUS_REMAP.get(raw_status.lower(), raw_status.lower())
        if mapped in {"draft", "review", "published", "archive"}:
            overrides["status"] = mapped

    raw_lang = metadata.get("language")
    if isinstance(raw_lang, str) and raw_lang in {"en", "pl", "mixed"}:
        overrides["language"] = raw_lang

    raw_tags = metadata.get("tags")
    if isinstance(raw_tags, list):
        existing = [t for t in raw_tags if isinstance(t, str)]
        merged_tags: list[str] = []
        for tag in [*existing, *fm.tags]:
            if tag and tag not in merged_tags:
                merged_tags.append(tag)
        overrides["tags"] = merged_tags[:8]

    raw_series = metadata.get("series")
    if isinstance(raw_series, str):
        overrides["series"] = raw_series

    for date_key in ("created_at", "updated_at"):
        value = metadata.get(date_key)
        if value is not None:
            overrides[date_key] = value

    return fm.model_copy(update=overrides)


def _decide_action(
    path: Path,
    classification: ClassificationResult,
    metadata: dict[str, object],
) -> tuple[str, BrandFrontmatter | None]:
    """Return ``(action, header_to_write_or_none)``.

    Three outcomes:

    * ``skip-valid`` - existing header parses cleanly as
      :class:`BrandFrontmatter`. Nothing to do.
    * ``apply`` - file has no header. Classifier output is written.
    * ``merge`` - file has a header but it doesn't match the schema.
      We rebuild a valid header by overlaying the user's salvageable
      fields onto the classifier's suggestion. The original (invalid)
      header is replaced.
    """
    if metadata:
        validated = parse_brand_frontmatter(metadata)
        if validated is not None:
            return "skip-valid", None
        merged = _merge_existing_with_classifier(metadata, classification)
        return "merge", merged
    return "apply", classification.frontmatter


def _migrate_one(
    path: Path,
    *,
    apply: bool,
) -> tuple[str, str]:
    """Process a single file. Returns ``(action, rationale_summary)``."""
    classification, metadata, raw_text, body = _classify_file(path)
    action, header_model = _decide_action(path, classification, metadata)

    rationale = (
        f"category={classification.frontmatter.category} "
        f"({classification.confidence}) - {classification.rationale}"
    )

    if not apply or header_model is None:
        return action, rationale

    header = _format_yaml_header(header_model)
    if action == "apply":
        # File had no YAML header - prepend the classifier's output
        # straight onto the original text.
        _apply_in_place(path, raw_text, header)
    elif action == "merge":
        # File had an invalid header. ``body`` from python-frontmatter
        # is the file minus its YAML block, so rebuilding is just
        # ``new_header + body``. We strip a leading blank line off the
        # body so we don't accumulate whitespace on every re-run.
        cleaned_body = body.lstrip("\n")
        path.write_text(header + cleaned_body, encoding="utf-8")

    return action, rationale


def _write_log(
    log_dir: Path,
    apply: bool,
    rows: list[tuple[Path, str, str]],
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "apply" if apply else "dry-run"
    log_path = log_dir / f"{timestamp}-frontmatter-migration-{mode}.md"
    lines = [
        f"# Frontmatter migration log ({mode})",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Files processed: {len(rows)}",
        "",
        "| File | Action | Rationale |",
        "| --- | --- | --- |",
    ]
    for path, action, rationale in rows:
        rationale_safe = rationale.replace("|", "\\|")
        lines.append(f"| `{path}` | {action} | {rationale_safe} |")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def _run(args: argparse.Namespace) -> int:
    files: list[Path] = []
    for root in args.paths:
        if not root.exists():
            print(f"WARNING: {root} does not exist, skipping", file=sys.stderr)
            continue
        files.extend(_iter_md(root))

    if not files:
        print("No Markdown files matched the given paths.", file=sys.stderr)
        return 1

    rows: list[tuple[Path, str, str]] = []
    counters: dict[str, int] = {
        "apply": 0,
        "merge": 0,
        "skip-valid": 0,
    }
    for file_path in files:
        try:
            action, rationale = _migrate_one(file_path, apply=args.apply)
        except Exception as exc:
            logger.exception("Failed to process %s", file_path)
            rows.append((file_path, "error", str(exc)))
            continue
        counters[action] = counters.get(action, 0) + 1
        rows.append((file_path, action, rationale))

    log_path = _write_log(args.log_dir, args.apply, rows)
    print(f"Processed {len(files)} files. Log: {log_path}")
    for action, count in sorted(counters.items()):
        print(f"  {action}: {count}")
    if not args.apply:
        print("Dry-run only. Re-run with --apply to write changes.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
