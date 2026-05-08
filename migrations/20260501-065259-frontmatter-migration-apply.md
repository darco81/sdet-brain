# Frontmatter migration log (apply)

Generated: 2026-05-01T06:52:59
Files processed: 88

> Migration log anonymized for the v0.5.1 public release. Original
> file paths point into a private corpus and were collapsed into the
> aggregate counts below. The migration tool itself is in
> `src/sdet_brain/cli/migrate_frontmatter.py` and is reproducible on
> any Markdown corpus.

## Aggregate breakdown

| Action | Count | Notes |
| --- | --- | --- |
| `apply` | 73 | files without prior frontmatter - schema written from classifier output |
| `merge` | 15 | files with prior frontmatter - only missing fields filled |

## Category distribution (post-migration)

| Category | Count |
| --- | --- |
| `prompt` | 23 |
| `sprint-report` | 8 |
| `decision` | 5 |
| `brand-strategy` | 5 |
| `execution-plan` | 3 |
| `raw-notes` | 4 |
| `case-study` | 1 |
| `outline` | 1 |
| `smaczki` | 1 |
| `other` | 37 |

## Reproducing on your own corpus

```bash
uv run python -m sdet_brain.cli.migrate_frontmatter \
  /path/to/your/markdown/corpus \
  --apply
```

The `--dry-run` mode writes a sibling `*-dry-run.md` log (gitignored)
which you can inspect before committing the schema in-place.
