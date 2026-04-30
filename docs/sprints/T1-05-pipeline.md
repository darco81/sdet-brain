# Sprint Report: T1-05 End-to-end ingestion pipeline

**Linear:** [SDE-22](https://linear.app/sdet-it/issue/SDE-22/t1-05-ingestion-pipeline-walker-chunk-embed-store-caching)
**Started:** 2026-04-30 15:50 (CET)
**Done:** 2026-04-30 16:00 (CET)
**CC time:** ~10 min
**Dariusz manual:** 0 min

## What shipped

- `src/sdet_brain/ingestion/source_classifier.py`: `SourceConfig`
  dataclass plus `classify_source(path, config)` returning one of
  `project-knowledge`, `drafts`, `articles`, `sprint-reports`,
  `unknown`. Project-knowledge filename pattern wins over the
  containing directory so a stray `01-PROJECT-CONTEXT-*` file inside
  the drafts tree still tags as project-knowledge.
- `src/sdet_brain/ingestion/pipeline.py`: `IngestStats` aggregate plus
  `ingest_path(path, storage, embedder, ...)` orchestrator. Per file:
  parse, classify, look up the cached `content_hash` via Qdrant
  scroll, skip on match, otherwise delete the old chunks and embed
  + upsert the new ones. Deterministic UUID5 chunk IDs (namespace
  `https://sdet-brain/chunks`, name `source_path#chunk_index`) keep
  point identity stable across re-embeds. Per-file errors are
  collected into `stats.errors` rather than aborting the walk.
- `src/sdet_brain/cli/ingest_cli.py`: `sdet-brain-ingest` console
  script. Initialises the collection (idempotent), wraps the file
  iterator in `tqdm`, prints the stats summary, exits non-zero only
  when at least one file errored.
- 7 pipeline tests against a live Qdrant container with a
  deterministic 16-dim fake embedder (no MLX startup cost): single
  file, directory walk, cache hit on re-run, modify + replace,
  `--force` bypass, source-type payload tagging, summary text shape.
- README "How to ingest your corpus" section + CHANGELOG
  `[Unreleased]` entry.

## Atomic commit

- `<sha> feat(ingestion): end-to-end pipeline with content-hash caching`

## Numbers

- Files added: 4 source + 1 test.
- Files modified: 4 (CHANGELOG, README, pyproject.toml, uv.lock).
- Tests added: 7 (49 total: 2 T1-01 + 7 T1-02 + 15 T1-03 + 25 T1-04 +
  T1-05 7).
- Quality gates: ruff clean, mypy strict 24 source files clean, pytest
  49/49.

## Runtime smoke (with real MLX + real Qdrant)

```
$ sdet-brain-ingest tests/ingestion/fixtures
Processed 3 files, created 7 chunks, skipped 0 files (cache), replaced 0 chunks

$ sdet-brain-ingest tests/ingestion/fixtures
Processed 0 files, created 0 chunks, skipped 3 files (cache), replaced 0 chunks

$ sdet-brain-ingest tests/ingestion/fixtures --force
Processed 3 files, created 7 chunks, skipped 0 files (cache), replaced 3 chunks

$ sdet-brain-qdrant status
points_count: 7
```

The cache hit is the bool answer to AC #3, the `--force` re-embed is
the answer to AC #5, and the chunk count remains 7 across all three
runs - confirming the delete-by-filter pass before re-upsert.

## Lessons learned

- Putting `force_reindex` *and* `progress` after a `*` made the
  signature a lot easier to read at the call site (everything except
  `path`/`storage`/`embedder` is keyword-only). Worth keeping the
  pattern as the pipeline grows.
- Tagging `source_type=drafts` (and friends) at ingest time means
  domain MCP tools (T2-02) can filter without running a second
  lookup. The classifier defaults to `unknown` so files outside the
  registered roots still ingest cleanly - we just lose the filter
  affordance for them, which is the correct trade-off.
- The Qdrant compose healthcheck is currently flagged "unhealthy"
  even though `/readyz` returns OK; the probe relies on bash's
  `/dev/tcp/`, which `dash` (the default `sh` on Debian) lacks.
  Cosmetic only - functionality works. Worth fixing in a small
  follow-up so `docker compose ps` shows green.

## Out-of-scope items captured

- **Suggested follow-up (Linear ticket idea):** swap the
  Qdrant healthcheck to `wget -qO- http://localhost:6333/readyz` so
  it works under `dash`. Not blocking T1-06.
- The CLI's hard-coded brand corpus paths (drafts/articles/
  sprint-reports) live in `_build_source_config`. Once T1-09 ingests
  the real corpus we'll likely lift these to `config.py` env vars so
  the same CLI works on the VPS.

## Next task

- T1-06 (SDE-23) FastAPI + FastMCP server skeleton. Unblocked.
