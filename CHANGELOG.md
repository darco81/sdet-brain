# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Qdrant `docker-compose` service with `/readyz` healthcheck, persistent
  bind mount, and dedicated `sdet-brain-network` bridge.
- `QdrantStorage` facade in `sdet_brain.storage.qdrant_client` wrapping
  ensure-collection, payload-index management, upsert, dense search via
  `query_points`, filter-based deletion, count, and status snapshots.
- `sdet_brain.storage.collections` exposing `COLLECTION_NAME`,
  `ChunkPayload` `TypedDict`, payload-index map, and idempotent
  `init_collections`.
- `sdet-brain-qdrant` CLI (`init` / `status` / `ping`) wired as a console
  script and module-runnable via `python -m sdet_brain.cli.qdrant_cli`.
- Storage integration tests covering idempotent collection creation,
  upsert + search round-trip, filter-based deletion, payload-index
  registration, and parametrised batch sizes.
- `IEmbedder` `Protocol` plus dual-path embedding providers:
  `MLXEmbedder` (lazy `mlx-embeddings` load, batch 32, `text_embeds`
  output) and `GeminiEmbedder` (Google `google-genai` SDK with
  exponential-backoff retries via `tenacity`).
- `sdet_brain.embeddings.factory.get_embedder` returning an
  `EmbedderSelection` that auto-falls-back when the primary provider
  fails its health check.
- `sdet-brain-embed` CLI (`encode` / `health`) with provider chain
  reporting.
- Unit + smoke tests for the embedding stack: protocol contract, factory
  fallback against in-process fakes, Gemini transient-error retry path,
  and MLX lazy-load assertion (skipped on non-Apple-Silicon).
- Markdown ingestion stack in `sdet_brain.ingestion`: `Chunk` and
  `ParsedDocument` dataclasses, YAML frontmatter parser (graceful fall
  back on malformed YAML), block-aware semantic chunker (heading
  hierarchy, atomic code fences and Markdown tables, configurable
  target size and overlap), and `parse_markdown(path)` orchestrator
  with deterministic SHA-256 content hashing.
- Test fixtures (`simple.md`, `voice-sample.md`, `complex.md`) plus 18
  ingestion tests covering chunker behaviour, frontmatter edge cases,
  and end-to-end document parsing.
- End-to-end ingestion pipeline: `sdet_brain.ingestion.pipeline`
  walks Markdown sources, batches embeddings (default 32), and upserts
  deterministic UUID5 points into Qdrant. Re-ingestion short-circuits
  on `content_hash` matches; modifications trigger a delete-and-replace
  pass.
- Path-driven source classifier
  (`sdet_brain.ingestion.source_classifier`) tagging chunks as
  `project-knowledge`, `drafts`, `articles`, `sprint-reports`, or
  `unknown`.
- `sdet-brain-ingest` CLI with progress bar (`tqdm`), `--force`
  bypass, and an `IngestStats` summary.
- 7 pipeline tests against a live Qdrant container with a
  deterministic 16-dim fake embedder, covering single/directory
  ingest, cache hits, modify-and-replace, `--force`, and source-type
  tagging.
- README "How to ingest your corpus" section.
- Chunk text is now persisted on the Qdrant payload (`payload.text`)
  so search results carry the original content alongside metadata.
- FastAPI application factory (`sdet_brain.server.app:create_app`)
  with a lifespan context that lazily wires Qdrant + the embedder and
  reports degraded states through `/health`. Routes:
  `/health`, `/status`, `/search`, `/ingest`. OpenAPI spec at
  `/openapi.json`, Swagger UI at `/docs`.
- FastMCP 3 wrapper exposing the server as MCP tools across three
  transports - stdio (`sdet-brain-mcp-stdio`), SSE
  (`sdet-brain-mcp-sse`), and streamable HTTP mounted on the FastAPI
  app under `/mcp`. T1-06 ships a placeholder `ping` tool; T1-07 adds
  the real surface.
- Server tests using `fastapi.testclient.TestClient` covering
  healthy/degraded/unavailable health responses, OpenAPI exposure,
  and lifespan resilience to a missing Qdrant.
- README "Running the server" section with Claude Desktop config.
- Four core MCP tools wired through `build_mcp(state_getter)` and a
  shared `build_default_state` helper:
  - `search(query, limit, source_type, min_score)` - Markdown-
    formatted dense-vector hits with score / heading / text.
  - `ingest_path(path, force)` - thin wrapper over the T1-05
    pipeline returning a `IngestStats` summary.
  - `list_sources(source_type)` - groups indexed chunks per source
    path with chunk count and last ingestion timestamp.
  - `get_chunk_neighbors(source_path, chunk_index, window)` -
    returns the surrounding chunks clamped to file bounds for
    context expansion.
- 11 tool-level tests against a live Qdrant + deterministic fake
  embedder (filter behaviour, empty corpus, edge cases at chunk 0
  and at total - 1).
- README "MCP tools" reference table.
- File watcher daemon (`sdet_brain.ingestion.watcher.BrainWatcher`)
  with thread-safe debounced re-ingest queue, delete propagation via
  `delete_by_filter`, hidden / vendored / non-Markdown filtering,
  and a graceful drain on shutdown.
- `sdet-brain-watcher` CLI entrypoint reading watch paths from
  `WATCH_PATHS`, with `SIGINT` / `SIGTERM` handling.
- Optional `watcher` profile in `docker/docker-compose.yml` so the
  daemon can run alongside Qdrant via `docker compose --profile
  watcher up`.
- Watcher tests: path filtering (Markdown / hidden / `node_modules`),
  debounce collapse, delete handling, directory-event suppression,
  and a live observer round-trip against the test Qdrant container.
- README "Live sync mode" section.

## [0.1.0] - 2026-04-30 - Initial bootstrap

### Added
- Python 3.12 project skeleton managed by `uv`.
- `pyproject.toml` declaring core runtime dependencies (FastAPI, FastMCP 3.0+,
  qdrant-client, pydantic-settings, watchdog, python-frontmatter, httpx) and
  dev tooling (pytest, mypy, ruff).
- `src/sdet_brain/config.py` with `pydantic-settings` `Settings` covering
  Qdrant, embedding providers, server ports, and ingestion knobs.
- Package layout with placeholder modules for ingestion, embeddings, storage,
  server, and CLI layers.
- Docker scaffolding: `docker/docker-compose.yml` running Qdrant (with the
  `sdet-brain` service stubbed for later phases) and a multi-stage
  `docker/Dockerfile`.
- `README.md` with project overview, Mermaid architecture diagram, and quick
  start instructions.
- `.env.example` documenting every environment variable consumed by the app.
- Smoke test ensuring the package imports and default settings load.

### Notes
- This release ships scaffolding only. Qdrant runtime config (T1-02),
  embeddings (T1-03), and the markdown parser (T1-04) follow as separate
  Linear tasks.
