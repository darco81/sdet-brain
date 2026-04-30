# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation

- docs: clarify Claude Desktop requires `mcp-remote` stdio bridge
  (HTTP transport works only in Claude Code CLI).

## [0.1.0] - 2026-04-30 - Tier 1 MVP shipped

First usable build. Persistent RAG for the SDET brand domain - shared
context across Claude Desktop, Claude Code, OpenCode, and any other
MCP-aware client over a single backend.

### Added

#### Project bootstrap (T1-01 / SDE-18)
- Python 3.12 project skeleton managed by `uv`.
- `pyproject.toml` declaring runtime dependencies (FastAPI,
  FastMCP 3, `qdrant-client`, `pydantic-settings`, `watchdog`,
  `python-frontmatter`, `httpx`, `uvicorn`, `google-genai`,
  `tenacity`, `tqdm`, `mlx-embeddings` on Apple Silicon) and dev
  tooling (`pytest`, `mypy`, `ruff`, `types-pyyaml`,
  `types-tqdm`).
- `sdet_brain.config.Settings` covering Qdrant, embedding
  providers, server ports, ingestion knobs, and watcher
  parameters.
- Multi-stage `docker/Dockerfile` and Compose scaffolding.
- `README.md` with Mermaid architecture diagram and quick start.
- `.env.example` listing every supported environment variable.
- Smoke tests covering the package import + default settings.

#### Storage layer (T1-02 / SDE-19)
- Qdrant `docker-compose` service with `/readyz` healthcheck,
  bind-mounted persistent storage, and a dedicated
  `sdet-brain-network` bridge.
- `QdrantStorage` facade wrapping ensure-collection, payload-index
  management, upsert, dense search via `query_points`, filter-based
  deletion, count, and status snapshots.
- `sdet_brain.storage.collections` exposing `COLLECTION_NAME`,
  `ChunkPayload` `TypedDict`, payload-index map, and idempotent
  `init_collections(name=COLLECTION_NAME)`.
- `sdet-brain-qdrant` CLI (`init` / `status` / `ping`).
- 7 storage tests against a live Qdrant container.

#### Embeddings layer (T1-03 / SDE-20)
- `IEmbedder` `Protocol` plus dual-path providers:
  `MLXEmbedder` (lazy `mlx-embeddings` load, batch 32, vectors from
  `BaseModelOutput.text_embeds`) and `GeminiEmbedder` (Google
  `google-genai` SDK with exponential-backoff retries via
  `tenacity.Retrying`).
- `sdet_brain.embeddings.factory.get_embedder` returning an
  `EmbedderSelection` that auto-falls-back when the primary
  provider fails its health check.
- `sdet-brain-embed` CLI (`encode` / `health`).
- 15 embedding tests (protocol contract, factory fallback against
  in-process fakes, Gemini transient-error retries, MLX lazy-load).

#### Ingestion pipeline (T1-04 + T1-05 / SDE-21 + SDE-22)
- Markdown ingestion stack in `sdet_brain.ingestion`:
  `Chunk` and `ParsedDocument` dataclasses, YAML frontmatter parser
  (graceful fallback on malformed YAML), block-aware semantic
  chunker (heading hierarchy, atomic code fences and Markdown
  tables, configurable target size and overlap), and
  `parse_markdown(path)` orchestrator with deterministic SHA-256
  content hashing.
- Test fixtures (`simple.md`, `voice-sample.md`, `complex.md`)
  plus 18 ingestion tests.
- End-to-end pipeline (`ingest_path`) walking sources, batching
  embeddings (default 32), and upserting deterministic UUID5 points
  into Qdrant. Re-ingestion short-circuits on `content_hash`
  matches; modifications trigger a delete-and-replace pass.
- Path-driven source classifier tagging chunks as
  `project-knowledge`, `drafts`, `articles`, `sprint-reports`, or
  `unknown`.
- `sdet-brain-cli` CLI (`--force`, `--exclude DIR`, `tqdm`
  progress bar) returning an `IngestStats` summary.
- 7 pipeline tests against a live Qdrant + deterministic fake
  embedder.

#### Server (T1-06 + T1-07 / SDE-23 + SDE-24)
- FastAPI application factory with a lifespan context that wires
  Qdrant + the embedder and reports degraded states through
  `/health`. Routes: `/health`, `/status`, `/search`, `/ingest`.
  OpenAPI at `/openapi.json`, Swagger UI at `/docs`.
- FastMCP 3 wrapper exposing the server as MCP tools across three
  transports - stdio (`sdet-brain-mcp-stdio`), SSE
  (`sdet-brain-mcp-sse`), and streamable HTTP mounted on the
  FastAPI app under `/mcp`.
- Four core MCP tools (plus a `ping` smoke probe), wired through
  `build_mcp(state_getter)` and a shared `build_default_state`
  helper:
  - `search(query, limit, source_type, min_score)` - Markdown-
    formatted dense-vector hits with score / heading / text.
  - `ingest_path(path, force)` - thin wrapper over the pipeline.
  - `list_sources(source_type)` - groups indexed chunks per
    source path with chunk count and last ingestion timestamp.
  - `get_chunk_neighbors(source_path, chunk_index, window)` -
    surrounding chunks clamped to file bounds.
- Chunk text persisted on the Qdrant payload so search results
  carry the original content alongside metadata.
- 19 server tests (8 health + 11 tool-level) using
  `fastapi.testclient.TestClient`.

#### Watcher (T1-08 / SDE-25)
- `BrainWatcher` (`watchdog.events.FileSystemEventHandler`
  subclass) with thread-safe debounced re-ingest queue, delete
  propagation via `delete_by_filter`, hidden / vendored /
  non-Markdown filtering, and a graceful drain on shutdown.
- `sdet-brain-watcher` CLI reading paths from `WATCH_PATHS`, with
  `SIGINT` / `SIGTERM` handling.
- Optional `watcher` profile in `docker/docker-compose.yml`.
- 9 watcher tests covering filter logic, debounce collapse, delete
  handling, directory-event suppression, and a live observer
  smoke.

#### Initial corpus + Tier 1 finalisation (T1-09 + T1-10 / SDE-26 + SDE-27)
- Ingested 76 files / 1486 chunks across four source types:
  `drafts` (1131), `articles` (137), `project-knowledge` (112),
  `sprint-reports` (106). Snapshot at
  `docs/sprints/v0.1.0-initial-ingest-snapshot.md`.
- Verified 5 sanity queries return relevant top-1 hits (scores
  0.59-0.78).
- Claude Desktop `mcpServers` entry wired (config backed up
  pre-change).
- README sections: Embeddings, Running the server, MCP tools,
  How to ingest your corpus, Live sync mode.
- Tier 1 sprint report at
  `docs/sprints/v0.1.0-tier-1-sprint-report.md`.

### Quality gates at release

- `uv run ruff check src tests` - 0 issues across 42 source files
  + tests.
- `uv run mypy --strict src` - 0 issues across 42 source files.
- `uv run pytest -q` - **77 passed**.

### Known limitations

- Single tenant (Dariusz only). Multi-tenant deferred to a future
  decision after the public-facing brand work lands.
- No hybrid search (dense + BM25); no reranking. Both land in
  Tier 2 (T2-03 / T2-04).
- No domain-specific MCP tools beyond the generic four. Tier 2
  (T2-02) adds search_voice_samples / search_decisions etc.
- Cosmetic: the Qdrant compose healthcheck uses bash `/dev/tcp/`
  which the Debian default `sh` lacks - container reports
  "(unhealthy)" even when responding correctly. Tracked as a Tier
  1 follow-up.
