# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Project highlights

- **6 sprints** between 2026-04-30 and 2026-05-01 - Tier 1 MVP
  through Tier 5 DX.
- **213 tests** passing; mypy --strict + ruff clean across 70 source
  files.
- **2906+ chunks** indexed across the brand corpus + the brain's own
  README / CHANGELOG / sprint reports.
- **11 MCP tools** (5 core + 5 domain + 3 LLM-backed) plus a REST
  API with SSE streaming on `/chat`.
- **Top-tier 2026 stack**: Qwen3-Embedding-8B-4bit-DWQ (MTEB 70.58)
  with MRL truncation to 1024, hybrid dense + BM25 fused via RRF,
  cross-encoder reranking, tiered local MLX LLMs (gemma-4-26B fast /
  Qwen3-Next-80B Instruct + Thinking), multi-query agentic
  retrieval.
- **Daily ergonomics**: `sdet-brain-chat` REPL, inline `[N]`
  citations with structured Source panel, `sdet-brain run`
  templates with Jinja2 substitution.
- **Zero cloud dependencies, zero API keys** for the core flow.
  Gemini stays available as a typed fallback for VPS deploys.

## [Unreleased]

### Fixed

- `_iter_markdown_files`: bare directory names in `--exclude` now match
  at any depth (gitignore-style). Previously `sdet-brain-cli <path>
  --exclude node_modules` only dropped `$(pwd)/node_modules` because
  argparse `type=Path` turns the bare name into a relative `Path` and
  `resolve()` anchors it to CWD. Real-world impact: ingesting a repo
  that contained `node_modules/` polluted the corpus with thousands
  of vendored README chunks (8079 in our brand collection - 71% of
  total). Absolute paths and relative-with-slashes still resolve as
  before, so existing callers are unaffected.

Backlog (post-Series #01 publication week):

- VPS deployment with `brand.sdet.it` + HMAC auth + Gemini fallback (`SDE-35`)
- SQLite conversation persistence with FTS5 search (`SDE-80`)
- Reranker upgrade to Qwen3-Reranker MLX (`SDE-66`)
- GraphRAG-lite entity + relation extraction (`SDE-67`)
- PDF and image ingestion (`SDE-68`, `SDE-69`)

Each backlog item has explicit reopen criteria in Linear.

## [0.5.0] - 2026-05-01 - Tier 5 DX: citations + REPL + templates

May Day evening Tier 5 DX sprint. Three feature commits, daily-use
quality-of-life improvements on top of the v0.4.0 stack.

### Added

- **Citation-aware chat** (`SDE-77`). System prompt instructs the LLM
  to mark factual statements with inline `[N]` markers (and `[N][M]`
  for combined sources). `ChatResponse.sources` is now a list of
  structured `Source` objects (`n`, `source_path`, `chunk_index`,
  `score`, `snippet`) instead of bare path strings. SSE final frame
  ships `Source.model_dump(mode="json")`. HTML test client renders a
  collapsible `<details>` Sources panel with paths, scores, and
  italic snippets.

- **`sdet-brain run` template runner** (`SDE-79`). YAML query
  templates with Jinja2 substitution, scanned from
  `~/.sdet-brain/templates/` and `examples/templates/` (user
  templates win on name collision). Subcommands:
  `sdet-brain run NAME [--var KEY=VAL]`,
  `sdet-brain template list`, `sdet-brain template show NAME`.
  Tools dispatch in-process so the daemon doesn't need to be
  running. Four pre-shipped templates ship under
  `examples/templates/`: voice-check (search_voice_samples),
  series-status (multi_query_search), decision-history
  (search_decisions), wcag-fact-check (search with sprint-reports
  filter).

- **`sdet-brain-chat` REPL** (`SDE-78`). Terminal chat client backed
  by the live `/chat` SSE endpoint. prompt_toolkit drives input +
  persistent history at `~/.sdet-brain/chat_history`. httpx streams
  tokens. Slash commands: `/help`, `/clear`, `/sources` (renders
  structured Source as `[N] path score=… snippet…`),
  `/save NAME` + `/load NAME` (JSON round-trip in
  `~/.sdet-brain/conversations/`), `/quit`. `parse_command()` is a
  pure function so unit tests cover every branch without
  prompt_toolkit or httpx.

### Dependencies

- Added `prompt-toolkit>=3.0`.

### Tests

- 186 → 213 (+27): 2 chat citation tests, 13 template tests, 12 REPL tests.

### Quality gates at release

- `uv run ruff check src tests` - clean.
- `uv run mypy --strict src` - 70 source files clean.
- `uv run pytest -q` - **213 passed**.

### Atomic commits

- `0a1bc9e` feat(chat): inline citations [N] + structured sources panel (SDE-77)
- `6066d76` feat(cli): saved templates + chat REPL (sdet-brain, sdet-brain-chat) (SDE-79, SDE-78)

### Deferred (created in Linear with explicit reopen criteria)

- `SDE-80` SQLite conversation persistence + FTS5 + `recall_conversation`
  MCP tool - 2h+ scope spanning schema migration, `/chat` endpoint
  backwards-compat plumbing, 4 new REST routes, new MCP tool, opt-out
  Settings flag, and REPL integration. Reopen post-Series #01 with a
  fresh session budgeted for the full design. The REPL's existing
  `/save NAME` + `/load NAME` JSON round-trip covers the casual
  "save this train of thought" use case in the meantime.

## [0.4.0] - 2026-05-01 - Tier 4 brain: tiered routing + agentic retrieval + 8B embedder

May Day evening Tier 4 ALL-IN sprint. Three substantive code shipments
plus four explicit deferrals.

### Added

- **Tiered LLM routing** (`SDE-63`). New `sdet_brain.llm.router` with
  `LLMRouter.select_model(task)` mapping six task types onto three
  tiers: `gemma-4-26B-A4B-it-OptiQ-4bit` (fast), `Qwen3-Next-80B-A3B-Instruct-4bit`
  (instruct/chat/summarize), `Qwen3-Next-80B-A3B-Thinking-4bit`
  (reasoning/decompose/judge). Per-model `MLXLLm` cache so the second
  call to a given task pays no cold start. `query_rewrite` now fires
  the gemma-4 fast tier; `summarize_results` uses Qwen-Next-Instruct.
  `Settings.LLM_ROUTING_ENABLED=False` collapses to v0.3.0 single-model
  behaviour.

- **Multi-query agentic retrieval** (`SDE-64`). New MCP tool
  `multi_query_search`. Decomposes a multi-hop / compound question
  into 3-5 sub-queries via the Thinking tier, hybrid-searches each in
  turn, fuses the ranked lists with Reciprocal Rank Fusion (k=60),
  de-dupes by chunk id, returns the merged top-K alongside the
  decomposition for auditability. Robust JSON extraction handles
  fenced and bare LLM replies; falls back to single-query behaviour
  on parse failure. Brain now exposes 11 MCP tools (was 10).

- **Embedding upgrade** (`SDE-65`). Default dense embedder swapped from
  `Qwen/Qwen3-Embedding-0.6B` to `mlx-community/Qwen3-Embedding-8B-4bit-DWQ`
  (top of MTEB at the time of the upgrade, 70.58). MRL truncation
  takes the leading 1024 dims of the 4096-dim native output (~95%
  retention per the published Qwen3 evaluations) so the existing
  Qdrant collection schema is unchanged. New `MLXEmbedder.mrl_truncate_to`
  parameter and `Settings.MLX_MRL_TRUNCATE_TO` (default 1024).

### Migration

- Pre-upgrade Qdrant snapshot:
  `sdet_brand_v1-3556520363950657-2026-05-01-16-04-43.snapshot` (1.4 GB)
  in the container, with a redundant local copy at
  `/tmp/qdrant-snapshot-pre-tier4-20260501-1804.snapshot`. Pre-upgrade
  git tag `pre-tier4-20260501-1804` for one-command code rollback.
- Collection wiped and re-ingested with the 8B embedder. Final state
  after self-knowledge re-ingest: **2882 chunks** across the brand
  corpus + the brain's own README, CHANGELOG, and `docs/sprints/`.

### Smoke regressions (preserved)

- "multi-page audit strategy" → MULTI-PAGE-AUDIT-FEATURE-PROMPT.md
- "port-collision" hyphenated keyword → THURSDAY-DEPLOY-SPRINT-REPORT.md
- `category=smaczki` payload filter → case-study-01-SMACZKI.md

### Tests

- 165 → 186 (+21): 11 router, 10 multi-query.

### Quality gates at release

- `uv run ruff check src tests` - clean.
- `uv run mypy --strict src` - 67 source files clean.
- `uv run pytest -q` - **186 passed**.

### Atomic commits

- `a0c1189` feat(llm): tiered routing (gemma-4 / Qwen-Next / Thinking) (SDE-63)
- `5ff18a2` feat(server): multi-query agentic retrieval with Thinking decomposition (SDE-64)
- `80aa336` feat(embeddings): Qwen3-Embedding-8B-4bit-DWQ with MRL 1024-dim truncation (SDE-65)

### Deferred (created in Linear with explicit rationale)

- `SDE-66` Reranker upgrade - Qwen3-Reranker MLX variants are
  decoder-style, not cross-encoders. Wrapping them as fastembed-
  compatible cross-encoders is a bigger lift than the gain over
  jinaai-v2 at this corpus shape. Reopen on demand.
- `SDE-67` GraphRAG-lite - entity/relation extraction over 2700
  chunks is a 90-180 min batch; doesn't fit cleanly in autonomous
  scope. Reopen with a supervised batch + checkpoint scaffold.
- `SDE-68` PDF ingestion - no PDFs in the active corpus paths.
  YAGNI; reopen when an actual PDF lands.
- `SDE-69` Image ingestion via Ollama qwen3-vl - corpus has very
  few image references. Reopen when Series #02+ brings visual
  material.

Block 3 of the May-Day overnight sprint. Adds local-first LLM
inference and a conversational chat endpoint on top of the v0.2.0
hybrid retrieval stack. **Zero API keys.**

### Added

- **Local MLX LLM** (`SDE-32`). New `sdet_brain.llm.protocol`
  defines `ILLM` (generate / chat / chat_stream / health_check) plus
  `ChatMessage` and `LLMError`. `mlx_provider.MLXLLm` wraps
  `mlx_lm.load + generate + stream_generate`; default model
  `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`. Thread-safe
  lazy load: ~30-60s cold start on M4 Pro, ~70 tok/s warm.
  `Settings` exposes `LLM_MODEL`, `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`.

- **Two new MCP tools** built on the local LLM:
  - `query_rewrite(query, limit, source_type)` - HyDE pattern. Local
    LLM writes a hypothetical answer paragraph in the corpus's
    voice, the paragraph (not the bare query) is hybrid-searched.
    Output Markdown shows both the hypothetical and matched chunks
    for auditability. Use for terse / under-specified queries.
  - `summarize_results(topic, limit, source_type)` - hybrid-search
    a topic, feed top chunks to LLM with a brand-aware system
    prompt, return one concise paragraph with `[n]` inline
    citations plus a Sources section. Polish topic → Polish
    summary; English → English. Use when the user wants the answer,
    not a list of chunks.

- **POST /chat** (`SDE-33`) - multi-turn conversational endpoint
  with optional Server-Sent Events streaming. Stateless server:
  every request carries the full history. The latest user turn is
  hybrid-searched against the corpus; retrieved chunks land in the
  system prompt so the LLM cites them inline.
  - JSON response (`stream=false`): `{reply, sources,
    retrieved_chunk_count}`.
  - SSE response (`stream=true`): `data: {"text": "..."}` per
    token, terminated by `data: {"event": "done", "sources": [...],
    "retrieved": N}`.
  - Brand-voice system prompt (Polish-default, blunt, cite-or-admit).
  - HTML test client checked in at `docs/chat-test.html` - open in
    a browser, point it at `localhost:8080/chat`, submit a turn,
    watch tokens stream in.

### Tests

- 159 → 165 passing (+6 chat tests, +9 LLM protocol/factory tests
  earlier in the day = +15 since v0.2.0).

### Quality gates at release

- `uv run ruff check src tests` - clean.
- `uv run mypy --strict src` - 65 source files clean.
- `uv run pytest -q` - **165 passed**.

### Atomic commits

- `9542ceb` feat(llm): MLX local LLM + query_rewrite + summarize tools (SDE-32)
- `6b3c3cf` feat(server): conversational chat with SSE streaming (SDE-33)

### Deferred

- `SDE-34` (RAGAS eval) - local-LLM-as-judge custom scaffold not
  justified within the weekend budget. Reopen with eval-shaped time;
  the LLM + chat layer is already production-shaped, so future eval
  work is pure harness-building.

## [0.2.0] - 2026-05-01 - Tier 2 brain: hybrid + reranking + taxonomy + domain tools

Block 1 + Block 2 of the May-Day overnight sprint.

### Added

- **Cross-encoder reranking** (`SDE-31`). New
  `sdet_brain.embeddings.reranker` exposing `IReranker`,
  `FastembedReranker`, `RerankCandidate`, `RerankResult`,
  `RerankerError`. Lazy-loads ONNX weights on first call. POST
  `/search` and the MCP `search` tool gain an opt-in `rerank` flag
  that over-fetches and re-orders via cross-encoder. Default model:
  `jinaai/jina-reranker-v2-base-multilingual` (PL+EN aware; the
  spec'd `BAAI/bge-reranker-v2-m3` is not in fastembed's registry).

- **Structured frontmatter taxonomy + payload indexes** (`SDE-28`).
  `BrandFrontmatter` Pydantic model (category, tags, status, series,
  episode, part, language, created_at, updated_at). Validation is
  graceful - failed parse logs WARNING and the file still ingests
  with raw header preserved. Pipeline lifts validated fields onto
  top-level Qdrant payload keys; 7 new payload indexes (`category`,
  `status`, `tags`, `series`, `language` keyword + `fm_created_at`,
  `fm_updated_at` datetime). Migration CLI
  (`uv run python -m sdet_brain.cli.migrate_frontmatter`) walks the
  corpus, classifies every file from filename heuristics, prepends
  YAML where missing, and *merges* into existing-but-invalid
  headers (preserving user-supplied tags / dates while normalising
  free-form statuses like `in-progress` → `draft`,
  `completed`/`done`/`decided`/`log` → `published`). Dry-run by
  default; `--apply` writes in place. Migration log committed under
  `migrations/`.

- **5 domain-specific MCP tools** (`SDE-29`) wrapping the search
  pipeline with preset payload filters:
  - `search_voice_samples` (category=voice-sample)
  - `search_smaczki` (category=smaczki)
  - `search_decisions(topic, since=YYYY-MM-DD)` - DatetimeRange on
    `fm_created_at`
  - `list_articles_by_status(status, series)` - scrolls case-study
    chunks, groups by file
  - `search_sprint_reports(query, project)` - `project` maps to
    `series`
  - Each tool's MCP description carries an explicit "Use when…"
    sentence so the LLM picks the right tool from the user's
    phrasing.

- **Hybrid search (dense + BM25 + RRF)** (`SDE-30`) - the headline
  Tier 2 win. New `sdet_brain.embeddings.sparse_embedder` exposes
  `ISparseEmbedder` Protocol and `FastembedBM25` impl. Collection
  is now named-vector: `dense` (cosine 1024) + sparse `bm25` (IDF
  modifier). `QdrantStorage.hybrid_search` runs two `Prefetch`es
  under `FusionQuery(Fusion.RRF)`; the same payload filter scopes
  both legs. Default search route runs hybrid; `hybrid: false` in
  the request opts out. The win is on hyphenated keywords and
  exact tokens (`port-collision`, `WCAG 2.2 AA`) where dense alone
  generalises tokens away. Benchmark
  (`docs/benchmarks/hybrid-vs-semantic.md`) shows 3 sample queries
  before/after.

### Migration

- Production collection wiped and recreated with named vectors.
  Pre-recreate Qdrant snapshot saved as
  `sdet_brand_v1-3556520363950657-2026-05-01-05-36-24.snapshot`
  (1.13 GB) inside the `sdet-brain-qdrant` container, with a
  redundant local copy at
  `/tmp/qdrant-snapshot-pre-hybrid-20260501-0736.snapshot`. Full
  corpus re-ingested: 139 files / 2700 chunks across drafts,
  strategy, brief, from-the-field, and both sprint-report
  directories.

### Tests

- 82 → 144 (+62) at the end of Block 2: 11 reranker, 36
  frontmatter (schema + classifier), 13 domain-tool, 2 hybrid
  storage tests.

### Quality gates at release

- `uv run ruff check src tests` - clean.
- `uv run mypy --strict src` - 54 source files clean.
- `uv run pytest -q` - **144 passed**.

### Atomic commits

- `0348db4` feat(search): cross-encoder reranking layer (SDE-31)
- `56539ac` feat(ingestion): structured frontmatter schema + payload indexes (SDE-28)
- `652888d` feat(server): 5 domain-specific MCP tools (SDE-29)
- `d1e5dd3` feat(search): hybrid semantic + BM25 with RRF fusion (SDE-30)

## [0.1.1] - 2026-04-30 - Tier 1 polish

Phase A of the Tier 2/3 overnight sprint - four T1 follow-ups
captured in the v0.1.0 sprint report.

### Fixed

- **Qdrant compose healthcheck reports `healthy`** (`SDE-37`).
  The `latest` Qdrant image strips `wget`, `curl`, `nc`, and
  `python` - only `bash` is available. Switched the probe from
  `CMD-SHELL` (which invokes `dash`) to `CMD` with explicit
  `bash -c` so the existing `/dev/tcp` redirect works. Container
  flips to `healthy` within ~15 s.

### Changed

- **Brand corpus paths now configurable via env vars** (`SDE-38`,
  unblocks T3-03 VPS deploy).
  New `Settings` fields: `PROJECT_KNOWLEDGE_PATHS`, `DRAFTS_PATHS`,
  `ARTICLES_PATHS`, `SPRINT_REPORTS_PATHS`, `BRIEF_PATHS`. Each is
  comma-separated; empty falls back to the local-dev defaults in
  `cli/ingest_cli.py:LOCAL_DEFAULT_PATHS`. The watcher CLI shares the
  same dict so a single `.env` controls both ingestion modes.
  README "Configure your corpus paths" subsection added.

- **Documentation:** clarified that Claude Desktop requires the
  `mcp-remote` stdio bridge - HTTP transport in `mcpServers` only
  works for Claude Code CLI.

### Performance

- **Batch cache-check during directory ingest** (`SDE-39`,
  O(N) -> O(1) round-trips). New `_load_existing_hashes()` issues a
  single Qdrant scroll using `MatchAny` over the union of source
  paths and builds an in-memory `{path: content_hash}` dict before
  the file walk. Single-file ingests (REST `/ingest`, watcher
  events) keep the per-file path - no overhead.

- **Chunker merges sub-250-char trailing sections** (`SDE-40`).
  New `_merge_small_tails()` post-process pass folds tiny tails
  into their predecessor when (a) the tail is below
  `SMALL_TAIL_THRESHOLD_CHARS = 250`, (b) the previous chunk did
  NOT end in a code fence (atomic protection), and (c) the
  combined size stays at or below 1.5x target. Re-ingest of the
  78-file `sdet-brand-drafts` directory: -3.2% chunks (2409 -> 2333).

### Tests

- 77 -> 82 (5 new chunker tests covering simple merge, code-block
  anchor protection, upper-bound enforcement, threshold boundary
  at 250, and post-merge index renumbering).

### Quality gates at release

- `uv run ruff check src tests` -> 0 issues.
- `uv run mypy --strict src` -> 42 source files clean.
- `uv run pytest -q` -> 82 passed.

### Atomic commits

- `b6af9ae` chore(docker): use bash + /dev/tcp for Qdrant healthcheck
- `b209a91` refactor(config): brand corpus paths via env vars (T3-03 prep)
- `15ba851` perf(ingestion): batch cache-check via single scroll
- `2e7abd9` perf(chunker): merge sub-250-char trailing sections

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
