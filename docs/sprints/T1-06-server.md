# Sprint Report: T1-06 FastAPI + FastMCP server skeleton

> Sprint report autorski. Linear refs (`SDE-XX`) są internal trackingiem
> i nie są publicznie linkowane.


**Linear:** [SDE-23](https://linear.app/sdet-it/issue/SDE-23/t1-06-fastapi-fastmcp-server-skeleton-stdio-sse-http)
**Started:** 2026-04-30 16:00 (CET)
**Done:** 2026-04-30 16:18 (CET)
**CC time:** ~18 min
**Dariusz manual:** 0 min

## What shipped

- `src/sdet_brain/server/dependencies.py`: `AppState` dataclass and
  `Depends`-friendly accessors (`get_state`, `require_storage`,
  `require_embedder`). Routes never touch `app.state` directly.
- `src/sdet_brain/server/app.py`: `create_app()` factory + async
  lifespan that builds a `QdrantStorage` and an `EmbedderSelection`,
  recording errors on the state rather than crashing. Exposes
  OpenAPI at `/openapi.json` and Swagger UI at `/docs`.
- `src/sdet_brain/server/__main__.py`: `python -m sdet_brain.server`
  entrypoint that runs `uvicorn` with the app factory.
- `src/sdet_brain/server/mcp_server.py`: `build_mcp()` returning a
  FastMCP instance with a placeholder `ping` tool (T1-07 adds the
  real surface).
- `src/sdet_brain/server/mcp_stdio.py` + `mcp_sse.py`: standalone
  process entrypoints; logs go to stderr on stdio so they don't
  collide with the MCP protocol on stdout.
- REST routes:
  - `GET /health`: status (`ok` / `degraded` / `unavailable`),
    `qdrant_ok`, `embedder_ok`, embedder provider + fallback flag,
    vector size, collection count, plus per-component error
    strings.
  - `GET /status`: scrolls payloads to compute the source-type
    breakdown and the most recent `created_at`.
  - `POST /search`: dense-vector query via the active embedder,
    optional `source_type` filter, returns the chunk text + payload.
  - `POST /ingest`: thin wrapper around
    `sdet_brain.ingestion.pipeline.ingest_path` returning the same
    `IngestStats` summary.
- Streamable HTTP MCP transport mounted on the FastAPI app at `/mcp`.
- Pipeline payload now stores the chunk text under `payload.text`
  (small follow-on tweak from T1-05) so `/search` returns content
  alongside metadata.
- 8 server tests using FastAPI's `TestClient`: ok / degraded
  (qdrant down) / degraded (embedder down) / unavailable health
  responses, OpenAPI exposure, lifespan resilience, MCP attachment.
- README "Running the server" section with Claude Desktop config
  snippet.
- CHANGELOG `[Unreleased]` entry.

## Atomic commit

- `<sha> feat(server): FastAPI + FastMCP dual transport scaffolding`

## Numbers

- Files added: 11 (server module + 4 routes + entrypoints) + 3 tests.
- Files modified: 4 (CHANGELOG, README, pyproject.toml, pipeline.py).
- Tests added: 8 (57 total).
- Quality gates: ruff clean, mypy strict 34 source files clean,
  pytest 57/57 in 8.89 s.

## Runtime verification

```
$ uv run sdet-brain-server &
INFO Server ready (qdrant_ok=True, embedder_ok=True)
INFO Uvicorn running on http://127.0.0.1:8080

$ curl localhost:8080/health
{"status":"ok","qdrant_ok":true,"embedder_ok":true,
 "embedder_provider":"mlx","embedder_fell_back":false,
 "vector_size":1024,"collection_count":7,
 "qdrant_error":null,"embedder_error":null}

$ curl localhost:8080/openapi.json | jq '.paths | keys'
["/health","/status","/search","/ingest"]

$ curl -X POST localhost:8080/search \
       -H 'Content-Type: application/json' \
       -d '{"query":"voice samples","limit":2}' | jq '.results[].score, .results[].source_path'
0.5369706
"tests/ingestion/fixtures/voice-sample.md"
0.341381
"tests/ingestion/fixtures/simple.md"
```

The 0.537 cosine similarity between the query "voice samples" and the
voice sample fixture is exactly what we want from the MLX
`Qwen/Qwen3-Embedding-0.6B` model - the brand voice fixture wins
clearly over the simple/complex fixtures.

## Lessons learned

- FastAPI's `Depends()` pattern uses function-call defaults by design;
  ruff's `B008` tags it. Added a per-file ignore for
  `server/routes/*.py` and `dependencies.py` rather than peppering
  `# noqa` everywhere.
- `qdrant-client` runs a background compatibility probe at
  construction time. Pointing it at an unreachable URL leaks an
  exception out of that thread, which pytest converts into a
  `PytestUnhandledThreadExceptionWarning` -> error under our
  `filterwarnings = ["error"]` config. Annotated the affected
  lifespan test with a per-test `filterwarnings("ignore::...")`.
- Storing chunk text on the Qdrant payload turns `/search` into a
  one-shot read - no separate fetch needed. Re-ingesting old chunks
  picks up the new payload field automatically.

## Out-of-scope items captured

None. T1-07 already covers the actual MCP tool surface and is the
natural next task.

## Next task

- T1-07 (SDE-24) MCP tool surface (`search`, `ingest`,
  `list_sources`, `get_chunk_neighbors`). Unblocked.
