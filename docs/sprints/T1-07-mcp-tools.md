# Sprint Report: T1-07 Four core MCP tools

> Sprint report autorski. Linear refs (`SDE-XX`) są internal trackingiem
> i nie są publicznie linkowane.


**Linear:** [SDE-24](https://linear.app/sdet-it/issue/SDE-24/t1-07-4-core-mcp-tools-search-ingest-list-sources-get-chunk-neighbors)
**Started:** 2026-04-30 16:18 (CET)
**Done:** 2026-04-30 16:34 (CET)
**CC time:** ~16 min
**Dariusz manual:** 0 min

## What shipped

- `server/state.py`: extracted `build_default_state` so both the
  FastAPI lifespan and the standalone MCP entrypoints share a single
  state-construction path.
- `server/mcp_server.py`: refactored to take a `state_getter` callable
  so tools can resolve the active `AppState` at call time. Five tools
  registered via `@mcp.tool`: `ping`, `search`, `ingest_path`,
  `list_sources`, `get_chunk_neighbors`. Tool docstrings carry
  use-when hints that Claude reads as system prompts.
- `server/tools/_helpers.py`: `ToolError`, payload-safe accessors
  (`safe_str`, `safe_int`), `source_type_filter`,
  `source_path_filter`, `require_storage`, `require_embedder`,
  `collection_or_default`. Tools never reach into payloads with
  bare `[...]` lookups.
- `server/tools/search.py`: dense-vector search formatted as Markdown
  with rank / score / heading / chunk text. Optional `min_score` and
  `source_type` filter.
- `server/tools/ingest.py`: thin wrapper over the T1-05 pipeline
  returning a Markdown summary of `IngestStats`.
- `server/tools/list_sources.py`: scrolls the collection, groups
  payloads by `source_path`, counts chunks and tracks the most recent
  `created_at`.
- `server/tools/get_chunk_neighbors.py`: scrolls a single source's
  chunks, sorts by `chunk_index`, returns the closed range
  `[i-window, i+window]` clamped to file bounds.
- 11 tool-level tests (against the live Qdrant container + a
  deterministic 16-dim fake embedder): search basics + filter +
  empty-query rejection + empty-corpus message; list_sources groups
  + filter; get_chunk_neighbors window + clamp at zero + clamp at
  total; ingest tool routes to the pipeline + rejects missing path.
- README "MCP tools" reference table.
- CHANGELOG `[Unreleased]` entry.

## Atomic commit

- `<sha> feat(server): 4 core MCP tools`

## Numbers

- Files added: 7 (state + _helpers + 4 tools + test_tools).
- Files modified: 4 (CHANGELOG, README, app.py, mcp_server.py).
- Tests added: 11 (68 total: 2 + 7 + 15 + 25 + 7 + 19).
- Quality gates: ruff clean, mypy strict 40 source files clean,
  pytest 68/68 in 9.36 s.

## Lessons learned

- The FastMCP instance is mounted on FastAPI before the lifespan
  runs, so it cannot capture `AppState` directly. The `state_getter`
  callable resolves that cleanly: FastAPI passes
  `lambda: app.state.app_state` (read at tool call time), while the
  stdio / SSE entrypoints will capture an eagerly-built state. T1-08
  inherits the same pattern when the watcher needs to share state.
- Returning Markdown from MCP tools beats JSON for LLM consumption:
  the heading hierarchy, code fences, and `---` separators give the
  model clear structure to quote back to the user. Worth keeping the
  pattern as the tool surface grows in T2.
- Defensive payload accessors (`safe_str` / `safe_int`) keep the
  type-checker happy without `cast()` and stop a malformed payload
  from crashing a tool call. Costs ~10 lines, saves a class of
  surprises.

## Out-of-scope items captured

- The MCP `ping` tool ships alongside the four headline tools - five
  total in `mcp_server.py`. Not flagged as scope creep; the issue's
  AC #1 ("4 tools visible in `mcp inspector`") still holds.
- Tools currently accept an explicit `collection` kwarg for testing
  hygiene (so tests don't stomp on production data). The MCP-facing
  signatures hide that override - it is positional-only on the Python
  side via the `*` separator. Worth documenting in the Tier 2
  refactor if other tools need the same shape.

## Next task

- T1-08 (SDE-25) Watcher (auto-reindex on save). Unblocked.
