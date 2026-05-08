# Sprint Report: T1-02 Qdrant Docker setup + collection scaffolding

> Sprint report autorski. Linear refs (`SDE-XX`) są internal trackingiem
> i nie są publicznie linkowane.


**Linear:** [SDE-19](https://linear.app/sdet-it/issue/SDE-19/t1-02-qdrant-docker-setup-collection-scaffolding)
**Started:** 2026-04-30 13:18 (CET)
**Blocked:** 2026-04-30 13:55 (CET) - Docker registry pulls hung indefinitely
**Resumed:** 2026-04-30 15:22 (CET) - after Docker Desktop restart
**Done:** 2026-04-30 15:28 (CET)
**CC time:** ~43 min (most of the gap was waiting on a frozen `docker pull`)
**Dariusz manual:** ~2 min (`osascript quit app "Docker"` + `open -a Docker` sequence)

## What shipped

- `docker/docker-compose.yml`: `qdrant/qdrant:latest`, bind mount
  `./qdrant_storage`, `/readyz` TCP probe healthcheck (5 s interval, 5 s
  start-period, 5 retries), bridge network `sdet-brain-network`, ports
  6333 + 6334 published.
- `src/sdet_brain/storage/qdrant_client.py`: typed `QdrantStorage` facade
  - `ensure_collection`, `ensure_payload_indexes` (idempotent, swallows
    409 on duplicate index), `upsert_points`, `search` via the
    post-1.10 `query_points` API, `delete_by_filter`, `count`,
    `get_collection`, `status` (returns flat `CollectionStatus`
    dataclass with name / vector_size / distance / points_count),
    `list_collections`, context manager + explicit `close`.
- `src/sdet_brain/storage/collections.py`: `COLLECTION_NAME =
  "sdet_brand_v1"` literal type, `ChunkPayload` `TypedDict` documenting
  the chunk-payload schema (source_path, source_type, chunk_index,
  total_chunks, content_hash, created_at, frontmatter dict),
  `PAYLOAD_INDEXES` dict (KEYWORD on source_type / source_path /
  content_hash, INTEGER on chunk_index), `init_collections(storage,
  vector_size, *, name=COLLECTION_NAME)` - the `name` override lets
  tests use disposable collections without nuking production data.
- `src/sdet_brain/cli/qdrant_cli.py`: `argparse` CLI with `init`,
  `status`, `ping` subcommands. Dispatched via `_HANDLERS` dict for
  exhaustiveness. Module-runnable plus `sdet-brain-qdrant` console
  script wired in `pyproject.toml`.
- `tests/storage/conftest.py`: real-Qdrant fixture; skips with reason if
  `/readyz` is unreachable. No mocks per project decision.
- `tests/storage/test_qdrant_client.py`: 5 cases (parametrised count
  expands to 7 invocations) - idempotent `ensure_collection`, upsert +
  search round-trip with deterministic 384-dim vectors, filter-based
  deletion removes only matching points, `init_collections` registers
  payload indexes on a disposable collection name, parametrised batch
  sizes (0 / 1 / 7).
- `CHANGELOG.md`: new `[Unreleased]` section enumerating the storage
  additions.

## Atomic commits

- `86a7e49 feat(storage): qdrant client + collection scaffolding`

(WIP branch `wip/t1-02-qdrant` retained for the audit trail; commits on
it are no longer needed - feel free to `git branch -D wip/t1-02-qdrant`
once you've reviewed.)

## Numbers

- Files added: 6
- Files modified: 3 (CHANGELOG, pyproject, docker-compose)
- Lines added: ~605 (incl. tests + CHANGELOG)
- Lines removed: ~11
- Tests added: 7 (4 cases + 3 parametrised)
- Quality gates: `ruff check src tests` -> 0 issues; `mypy --strict src`
  -> no issues found in 12 source files; `pytest -q` -> 9 passed.

## Runtime verification

- `docker compose -f docker/docker-compose.yml up -d qdrant` -> container
  up, `/readyz` returns "all shards are ready".
- `python -m sdet_brain.cli.qdrant_cli ping` -> reachable, empty
  collections.
- `python -m sdet_brain.cli.qdrant_cli init` (1st run) -> "Created
  collection sdet_brand_v1 (vector_size=1024)".
- `python -m sdet_brain.cli.qdrant_cli init` (2nd run) -> "Collection
  sdet_brand_v1 already exists - no-op."
- `python -m sdet_brain.cli.qdrant_cli status` -> name, vector_size
  1024, distance Cosine, points_count 0.
- Persistence: inserted a 1024-dim point, `docker compose restart
  qdrant`, status still showed `points_count: 1`.

## Lessons learned

- Docker Desktop on macOS can wedge image pulls without surfacing any
  error - even `hello-world:latest` hung. The fix was a hard restart
  (`osascript quit app "Docker"` + `open -a Docker`). Worth a note in a
  future runbook: if a pull stalls > 30 s with zero progress lines,
  restart Docker Desktop before debugging anything else.
- `qdrant-client` 1.17 has soft-deprecated `search()` - the wrapper
  uses `query_points()` and unwraps `QueryResponse.points`. Same return
  shape, future-proof.
- Initially the `init_collections` test wiped the production
  `sdet_brand_v1` collection in its teardown. Refactored
  `init_collections` to accept a `name` override so tests use a
  per-process disposable name. The production constant stays the
  default and the public API is unchanged.

## Out-of-scope items captured

None this round.

## Next task

- T1-03 (SDE-20) Embeddings - MLX primary, Gemini fallback. Unblocked
  by this storage layer.
