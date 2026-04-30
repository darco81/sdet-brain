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
