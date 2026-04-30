# Sprint Report: T1-01 Bootstrap repo + project structure + Docker scaffolding

**Linear:** [SDE-18](https://linear.app/sdet-it/issue/SDE-18/t1-01-bootstrap-repo-project-structure-docker-scaffolding)
**Started:** 2026-04-30 13:12 (CET)
**Done:** 2026-04-30 13:18 (CET)
**CC time:** ~6 min
**Dariusz manual:** 0 min (gh already authenticated as `darco81`)

## What shipped

- Empty Python 3.12 project under `/Users/dariusz/dev/darco81/sdet-brain` with
  the package layout from the architecture doc (ingestion / embeddings /
  storage / server / cli + matching `tests/`).
- `pyproject.toml` declaring runtime deps (FastAPI, FastMCP 3.0, qdrant-client,
  pydantic-settings, watchdog, python-frontmatter, httpx, uvicorn) and dev
  tooling (pytest, mypy, ruff, types-pyyaml). Five `sdet-brain-*` console
  scripts wired through entry points.
- `src/sdet_brain/config.py` with a typed `Settings` class covering Qdrant,
  embedding providers (MLX + Gemini), server ports, and ingestion knobs.
- Docker scaffolding: `docker/docker-compose.yml` running Qdrant `v1.12.4`
  with a healthcheck and a `sdet-brain` service stub commented out until
  T1-06; multi-stage `docker/Dockerfile` (builder venv via uv, slim runtime
  with non-root user + healthcheck).
- `README.md` with a four-layer Mermaid architecture diagram (Clients,
  Server, Pipeline, Storage), stack table, and quick-start instructions.
- `.env.example` documenting every variable on `Settings`. `.gitignore`
  covering venvs, tooling caches, qdrant_storage, secrets, IDE files, and
  the local-only `.remember/` directory.
- Smoke test asserting package import + default settings load.
- `CHANGELOG.md` with `## [0.1.0] - 2026-04-30 - Initial bootstrap`.

## Atomic commits

- `<sha> feat: initial project bootstrap` (single commit per the issue's
  "Commit Message" section).

## Numbers

- Files added: 18 tracked (excludes `.remember/` and tooling caches).
- Files modified: 0 (greenfield).
- Lines added: ~520 (incl. README + CHANGELOG + uv.lock omitted from this
  count since lockfile is generated).
- Tests added: 2 (smoke).
- Quality gates: `ruff check src tests` -> 0 issues, `mypy --strict src` ->
  no issues found in 9 source files, `pytest -v` -> 2 passed.

## Lessons learned

- Pinning the Qdrant image tag (`v1.12.4`) avoids a silent breakage when
  `latest` rolls forward; same reason the Dockerfile pins the uv binary
  image to `0.5.0`.
- `pyproject.toml`'s mypy override block for `frontmatter` / `watchdog`
  triggers a "unused section" note today because those modules aren't
  imported yet. Harmless - the note disappears once T1-04 / T1-08 land.
- The `.remember/` directory existed before this commit (CC session
  memory). Added to `.gitignore` rather than removing it, since the
  hook expects it to be there.

## Out-of-scope items captured

None for this task. Scope matched the AC exactly.

## Next task

- T1-02 (SDE-19) Qdrant container + collection schema. Unblocked by this
  bootstrap commit.
