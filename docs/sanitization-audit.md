# Sanitization audit - v0.5.1 public-ready flip (mode 1: defensive)

Generated: 2026-05-08
Scope: enumerate all sensitive content that must be REMOVED, REPLACED, or
explicitly KEPT before flipping repo visibility from private to public.

Mode 1 principle: **patterns and architecture stay; concrete brand
content disappears or is replaced by generic placeholders**.

## 1. Hardcoded `/Users/dariusz/...` paths

| File | Line(s) | Action | Replacement |
| --- | --- | --- | --- |
| `src/sdet_brain/cli/ingest_cli.py` | 38-49 | REPLACE | empty defaults - paths come from env vars only; remove "Dariusz's setup" comment |
| `docker/docker-compose.yml` | 66 | REPLACE | `${SDET_BRAIN_CORPUS_HOST:-/path/to/your/markdown/corpus}` |
| `docs/sprints/v0.1.0-initial-ingest-snapshot.md` | 23-25, 48-49, 61-65 | REPLACE | generic `<your-corpus-path>` + add disclaimer header (Phase 3) |
| `docs/sprints/v0.3.0-tier-2-3-overnight-sprint.md` | 165 | REPLACE | `cd <repo>` |
| `docs/sprints/v0.5.0-tier-5-dx-sprint.md` | 134 | REPLACE | `cd <repo>` |
| `migrations/20260501-065259-frontmatter-migration-apply.md` | full file (88-row table) | REPLACE | collapse to anonymized summary (count + categories breakdown, no filenames) |
| `.claude/settings.local.json` | 4-6 | KEEP | not git-tracked (`.claude/` not in `git ls-files`); user-local settings |

## 2. Brand corpus references (`sdet-brand-drafts`, `sdet-brand-strategy`, `portfolio-v2`)

| File | Line(s) | Action | Replacement |
| --- | --- | --- | --- |
| `README.md` | 470, 556, 558 | REPLACE | generic `your-corpus/` references; remove `SDET-BRAIN-*-PROMPT.md` pointers (these were brand-private prompts) |
| `docker/docker-compose.yml` | 66 | REPLACE | covered above (mount default) |
| `tests/server/test_domain_tools.py` | 317, 336, 337, 343 | REPLACE | `series="portfolio-v2"` → `series="case-study-01"`; `/sr/portfolio.md` → `/sr/example.md` |
| `tests/ingestion/test_frontmatter_classifier.py` | 21, 31, 76, 78-79 | REPLACE | keep `SDET-BRAND-STRATEGY.md` test case (it tests a generic naming pattern that the classifier matches - acceptable as architecture); replace `portfolio-v2-*.md` filenames with `my-project-*.md`; keep the assertion structure |
| `migrations/20260501-...md` | full file | REPLACE | covered above |
| `CHANGELOG.md` | 390 | KEEP | one historical mention, autorial changelog - acceptable per spec |
| `docs/sprints/v0.1.0-initial-ingest-snapshot.md` | 23-25 | REPLACE | corpus listing → generic counts + categories |

## 3. Linear refs (pattern `SDE-\d+`)

| File | Action | Notes |
| --- | --- | --- |
| `README.md` lines 499-518 | REMOVE | drop "Tracked in" column from tier table; drop SDE-XX from "Deferred / out-of-scope" bullets |
| `CHANGELOG.md` (many) | KEEP | autorial changelog - acceptable; SDE-XX referenced as historical work-tracking is fine |
| `docs/sprints/T1-*.md` | KEEP + DISCLAIMER | add header disclaimer per spec |
| `docs/sprints/v0.*.md` | KEEP + DISCLAIMER | add header disclaimer per spec |
| `tests/ingestion/test_chunker.py` line 148 | KEEP | `# --- SDE-40 small-tail merge ---` is an in-test code organizer; harmless |

## 4. "brand" references in source code

These are **architectural identity**, not corpus leakage. Per Mode 1
rule "patterns and architecture stay" - KEEP all of these:

- `src/sdet_brain/ingestion/frontmatter_schema.py` - `BrandFrontmatter`
  type, `parse_brand_frontmatter` function, "brand-strategy" category
  literal
- `src/sdet_brain/ingestion/frontmatter_classifier.py` - classifier
  for "brand corpus" docstrings
- `src/sdet_brain/storage/collections.py` - `COLLECTION_NAME =
  "sdet_brand_v1"` (Qdrant collection identifier)
- `src/sdet_brain/config.py` - `default="sdet_brand_v1"` config
- `src/sdet_brain/__init__.py` - package docstring "SDET brand domain"
- `src/sdet_brain/server/{app,mcp_server}.py` - server description
  strings
- `src/sdet_brain/server/tools/{summarize_results,query_rewrite,multi_query}.py`
  - "brand-aware" prompt strings (system-prompt character)
- `src/sdet_brain/server/chat/prompt_template.py` - "brand voice"
  Polish system prompt
- `src/sdet_brain/cli/migrate_frontmatter.py` - uses `parse_brand_frontmatter`
- `pyproject.toml` line 4 - package description "Persistent RAG for SDET
  brand domain"

Rationale: "brand" is a generic word here - describes the system's
character (single-user, voice/decisions/articles oriented), not a
specific company or proprietary corpus. The architecture is built
around personal-corpus RAG; renaming it would be cosmetic.

## 5. .env.example

KEEP as-is. All values are placeholders or env-var keys. `WATCH_PATHS=`
empty default. Brand corpus path env vars (`PROJECT_KNOWLEDGE_PATHS`,
etc.) are empty strings. Comment about "Dariusz's setup" on lines 41-44
should be REPLACED with generic phrasing.

## 6. Secrets / snapshots / local state

| Item | Status | Notes |
| --- | --- | --- |
| `.env`, `.env.*` | gitignored ✓ | confirmed in `.gitignore` |
| `qdrant_storage/`, `docker/qdrant_storage/` | gitignored ✓ | confirmed (local snapshot exists at `docker/qdrant_storage/` - not tracked) |
| `data/`, `logs/` | gitignored ✓ | not present anyway |
| `.cache/`, `models/` | gitignored ✓ | optional MLX model caches |
| `.remember/` | gitignored ✓ | local Claude Code session memory |
| `migrations/*-dry-run.md` | gitignored ✓ | only `*-apply.md` is committed |
| API keys in source | none found | greps clean |

## 7. Internal frontmatter taxonomy specifics

KEEP - taxonomy values (`voice-sample`, `smaczki`, `decision`,
`brand-strategy`, etc.) are architectural categories baked into the
schema. They describe the *shape* of a personal RAG corpus, not the
content. A new user can populate any of these categories with their
own material.

## 8. Examples / templates

`examples/templates/*.yaml` - KEEP. Generic enough:
- `decision-history.yaml` - generic decision search
- `series-status.yaml` - generic Series N status (no specific series
  named)
- `voice-check.yaml` - generic tone check
- `wcag-fact-check.yaml` - generic WCAG fact-check (the topic is
  publicly-relevant accessibility, not brand-private)

## Summary counts

- REPLACE: 7 files
- REMOVE (sections within files): 1 file (README tier table column)
- ADD disclaimer: 13 files in `docs/sprints/`
- KEEP (audited and confirmed safe): 21 source files, .env.example,
  CHANGELOG.md (with caveats), examples/templates/*.yaml

## Phase 2-5 atomic steps follow this audit
