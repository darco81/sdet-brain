# Instructions for Claude (Desktop + Code) — using the sdet-brain MCP

> Keep this as project-level `CLAUDE.md` for Claude Code, or drop it into
> Claude Desktop's custom-instructions field — both clients respect it.

You have access to the **sdet-brain** MCP server, a persistent RAG over a
personal Markdown corpus. Think of it as long-term memory across
conversations.

## When to use sdet-brain

**Use it BEFORE answering** any question that touches:

- Past decisions, sprints, or work history ("what did I decide about X",
  "what shipped last sprint")
- Brand voice / writing samples ("how do I usually phrase X")
- Ongoing projects by name (`sdet-brain`, `wcag-toolkit`, `jarvis`,
  `sdet-canvas`, `cdat-pattern`, etc.)
- Technical decisions, architecture choices, trade-offs, lessons learned

**Don't use it for**: generic programming questions, real-time facts (search
the web), live system state (use shell), or anything not in the corpus.

## Available tools (prefix `sdet-brain__`)

Core (read-only): **`search`** (hybrid semantic + BM25, the default),
**`list_sources`**, **`get_chunk_neighbors`**. Ingest (mutating):
**`ingest_path`**, **`ingest_image`**. Domain helpers (pre-baked filters):
**`list_articles_by_status`**, **`search_voice_samples`**,
**`search_smaczki`**, **`search_decisions`**, **`search_sprint_reports`**.
LLM-backed (local MLX): **`multi_query_search`**, **`query_rewrite`**,
**`summarize_results`**. Plus **`ping`**.

## Default search recipe

1. Call `search` with the user's query as-is, top 5.
2. Read the `text` snippets. If the answer is there → quote with a
   `[source: file_path.md]` citation.
3. If a hit is mid-thought → `get_chunk_neighbors` on its `id` for ±1 chunk.
4. Synthesize in the brand voice; cite sources.

## Honest signals

- Weak results (top score low) → say so; don't fabricate citations.
- `qdrant_ok=false` / `embedder_ok=false` on `/health` → the brain is down;
  fall back to general knowledge and warn.
- Always cite source files when quoting from the corpus.

## Server

- Server: `http://localhost:8080` (MLX bge-style embeddings + Qdrant +
  fastembed reranker).
- Qdrant: `http://localhost:6333` (Docker, persistent in
  `docker/qdrant_storage/`).
- Reingest the corpus: `uv run sdet-brain-ingest <path>` (or the watcher).
