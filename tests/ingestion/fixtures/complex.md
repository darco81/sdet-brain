---
title: Complex fixture
slug: complex-fixture
status: published
language: en
tags:
  - architecture
  - storage
  - embeddings
created_at: 2026-04-30
---

# Architecture overview

This fixture exercises the chunker against a realistic mid-sized
document. It carries a YAML header, several headings up to depth three,
two fenced code blocks, a markdown table, and enough prose to force at
least three or four chunks at the default 800-character target. The
purpose is to make sure heading paths, code-block atomicity, and
paragraph overlap all behave together when stressed.

## Storage layer

We use Qdrant 1.12 with cosine distance and a single collection
`sdet_brand_v1`. Payload indexes cover `source_type`, `source_path`,
`content_hash`, and `chunk_index`. The collection is created
idempotently by the application on startup so a fresh container only
needs `docker compose up -d qdrant`.

### Why Qdrant

Qdrant's filter-first architecture matches the brand corpus's
metadata-heavy access patterns. The Rust core gives us sub-10ms p99
latency at our scale, and the Apache 2.0 licence keeps deployment
options open.

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="sdet_brand_v1",
    vectors_config={"size": 1024, "distance": "Cosine"},
)
```

### Persistence

Local development uses a docker-compose bind mount under
`docker/qdrant_storage`. Production deploys (Tier 3) move this to a
named volume on the VPS so backups can be Time-Machine-style.

## Embeddings layer

Two interchangeable providers expose the same `IEmbedder` contract.
Switching between them only requires flipping `EMBEDDING_PROVIDER` in
the environment file.

| Provider | Vector size | When to use                  |
| -------- | ----------- | ---------------------------- |
| MLX      |        1024 | Apple Silicon dev box        |
| Gemini   |         768 | Cloud / VPS / laptop offline |

The factory falls back automatically when the primary provider's
health-check fails.

```python
from sdet_brain.embeddings.factory import get_embedder
from sdet_brain.config import get_settings

selection = get_embedder(get_settings())
print(selection.provider, selection.embedder.vector_size)
```

## Ingestion pipeline

The pipeline reads Markdown files, computes a SHA-256 content hash,
parses frontmatter, chunks the body at heading boundaries, embeds the
chunks in batches of 32, and upserts them into Qdrant. Re-ingestion is
idempotent: identical hashes are skipped, and changed files have their
old chunks removed by source-path filter before the new ones land.

## Watcher

A `watchdog` observer monitors the configured paths with a 300 ms
debounce so a single editor save does not enqueue three reindex jobs.
The worker thread drains the queue serially to keep memory predictable.
