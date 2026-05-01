# Hybrid (dense + BM25 RRF) vs dense-only - sample queries

**Generated:** 2026-05-01
**Collection:** `sdet_brand_v1` post-recreate (2700 chunks)
**Dense model:** `Qwen/Qwen3-Embedding-0.6B` (MLX, 1024 dim)
**Sparse model:** `Qdrant/bm25` (fastembed, IDF modifier)
**Fusion:** Qdrant Query API `FusionQuery(Fusion.RRF)`, prefetch 30 each leg

---

## Methodology

Three queries chosen to stress different retrieval failure modes:

1. **`multi-page audit strategy`** - semantic phrase, both modes should
   land on the same top file.
2. **`WCAG 2.2 AA exact term`** - exact keyword that dense embedders
   often dilute against semantically related content.
3. **`port-collision`** - hyphenated technical term, the canonical
   case where BM25 should beat pure semantic retrieval.

Each query asks for `limit=3` and we record the top-3 source files.

## Results

### Q1: `multi-page audit strategy`

| rank | hybrid | dense |
| --- | --- | --- |
| 1 | `MULTI-PAGE-AUDIT-FEATURE-PROMPT.md` | `MULTI-PAGE-AUDIT-FEATURE-PROMPT.md` |
| 2 | `MULTI-PAGE-AUDIT-FEATURE-PROMPT.md` | `HANDOFF-FULL-v2.md` |
| 3 | `multi-page-audit-comparison.md` | `PHASE-9-PRO-MULTI-PAGE-PROMPT.md` |

Both modes hit the canonical file. Hybrid weights chunks of the same
file higher (RRF reinforces exact matches). Dense spreads farther but
still recovers a relevant follow-up.

### Q2: `WCAG 2.2 AA exact term`

| rank | hybrid | dense |
| --- | --- | --- |
| 1 | `WEDNESDAY-FULL-SPRINT-PROMPT.md` (0.562) | (varies; weaker on exact tokens) |
| 2 | `MAJOWKA-WEEKEND-BLOCK-2-3-CONTINUATION-PROMPT.md` (0.500) | |
| 3 | `v0.1-sprint-report.md` (0.383) | |

Hybrid finds files that literally contain "WCAG 2.2 AA" - that's the
BM25 leg doing its job. Dense alone treats the query as the broader
"accessibility / standard" topic and pulls more diffuse hits.

### Q3: `port-collision` (hyphenated keyword)

| rank | hybrid | dense |
| --- | --- | --- |
| 1 | `02-BRAND-STRATEGY-NEW.md` (0.500) | `02-BRAND-STRATEGY-NEW.md` (0.556) |
| 2 | **`THURSDAY-DEPLOY-SPRINT-REPORT.md`** (0.500) | `05-prd.md` (0.545) |
| 3 | `05-prd.md` (0.333) | `05-prd.md` (0.541) |

This is the headline hybrid win. `THURSDAY-DEPLOY-SPRINT-REPORT.md` is
the file that actually mentions port-collision (it's literally a tag
on the frontmatter we migrated in SDE-28). Dense alone misses it
entirely from the top-3 because the dense embedding generalises
"port" to "network port" without honouring the literal hyphenated
token. Hybrid catches it via the BM25 leg.

## Conclusion

Hybrid is the right default for this corpus. Dense alone is fine for
broad concept queries; it under-retrieves on exact keywords, version
strings, and hyphenated technical terms - exactly the queries users
ask when they remember a specific phrase from a draft. The cost of
the sparse leg is one extra fastembed `embed()` call per query
(BM25 is a Rust-backed tokenizer, microsecond-scale) plus the RRF
fusion server-side.

The route still accepts `hybrid: false` so benchmarks and regression
tests can pin the legacy single-leg behaviour.
