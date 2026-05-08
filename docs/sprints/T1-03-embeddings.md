# Sprint Report: T1-03 Embedding service - MLX primary + Gemini fallback

> Sprint report autorski. Linear refs (`SDE-XX`) są internal trackingiem
> i nie są publicznie linkowane.


**Linear:** [SDE-20](https://linear.app/sdet-it/issue/SDE-20/t1-03-embedding-service-mlx-primary-gemini-fallback)
**Started:** 2026-04-30 15:28 (CET)
**Done:** 2026-04-30 15:38 (CET)
**CC time:** ~10 min
**Dariusz manual:** 0 min

## What shipped

- `src/sdet_brain/embeddings/protocol.py`: runtime-checkable
  `IEmbedder` `Protocol` (`vector_size`, `model_name`, `embed`,
  `health_check`) + `EmbeddingError`.
- `src/sdet_brain/embeddings/mlx_provider.py`: `MLXEmbedder` with
  thread-safe lazy load, batch of 32, dimension self-correction (warns
  if model output disagrees with the configured `vector_size`).
  Embeddings come from `BaseModelOutput.text_embeds` - the helper
  raises a clear `EmbeddingError` if a future model lacks that
  attribute rather than silently mean-pooling and producing a
  different vector space.
- `src/sdet_brain/embeddings/gemini_provider.py`: `GeminiEmbedder`
  built on the new `google-genai` SDK (`Client.models.embed_content`),
  classifies 429/503/timeout strings as `GeminiTransientError` and
  retries them with `tenacity.Retrying` (4 attempts, 0.5-8 s
  exponential backoff). Permanent errors surface as `EmbeddingError`.
- `src/sdet_brain/embeddings/factory.py`: `get_embedder(settings) ->
  EmbedderSelection`. Reads `EMBEDDING_PROVIDER`, attempts the primary,
  falls back when the primary either cannot be initialised
  (e.g. missing API key) or fails its `health_check()`. Records the
  full chain of attempted providers in the result.
- `src/sdet_brain/cli/embed_cli.py`: `encode` (head/tail or `--full`
  output) and `health` subcommands, dispatched via `_HANDLERS`. Wired
  as `sdet-brain-embed` console script.
- 15 tests: protocol contract, factory fallback against in-process
  fakes (4 cases), Gemini retry/permanent-error/health (7 cases),
  MLX lazy-load + empty-input (2 cases, gated to `darwin/arm64`).
- README "Embeddings" section, CHANGELOG entry under `[Unreleased]`.

## Atomic commit

- `<sha> feat(embeddings): MLX primary + Gemini fallback dual-path`

## Numbers

- Files added: 9
- Files modified: 4 (CHANGELOG, README, pyproject.toml, uv.lock)
- Tests added: 15
- Quality gates: ruff clean, mypy strict 17 source files clean,
  pytest 24/24 (2 T1-01 + 7 T1-02 + 15 T1-03).

## Runtime verification

- `sdet-brain-embed health` -> primary=mlx, active=mlx, fell_back=False,
  attempted=mlx, model=`Qwen/Qwen3-Embedding-0.6B`, vector_size=1024.
- `sdet-brain-embed encode "Hello SDET Brain"` -> 1024-dim vector
  preview, model loaded lazily on first call.

## Lessons learned

- `mlx-embeddings.generate()` does NOT return `mx.array` despite the
  docstring - it returns a transformers-style `BaseModelOutput` with
  `text_embeds`, `last_hidden_state`, and (when set) `pooler_output`.
  Hard-coding `text_embeds` keeps the dense vector space stable; if a
  future model swaps to `pooler_output` we'll catch it via the explicit
  error rather than silent drift.
- `tenacity.retry` is a decorator class - not directly typeable as a
  return value. Switched to `Retrying()` instances + `for attempt in
  retrying: with attempt:` so mypy and ruff can see clean types.
- `mlx-embeddings` and `mlx.*` ship without `py.typed`. Added them to
  the mypy override list alongside `frontmatter`/`watchdog`. The
  "unused section" note remains a nuisance until at least one of those
  modules is actually imported under strict mode - happy to live with
  it for one more task.

## Out-of-scope items captured

None.

## Next task

- T1-04 (SDE-21) Markdown parser + chunker. Unblocked.
