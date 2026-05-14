---
category: reference
series: sdet-brain
status: living
tags: [cross-platform, windows, fork, sync]
created_at: 2026-05-14
---

# Cross-platform sync notes (Mac flagship → Windows fork)

Mac is the flagship. The Windows fork lives at
[`darco81/sdet-brain-win`](https://github.com/darco81/sdet-brain-win)
and trails by one minor — features ship there a few days after the
Mac release.

This doc captures everything that has to be different between the
two repos so a one-shot sync diff stays small.

## Deps that don't exist on Windows

Stripped via PEP 508 markers (`sys_platform == 'darwin'`):

- `mlx-embeddings` — Apple Neural Engine embedding runtime.
- `mlx-vlm` — Apple Neural Engine vision-language runtime (v0.6.0+).

The Win bootstrap (`bootstrap.ps1`) installs the rest of the deps
plus `pypdfium2` and `pillow-heif` (cross-platform).

## Settings overrides

The Win fork ships an `.env.example` with these defaults:

```ini
# Force Ollama-only OCR — no MLX-VLM on Win.
OCR_PROVIDER=ollama

# Drop the heavyweight tertiary — 4 GB VRAM cannot fit qwen2.5-vl:32b.
OCR_OLLAMA_FALLBACK_MODEL=

# Embeddings stay on Gemini (or a lightweight Ollama embedding model
# in 0.3.0-win+).
EMBEDDING_PROVIDER=gemini
```

The factory chain on Win therefore degenerates to a single link:
`ollama + deepseek-ocr`. Health-check failure → no fallback →
`OCRError` propagates to the caller, which surfaces in
`stats.errors` rather than corrupting the index.

## Tests that skip on Win

Anything that requires `mlx_vlm` or `mlx_embeddings` should
auto-skip via:

```python
pytest.importorskip("mlx_vlm")
```

The factory tests stay green because they monkeypatch `_BUILDERS`
with in-process stubs (no real MLX). The provider tests for
`MLXVLMOCREngine` need the import-skip guard.

## Encoding gotchas (per memory `feedback_windows_python_stdout_cp1252`)

Any Python MCP stdio entrypoint on Windows MUST reconfigure UTF-8 or
set `PYTHONIOENCODING=utf-8` — otherwise non-ASCII payloads come back
mojibake. Pin this at the top of `__main__.py` for the stdio CLI:

```python
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
```

Or via env-var in the launcher script. The PowerShell debugging
trap (`Invoke-RestMethod | ConvertTo-Json` mangles UTF-8) is
documented in memory `feedback_powershell_restmethod_codepage` —
use `Invoke-WebRequest -OutFile` + raw bytes for round-trips.

## What needs hand-merge (not just sync)

- `pyproject.toml` — version stays at `0.X.0-win.N`, deps drop the
  darwin-marked entries.
- `README.md` — Win-specific install (PowerShell, Scoop / winget),
  no MLX section, dual-GPU note for v0.3.0-win+.
- `CHANGELOG.md` — independent versioning (`0.1.0-win.0`,
  `0.2.0-win.0`, ...).
- `scripts/` — `daily.sh` doesn't apply; ship a `Daily.ps1` Task
  Scheduler equivalent if needed.

## Sync workflow

When prepping a Win release from a Mac feature branch:

1. Cherry-pick or rebase the feature commits onto `win/main`.
2. Run `python scripts/strip_mac_deps.py pyproject.toml` (TODO —
   currently manual delete of the two `mlx-*` lines).
3. Patch `pillow-heif` and `pypdfium2` versions to match Win wheels
   on PyPI (they ship for `win_amd64`).
4. Run the full test suite; skipped MLX tests should be the only
   change vs Mac.
5. Tag `0.X.0-win.N` and push.

## Future cross-platform improvements

- **dual-GPU CUDA** on Win — reranker offload to secondary card,
  frees primary for OCR + bge-m3 coexistence (target: `0.3.0-win`).
- **Linux flagship** — currently both forks assume desktop GPUs;
  a headless VPS variant would need to drop MLX entirely and stand
  on Gemini + Ollama-remote (separate fork or feature flags).
