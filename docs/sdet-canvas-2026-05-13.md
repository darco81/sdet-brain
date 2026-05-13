# sdet-canvas — post-v0.4.0 polish digest (2026-05-13)

A digest for the brain so cross-repo queries about Canvas surface the
current state. Canvas is at v0.4.0 + ~10 polish commits.

## Where Canvas sits today

Canvas is a local-first Electron + React 19 design tool — open-source
alternative to Anthropic Claude Design — with a multi-runtime backend.
v0.4 pivoted from "single self-contained HTML doc generator" to a
real prototype workbench:

- **Multi-file workspace** — model emits separate HTML pages + shared
  CSS/JS via `===FILE: name===` / `===END FILE===` markers. Iframe
  router intercepts `<a href="other.html">` clicks for in-canvas nav
  without re-streaming.
- **Design council ritual** — 5- or 11-seat parallel role debate
  fires before code. Roster picker per provider (`none` / `design` /
  `full`). Full council is the default for Claude Code + OpenCode
  subscriptions.
- **Stage timeline + chat narrative** — `[STAGE: name]` markers parsed
  live; prose between FILE blocks streams into a chat-panel transcript.
- **Subscription routes** — Claude Code adapter + OpenCode adapter
  (GPT-5.x Codex family) both work via subscription, no per-token
  costs.

## Multi-runtime backends

1. **Anthropic API** (BYOK; keytar-backed key store)
2. **Claude Code** (subscription via spawned `claude` child process)
3. **OpenCode** (subscription via spawned `opencode` CLI; 60+ models
   under one auth)
4. **OpenRouter** (BYOK aggregator; 200+ models; free-form slug input
   with shape validation)
5. **MLX-local** (Apple Silicon — LM Studio / mlx-omni / mlx_lm.server
   detected on configurable endpoint)
6. **Ollama** (fully offline; live model list from /api/tags)

## Polish wave shipped since v0.4.0 (2026-05-10 → 2026-05-13)

- Live AI narrative streams into a Chat panel (separate from iframe)
  with auto-scroll + scroll-position preservation. Chat panel reads
  like a transcript: user prompt → council debate → stage timeline +
  commentary → "✓ rendered · N tok" footer.
- Foundation-first builds no longer dump raw CSS as plaintext into
  the iframe. New build-progress placeholder lists files-in-flight
  with byte counters until the first HTML lands. VfsParser now
  refuses to activate non-HTML files.
- Inspector Code tab auto-scrolls to streaming tail. Toolbar wraps
  when narrow instead of clipping. Live tweaks (Colors / Typography
  / Layout sliders) survive iframe rebuilds during streaming via a
  `<style id="sdet-canvas-live-tweaks">` block injected into every
  effective srcdoc.
- Default e-commerce starter prompt rewritten — intent + acceptance
  criteria, NOT architectural cookbook. Frontier models architect the
  JS fine; the cookbook just stole tokens. Final
  `[STAGE: Acceptance audit]` step makes the model self-check
  dead links + state round-trips before declaring done.
- GenerateWizard auto-opens once per fresh project. Apply enriches +
  fires; Skip / Esc fires generate with original prompt.
- "Reset prompt to default" affordance (chip + link). `DEFAULT_PROMPT`
  hoisted into `src/renderer/lib/default-prompt.ts` as shared source
  of truth.
- Modal a11y (`role="dialog"` + `aria-modal` + Escape handler) on
  GenerateWizard / PresetBrowser / StarterTemplates. Defensive iframe
  source guard on `window.message` handlers.

## Parked / not yet built

- Auto-update re-enable — blocked on Apple Dev cert ($99/y) or
  making the repo public. Product decision.
- App.tsx / Inspector.tsx file splits — large but working; DX-only.
- E2E happy-path Playwright test.
- Provider health pill (Ollama daemon, MLX endpoint, key valid).
- OpenRouter model list TTL cache.

## Knowledge graph hooks

- Repository: `/Users/dariusz/dev/darco81/sdet-canvas`
- Mirror (legacy): `/Users/dariusz/dev/dar-kow/sdet-canvas` (canonical
  moved to darco81 path; GitHub redirect handles the URL alias)
- Linear epic: SDE-245 (v0.4 multi-file + stages + council)
- Sub-issues: SDE-246..SDE-252 (all closed)
- Recent commits: `0b04d1c` backlog sweep, `8015226` audit wave 1
