# Daily automation

Optional macOS automation that wraps the brain in a launchd job:
re-index your corpus, generate an LLM digest, post to Discord, and
ping you with a banner if anything is broken. None of this is
required to use the brain - it just makes the daily flow hands-off.

The scripts are intentionally simple. If you want a different
scheduler (cron, systemd, GitHub Actions on a self-hosted runner),
the same three commands work standalone.

## What you get

- `daily.sh` - sequencer: ingest every configured corpus path,
  trigger `digest.py`, then call `healthcheck.sh`. Two macOS banners
  per run (ingest delta, OK/ALERT). Idempotent: ingest skips files
  whose `content_hash` hasn't changed.
- `digest.py` - pulls recent chunks from Qdrant (default
  `source_type=drafts`, last 24h) and asks the local MLX
  Qwen3-Next-80B-Instruct for a 3-5 bullet summary with `[n]`
  citations. Writes a markdown changelog to
  `~/Documents/sdet-digests/YYYY-MM-DD.md` and POSTs an embed to
  Discord when `DISCORD_WEBHOOK_URL` is set. Cold-starts MLX in
  process so the 80B weights leave RAM on exit.
- `healthcheck.sh` - pings Qdrant `/readyz`, the server's `/health`,
  and verifies the server daemon is loaded in launchd. No LLM, no
  ingest. Banner shows current chunk count + embedder.
- `examples/` - config templates and a launchd plist with `__PLACEHOLDERS__`.

## Setup

```bash
# 1. Configuration - paths and (optional) Discord webhook
mkdir -p ~/.config/sdet-brain
cp scripts/examples/paths.env.example   ~/.config/sdet-brain/paths.env
cp scripts/examples/discord.env.example ~/.config/sdet-brain/discord.env  # optional
chmod 600 ~/.config/sdet-brain/*.env
$EDITOR ~/.config/sdet-brain/paths.env  # set your real corpus paths

# 2. Smoke-test by hand (cold MLX, 30-90s for the 80B model)
SDET_BRAIN_DIR=$(pwd) ./scripts/daily.sh

# 3. Install the LaunchAgent (Mon-Fri 7:30 by default)
sed -e "s|__SDET_BRAIN_DIR__|$(pwd)|g" \
    -e "s|__LABEL__|com.$USER.sdet-brain-daily|g" \
    -e "s|__DAEMON_LABEL__|com.$USER.sdet-brain|g" \
    -e "s|__YOU__|$USER|g" \
    scripts/examples/com.example.sdet-brain-daily.plist.example \
    > ~/Library/LaunchAgents/com.$USER.sdet-brain-daily.plist

plutil -lint ~/Library/LaunchAgents/com.$USER.sdet-brain-daily.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.$USER.sdet-brain-daily.plist
```

## Configuration reference

All overridable via env (set in your shell, in `paths.env`, or in the
plist's `EnvironmentVariables`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `SDET_BRAIN_DIR` | `~/dev/sdet-brain` | Repo checkout (where `pyproject.toml` lives) |
| `SDET_BRAIN_CONFIG_DIR` | `~/.config/sdet-brain` | Where `paths.env` and `discord.env` are sourced from |
| `SDET_BRAIN_DAEMON_LABEL` | `com.local.sdet-brain` | launchd label of your `sdet-brain-server` agent |
| `SDET_BRAIN_SERVER_URL` | `http://localhost:8080` | Override if you run on a non-default port |
| `QDRANT_URL` | `http://localhost:6333` | |
| `UV_BIN` | `/opt/homebrew/bin/uv` | Falls back to `which uv` if missing |
| `DRAFTS_PATHS` etc. | _empty_ | Comma-separated absolute dirs per `source_type` (sourced from `paths.env`) |
| `DISCORD_WEBHOOK_URL` | _empty_ | When set, `digest.py` POSTs an embed |
| `SDET_DIGEST_DIR` | `~/Documents/sdet-digests` | Where digest changelogs land |
| `SDET_DIGEST_WINDOW_HOURS` | `24` | "Last X hours" cutoff for chunk inclusion |
| `SDET_DIGEST_MAX_PASSAGES` | `12` | Max chunks fed to the LLM per run |
| `SDET_DIGEST_SOURCE_TYPE` | `drafts` | Filter by `source_type` payload field |
| `SDET_DIGEST_LANG` | `pl` | `pl` or anything else (English fallback) |

## Common ops

```bash
# Manual run (banners + log entries)
./scripts/daily.sh

# Just the health probe - no ingest, no LLM
./scripts/healthcheck.sh

# Just the digest - useful for re-running today's
DISCORD_WEBHOOK_URL=... uv run ./scripts/digest.py

# Force the schedule to fire now (treat as if it's 7:30)
launchctl kickstart gui/$(id -u) com.$USER.sdet-brain-daily

# Tail recent activity
tail -f /tmp/sdet-brain-daily.log
tail -f /tmp/sdet-brain-digest.log
tail -f /tmp/sdet-brain-health.log

# Disable temporarily
launchctl bootout gui/$(id -u)/com.$USER.sdet-brain-daily
```

## Caveats

- **MLX cold start.** First call to the 80B Qwen-Next-Instruct after
  boot pays ~30-60s of weight loading, plus inference. The whole
  digest typically lands in under 90s end-to-end. Subsequent runs in
  the same process would be warm - but `digest.py` is intentionally
  one-shot, so the weights leave RAM on exit (the daemon never holds
  them).
- **Memory pressure.** On 64 GB unified memory, running the digest
  pushes free RAM near zero while the model is resident. macOS
  swaps. If your corpus contains very large markdown files (~30 KB+
  each, e.g. design tokens dumps), you may see the embedder balloon
  too - split or exclude those paths until the issue is tracked.
- **Discord and Cloudflare.** Discord webhooks live behind
  Cloudflare, which 1010s Python's default `User-Agent`. `digest.py`
  sends `User-Agent: sdet-brain-digest/...` to bypass it. If you
  swap in another HTTP client, keep a custom UA.
- **`source_type` payload.** Chunks indexed before you set the
  `*_PATHS` env vars get tagged `unknown`. The digest filters on
  `source_type`, so retroactively classifying old chunks needs a
  Qdrant payload-update (not implemented here - it's a one-liner
  with `points/payload`).
