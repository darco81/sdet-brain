#!/bin/bash
# Daily routine: re-index corpus through MLX embeddings, run the
# local-LLM digest, then a health check. Wired into launchd via the
# `*-daily.plist` example. Idempotent: ingest skips files whose
# `content_hash` hasn't changed.
#
# Configuration is sourced from ${SDET_BRAIN_CONFIG_DIR:-$HOME/.config/sdet-brain}/:
#   - paths.env    DRAFTS_PATHS, ARTICLES_PATHS, PROJECT_KNOWLEDGE_PATHS,
#                  SPRINT_REPORTS_PATHS, BRIEF_PATHS (comma-separated)
#   - discord.env  optional: DISCORD_WEBHOOK_URL
#
# Override the brain repo location with $SDET_BRAIN_DIR, the server
# daemon label with $SDET_BRAIN_DAEMON_LABEL.

set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SDET_BRAIN_DIR=${SDET_BRAIN_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
SDET_BRAIN_CONFIG_DIR=${SDET_BRAIN_CONFIG_DIR:-$HOME/.config/sdet-brain}
LOG=/tmp/sdet-brain-daily.log
NOW=$(date '+%Y-%m-%d %H:%M:%S')

[ -d "$SDET_BRAIN_DIR" ] || { printf '[%s] FATAL: SDET_BRAIN_DIR=%s does not exist\n' "$NOW" "$SDET_BRAIN_DIR" >> "$LOG"; exit 2; }

# --- Source local config (paths + secrets, never committed) ---
for env_file in paths.env discord.env; do
  full="$SDET_BRAIN_CONFIG_DIR/$env_file"
  if [ -f "$full" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$full"
    set +a
  fi
done

# --- Defaults so unset vars don't trip set -u below ---
: "${DRAFTS_PATHS:=}"
: "${ARTICLES_PATHS:=}"
: "${PROJECT_KNOWLEDGE_PATHS:=}"
: "${SPRINT_REPORTS_PATHS:=}"
: "${BRIEF_PATHS:=}"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }

count_chunks() {
  /usr/bin/curl -sS -m 5 "${SDET_BRAIN_SERVER_URL:-http://localhost:8080}/health" 2>/dev/null \
    | /usr/bin/sed -n 's/.*"collection_count":\([0-9]*\).*/\1/p'
}

notify() {
  local title=$1
  local subtitle=$2
  local message=$3
  local sound=${4:-}
  local sound_clause=""
  [ -n "$sound" ] && sound_clause=" sound name \"$sound\""
  /usr/bin/osascript -e "display notification \"$message\" with title \"$title\" subtitle \"$subtitle\"$sound_clause" 2>>"$LOG"
}

log "=== daily start ==="
chunks_before=$(count_chunks)
log "chunks_before=${chunks_before:-unknown}"

# Build the list of paths to walk in ingest order. Empty vars produce
# no entries.
ALL_PATHS=()
for env_var in DRAFTS_PATHS ARTICLES_PATHS PROJECT_KNOWLEDGE_PATHS SPRINT_REPORTS_PATHS BRIEF_PATHS; do
  raw=${!env_var}
  [ -z "$raw" ] && continue
  IFS=',' read -ra entries <<< "$raw"
  ALL_PATHS+=("${entries[@]}")
done

if [ ${#ALL_PATHS[@]} -eq 0 ]; then
  log "no corpus paths configured - set DRAFTS_PATHS / ARTICLES_PATHS / ... in $SDET_BRAIN_CONFIG_DIR/paths.env"
  notify "sdet-brain daily" "$NOW" "no corpus paths configured" "Basso"
  exit 1
fi

ingest_failures=0

cd "$SDET_BRAIN_DIR" || { log "FATAL: cd $SDET_BRAIN_DIR failed"; exit 2; }

UV_BIN=${UV_BIN:-/opt/homebrew/bin/uv}
[ -x "$UV_BIN" ] || UV_BIN=$(/usr/bin/which uv 2>/dev/null || echo uv)

for path in "${ALL_PATHS[@]}"; do
  [ -d "$path" ] || { log "skip (not a dir): $path"; continue; }
  log "ingest: $path"
  out=$("$UV_BIN" run sdet-brain-cli "$path" --exclude node_modules --exclude __pycache__ 2>&1)
  rc=$?
  echo "$out" >> "$LOG"
  [ $rc -ne 0 ] && { ingest_failures=$((ingest_failures + 1)); log "FAIL rc=$rc on $path"; }
done

chunks_after=$(count_chunks)
delta="?"
if [ -n "${chunks_before:-}" ] && [ -n "${chunks_after:-}" ]; then
  delta=$((chunks_after - chunks_before))
fi
log "chunks_after=${chunks_after:-unknown} delta=$delta failures=$ingest_failures"

# --- Daily LLM digest (cold-start MLX). Skipped on ingest failure
#     to avoid noisy alerts on broken state. ---
digest_rc=0
if [ $ingest_failures -eq 0 ] && [ -x "$SDET_BRAIN_DIR/scripts/digest.py" ]; then
  log "running digest.py"
  "$UV_BIN" run "$SDET_BRAIN_DIR/scripts/digest.py" >> "$LOG" 2>&1
  digest_rc=$?
  log "digest_rc=$digest_rc"
elif [ $ingest_failures -gt 0 ]; then
  log "skip digest - ingest had failures"
fi

# --- Health check banner ---
"$SDET_BRAIN_DIR/scripts/healthcheck.sh"
health_rc=$?

# --- Ingest-delta banner (separate from health banner) ---
if [ $ingest_failures -eq 0 ]; then
  notify "sdet-brain ingest" "$NOW" "delta=$delta · total=${chunks_after:-?} chunks"
  log "OK delta=$delta"
  exit $health_rc
fi
notify "sdet-brain ingest FAIL" "$NOW" "$ingest_failures path(s) failed - see $LOG" "Basso"
log "FAIL failures=$ingest_failures"
exit 1
