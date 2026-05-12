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

# --- Memory guard (avoid OOM-killing the laptop) ---
# Ingest pulls a second MLX embedder in CLI mode and digest cold-starts
# a 30B+ LLM. Skip both if the machine can't spare the headroom.
INGEST_MIN_GB=${INGEST_MIN_GB:-20}
DIGEST_MIN_GB=${DIGEST_MIN_GB:-30}
available_gb=$(/usr/bin/vm_stat | /usr/bin/awk '
  /Pages free/ {f=$3+0}
  /Pages inactive/ {i=$3+0}
  /Pages speculative/ {s=$3+0}
  END {printf "%.1f", (f+i+s)*16384/1073741824}
')
log "memory_free=${available_gb}GB ingest_min=${INGEST_MIN_GB}GB digest_min=${DIGEST_MIN_GB}GB"
mem_int=${available_gb%.*}
if [ "${mem_int:-0}" -lt "$INGEST_MIN_GB" ]; then
  log "SKIP daily — only ${available_gb}GB free, need >=${INGEST_MIN_GB}GB for ingest"
  notify "sdet-brain daily SKIPPED" "$NOW" "low memory: ${available_gb}GB free (need ${INGEST_MIN_GB}GB)" "Basso"
  exit 0
fi

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

# Route ingest through the already-running server's POST /ingest. This
# avoids spawning a second MLX embedder in CLI mode (which doubled the
# RAM footprint and used to OOM the laptop). The server reuses its
# loaded embedder and storage handles via FastAPI deps.
SERVER_URL=${SDET_BRAIN_SERVER_URL:-http://localhost:8080}

for path in "${ALL_PATHS[@]}"; do
  [ -d "$path" ] || { log "skip (not a dir): $path"; continue; }
  log "ingest (HTTP): $path"
  body=$(/usr/bin/python3 -c "
import json, sys
print(json.dumps({
    'path': sys.argv[1],
    'exclude_dirs': ['node_modules', '__pycache__', '.claude'],
}))
" "$path")
  out=$(/usr/bin/curl -sS --max-time 1800 \
    -H 'Content-Type: application/json' \
    -X POST "$SERVER_URL/ingest" \
    -d "$body" 2>&1)
  rc=$?
  echo "$out" >> "$LOG"
  if [ $rc -ne 0 ]; then
    ingest_failures=$((ingest_failures + 1))
    log "FAIL curl rc=$rc on $path"
  elif ! echo "$out" | /usr/bin/grep -q '"summary"'; then
    ingest_failures=$((ingest_failures + 1))
    log "FAIL non-200 response on $path"
  fi
done

chunks_after=$(count_chunks)
delta="?"
if [ -n "${chunks_before:-}" ] && [ -n "${chunks_after:-}" ]; then
  delta=$((chunks_after - chunks_before))
fi
log "chunks_after=${chunks_after:-unknown} delta=$delta failures=$ingest_failures"

# --- Daily LLM digest (cold-start MLX). Skipped on ingest failure
#     to avoid noisy alerts on broken state, and skipped on low memory
#     because the LLM cold-start pulls 10-15 GB.
UV_BIN=${UV_BIN:-/opt/homebrew/bin/uv}
[ -x "$UV_BIN" ] || UV_BIN=$(/usr/bin/which uv 2>/dev/null || echo uv)
digest_rc=0
digest_free_gb=$(/usr/bin/vm_stat | /usr/bin/awk '
  /Pages free/ {f=$3+0}
  /Pages inactive/ {i=$3+0}
  /Pages speculative/ {s=$3+0}
  END {printf "%.1f", (f+i+s)*16384/1073741824}
')
digest_mem_int=${digest_free_gb%.*}
if [ $ingest_failures -gt 0 ]; then
  log "skip digest - ingest had failures"
elif [ "${digest_mem_int:-0}" -lt "$DIGEST_MIN_GB" ]; then
  log "skip digest - only ${digest_free_gb}GB free, need >=${DIGEST_MIN_GB}GB"
elif [ -x "$SDET_BRAIN_DIR/scripts/digest.py" ]; then
  log "running digest.py (free=${digest_free_gb}GB)"
  "$UV_BIN" run "$SDET_BRAIN_DIR/scripts/digest.py" >> "$LOG" 2>&1
  digest_rc=$?
  log "digest_rc=$digest_rc"
fi

# --- Auto-restart server to release MLX arena (~tens of GB after large ingest) ---
# Apple Silicon unified memory + MLX framework holds GPU arena even when
# inference is idle. Restart drops the arena; model lazy-reloads on next
# /search or /ingest. See known failure modes #4 in memory note.
SERVER_LABEL=${SDET_BRAIN_DAEMON_LABEL:-com.darkow.sdet-brain}
log "auto-restart server $SERVER_LABEL to release MLX arena"
/bin/launchctl kickstart -k "gui/$(/usr/bin/id -u)/$SERVER_LABEL" >> "$LOG" 2>&1
sleep 3

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
