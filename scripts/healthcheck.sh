#!/bin/bash
# sdet-brain health check + macOS banner. Standalone (no LLM, no
# ingest) - pings Qdrant, the server's `/health`, and verifies the
# server daemon is loaded in launchd. Run by daily.sh, callable by
# hand for spot checks.
#
# Override:
#   QDRANT_URL                   default http://localhost:6333
#   SDET_BRAIN_SERVER_URL        default http://localhost:8080
#   SDET_BRAIN_DAEMON_LABEL      default com.local.sdet-brain
#                                (set this if you renamed the plist)

set -u

LOG=/tmp/sdet-brain-health.log
QDRANT_URL=${QDRANT_URL:-http://localhost:6333}
SERVER_URL=${SDET_BRAIN_SERVER_URL:-http://localhost:8080}
DAEMON_LABEL=${SDET_BRAIN_DAEMON_LABEL:-com.local.sdet-brain}
NOW=$(date '+%Y-%m-%d %H:%M:%S')

log() { printf '[%s] %s\n' "$NOW" "$*" >> "$LOG"; }

notify() {
  local title=$1
  local subtitle=$2
  local message=$3
  local sound=${4:-}
  local sound_clause=""
  [ -n "$sound" ] && sound_clause=" sound name \"$sound\""
  /usr/bin/osascript -e "display notification \"$message\" with title \"$title\" subtitle \"$subtitle\"$sound_clause" 2>>"$LOG"
}

problems=()

qdrant_body=$(/usr/bin/curl -sS -m 5 "$QDRANT_URL/readyz" 2>>"$LOG")
qdrant_code=$?
if [ $qdrant_code -ne 0 ] || ! printf '%s' "$qdrant_body" | grep -q 'all shards are ready'; then
  problems+=("Qdrant down ($QDRANT_URL)")
fi

server_body=$(/usr/bin/curl -sS -m 5 "$SERVER_URL/health" 2>>"$LOG")
server_code=$?
chunks="?"
embedder="?"
if [ $server_code -ne 0 ] || [ -z "$server_body" ]; then
  problems+=("Server unreachable ($SERVER_URL)")
else
  status_ok=$(printf '%s' "$server_body" | /usr/bin/sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
  chunks=$(printf '%s' "$server_body" | /usr/bin/sed -n 's/.*"collection_count":\([0-9]*\).*/\1/p')
  embedder=$(printf '%s' "$server_body" | /usr/bin/sed -n 's/.*"embedder_provider":"\([^"]*\)".*/\1/p')
  fell_back=$(printf '%s' "$server_body" | /usr/bin/sed -n 's/.*"embedder_fell_back":\([a-z]*\).*/\1/p')
  qdrant_ok=$(printf '%s' "$server_body" | /usr/bin/sed -n 's/.*"qdrant_ok":\([a-z]*\).*/\1/p')
  embedder_ok=$(printf '%s' "$server_body" | /usr/bin/sed -n 's/.*"embedder_ok":\([a-z]*\).*/\1/p')

  [ "$status_ok" = "ok" ] || problems+=("server status=$status_ok")
  [ "$qdrant_ok" = "true" ] || problems+=("server reports qdrant_ok=$qdrant_ok")
  [ "$embedder_ok" = "true" ] || problems+=("embedder unhealthy ($embedder)")
  [ "$fell_back" = "true" ] && problems+=("primary embedder fell back to fallback")
fi

if /bin/launchctl list | /usr/bin/grep -q "$DAEMON_LABEL"; then
  daemon_state="loaded"
else
  daemon_state="MISSING"
  problems+=("daemon $DAEMON_LABEL not loaded")
fi

log "qdrant=$([ ${#problems[@]} -eq 0 ] && echo ok || echo see-below) chunks=${chunks:-?} embedder=${embedder:-?} daemon=$daemon_state"

if [ ${#problems[@]} -eq 0 ]; then
  notify "sdet-brain OK" "$NOW" "embedder=$embedder · chunks=$chunks · daemon=$daemon_state"
  log "OK"
  exit 0
fi
joined=$(printf '%s; ' "${problems[@]}" | /usr/bin/sed 's/; $//')
notify "sdet-brain ALERT" "$NOW" "$joined" "Basso"
log "FAIL: $joined"
exit 1
