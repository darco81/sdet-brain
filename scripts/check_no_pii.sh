#!/usr/bin/env bash
# Fail if hardcoded home-path PII (OS username in /Users|/home paths) is found
# in tracked files. Wired into pre-commit + CI so a future doc/script cannot
# re-leak the local FS layout after the v0.5.1 scrub.
#
# The [d] character class keeps the pattern from matching this script itself.
set -euo pipefail

pattern='/(Users|home)/[d]ariusz'

# Only scan tracked files; never the .git dir or lockfiles.
matches=$(git ls-files | grep -vE '(^|/)uv\.lock$' \
  | xargs grep -InE "$pattern" 2>/dev/null || true)

if [ -n "$matches" ]; then
  echo "ERROR: hardcoded home-path PII found in tracked files:" >&2
  echo "$matches" >&2
  exit 1
fi
echo "no-home-path-pii: clean"
