#!/bin/sh
# Run a command with Confidant's API keys taken from the macOS keychain.
# The keys are never printed or written to disk; they only live in the child
# process environment.
#
# One-time setup (paste each key at the prompt, it is not echoed):
#   security add-generic-password -U -a "$USER" -s nebius-token-factory -w
#   security add-generic-password -U -a "$USER" -s tavily -w
#
# Usage: scripts/with-keys.sh python3 -m confidant chat
set -eu

key() { security find-generic-password -s "$1" -w 2>/dev/null || true; }

NEBIUS_API_KEY="$(key nebius-token-factory)"
TAVILY_API_KEY="$(key tavily)"
export NEBIUS_API_KEY TAVILY_API_KEY

if [ -z "$NEBIUS_API_KEY" ]; then
  echo "note: no 'nebius-token-factory' keychain item; Confidant will run in replay mode" >&2
fi
if [ -z "$TAVILY_API_KEY" ]; then
  echo "note: no 'tavily' keychain item; web search is disabled" >&2
fi

exec "$@"
