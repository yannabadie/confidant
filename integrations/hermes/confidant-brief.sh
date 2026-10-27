#!/bin/sh
# Hermes Agent cron job in no-agent mode: print the daily brief as it may leave the device (private items
# as their label only). Hermes delivers this script's standard output to Telegram; no Hermes model reads it.
set -eu
cd "${CONFIDANT_DIR:-$HOME/Documents/code/confidant}"
exec /bin/sh scripts/with-keys.sh .venv/bin/python -m confidant --home "${CONFIDANT_HOME:-$HOME/.confidant}" \
  brief --delivery
