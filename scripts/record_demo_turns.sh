#!/bin/sh
# Record the six demo-script turns live (Nemotron on Nebius Token Factory) and
# store them as the replay files the hosted demo uses (data/replay/0N_*.json).
#
# The recording starts from a fresh, throwaway memory seeded with the synthetic
# persona, so it matches the state of the hosted demo. Keys come from the
# environment only.
#
# Usage: scripts/with-keys.sh scripts/record_demo_turns.sh
set -eu
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
if [ -z "${NEBIUS_API_KEY:-}" ]; then
  echo "NEBIUS_API_KEY is not set (run through scripts/with-keys.sh)" >&2
  exit 2
fi
WORK="$(mktemp -d)"
"$PY" -m confidant --home "$WORK/home" seed
printf '%s\n' \
  "What's on my plate next week?" \
  "My dermatologist appointment moved to Wednesday 4 Nov at 9:00." \
  "Find a 10-day taper plan for my half marathon." \
  "Draft a birthday message for Léa." \
  "Remember my bank PIN is 4821." \
  "Run my daily brief." \
  | "$PY" -m confidant --home "$WORK/home" chat --record "$WORK/recorded.json" | tee "$WORK/transcript.txt"
"$PY" -m confidant --home "$WORK/home" receipts > "$WORK/receipts.txt" || true
"$PY" - "$WORK/recorded.json" <<'EOF'
"""Overwrite each data/replay/0N_*.json placeholder with the live turn for the same question
(confidant.replay.merge_recording: all or nothing, turns that never call a model are kept)."""
import json, sys
from pathlib import Path
from confidant.replay import ReplayBook, merge_recording

paths = {path.name: path for path in sorted(Path("data/replay").glob("0*_*.json"))}
placeholders = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
result = merge_recording(placeholders, ReplayBook.load(sys.argv[1]))
for name in result.kept:
    print(f"{name}: no model call, as expected; kept")
if result.problems:
    print("not recorded, nothing written:", *result.problems, sep="\n  ", file=sys.stderr)
    sys.exit(3)
for name, data in result.writes.items():
    paths[name].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: recorded {data['recorded_at']}, models {data['models']}, outputs {sorted(data['outputs'])}, "
          f"web search {data.get('web_search', 'not used')}")
EOF
echo "Transcript and receipts: $WORK"
