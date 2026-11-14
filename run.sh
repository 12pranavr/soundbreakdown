#!/usr/bin/env bash
# Usage: ./run.sh path/to/song.mp3   → writes breakdowns/<song>/report.md, timeline.png, stems/
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/../.venv/bin/python" "$DIR/breakdown.py" "$@" 2> >(grep -v -E "%\||Warning|HF_TOKEN" >&2)
