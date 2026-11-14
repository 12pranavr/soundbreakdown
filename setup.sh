#!/usr/bin/env bash
# One-time setup: creates ../.venv next to this folder and installs everything (free / open-source only).
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
python3 -m venv "$DIR/../.venv"
"$DIR/../.venv/bin/pip" install -q --upgrade pip
"$DIR/../.venv/bin/pip" install -q -r "$DIR/requirements.txt"
echo "Setup done. Run: ./run.sh path/to/song.mp3"
