#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Create venv if missing
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Install deps (editable) if needed
pip install -e . >/dev/null

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8008}
CONFIG=${CONFIG:-config/config.yaml}

exec tradebot dashboard --config "$CONFIG" --host "$HOST" --port "$PORT"
