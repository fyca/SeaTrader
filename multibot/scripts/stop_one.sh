#!/usr/bin/env bash
set -euo pipefail
BOT="${1:-}"
if [[ -z "$BOT" ]]; then
  echo "Usage: $0 <bot-name>"
  exit 1
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$ROOT/pids"
P="$PID_DIR/$BOT-dashboard.pid"

if [[ ! -f "$P" ]]; then
  echo "$BOT dashboard pid file not found"
  exit 0
fi

PID="$(cat "$P")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID" || true
  echo "Stopped $BOT dashboard (pid $PID)"
else
  echo "$BOT dashboard not running"
fi
rm -f "$P"
