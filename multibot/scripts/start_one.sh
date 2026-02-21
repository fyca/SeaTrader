#!/usr/bin/env bash
set -euo pipefail
BOT="${1:-}"
if [[ -z "$BOT" ]]; then
  echo "Usage: $0 <bot-name>"
  exit 1
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

case "$BOT" in
  alpha) PORT=8008 ;;
  beta) PORT=8009 ;;
  gamma) PORT=8010 ;;
  delta) PORT=8011 ;;
  epsilon) PORT=8012 ;;
  zeta) PORT=8013 ;;
  eta) PORT=8014 ;;
  theta) PORT=8015 ;;
  iota) PORT=8016 ;;
  *) echo "Unknown bot: $BOT"; exit 1 ;;
esac

if [[ -f "$PID_DIR/$BOT-dashboard.pid" ]] && kill -0 "$(cat "$PID_DIR/$BOT-dashboard.pid")" 2>/dev/null; then
  echo "$BOT dashboard already running (pid $(cat "$PID_DIR/$BOT-dashboard.pid"))"
  exit 0
fi
nohup "$ROOT/scripts/run_bot.sh" "$BOT" dashboard --port "$PORT" > "$LOG_DIR/$BOT-dashboard.log" 2>&1 &
echo $! > "$PID_DIR/$BOT-dashboard.pid"
echo "Started $BOT dashboard on :$PORT (pid $!)"
