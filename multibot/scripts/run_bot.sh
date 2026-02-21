#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <bot-name> <rebalance|risk-check|dashboard> [extra args...]"
  exit 1
fi

BOT="$1"; shift
CMD="$1"; shift || true

MB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # tradebot/multibot
REPO_ROOT="$(cd "$MB_ROOT/.." && pwd)"                         # tradebot
BOT_DIR="$MB_ROOT/bots/$BOT"
TB_BIN="$REPO_ROOT/.venv/bin/tradebot"

if [[ ! -d "$BOT_DIR" ]]; then
  echo "Unknown bot: $BOT"
  exit 1
fi

if [[ ! -x "$TB_BIN" ]]; then
  echo "tradebot CLI not found at $TB_BIN"
  exit 1
fi

if [[ -f "$BOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$BOT_DIR/.env"
  set +a
fi

cd "$BOT_DIR"

case "$CMD" in
  rebalance)
    exec "$TB_BIN" rebalance --config "$BOT_DIR/config/config.yaml" "$@"
    ;;
  risk-check)
    exec "$TB_BIN" risk-check --config "$BOT_DIR/config/config.yaml" "$@"
    ;;
  dashboard)
    PORT="${PORT:-}"
    if [[ -z "$PORT" ]]; then
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
    fi
    exec "$TB_BIN" dashboard --config "$BOT_DIR/config/config.yaml" --host 127.0.0.1 --port "$PORT"
    ;;
  *)
    echo "Unknown command: $CMD"
    exit 1
    ;;
esac
