#!/usr/bin/env bash
set -euo pipefail

BOT="${1:-}"
if [[ -z "$BOT" ]]; then
  echo "Usage: $0 <bot-name>"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # tradebot/multibot
REPO="$(cd "$ROOT/.." && pwd)"                                # tradebot
BOT_DIR="$ROOT/bots/$BOT"
PY="$REPO/.venv/bin/python"

if [[ ! -d "$BOT_DIR" ]]; then
  echo "Unknown bot: $BOT"
  exit 1
fi
if [[ ! -f "$BOT_DIR/.env" ]]; then
  echo "Missing env file: $BOT_DIR/.env"
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$BOT_DIR/.env"
set +a

"$PY" - <<'PY'
import os
from alpaca.trading.client import TradingClient
key = os.getenv('APCA_API_KEY_ID')
secret = os.getenv('APCA_API_SECRET_KEY')
paper = str(os.getenv('APCA_PAPER','true')).lower() == 'true'
if not key or not secret:
    raise SystemExit('Missing APCA_API_KEY_ID/APCA_API_SECRET_KEY in bot .env')
c = TradingClient(key, secret, paper=paper)
try:
    c.cancel_orders()
except Exception:
    pass
try:
    c.close_all_positions(cancel_orders=True)
except Exception:
    pass
print('broker account flatten requested')
PY

mkdir -p "$BOT_DIR/data" "$BOT_DIR/logs"
find "$BOT_DIR/data" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
find "$BOT_DIR/logs" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
echo "wipe complete for $BOT"
