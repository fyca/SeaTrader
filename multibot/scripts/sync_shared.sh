#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # tradebot/multibot
REPO="$(cd "$ROOT/.." && pwd)"                                # tradebot

for bot in alpha beta gamma delta epsilon zeta eta theta iota; do
  BOT_DIR="$ROOT/bots/$bot"
  mkdir -p "$BOT_DIR/config" "$BOT_DIR/strategies/user"

  if [[ -d "$REPO/strategies/user" ]]; then
    rsync -a --delete "$REPO/strategies/user/" "$BOT_DIR/strategies/user/"
  fi

  [[ -f "$REPO/config/presets.yaml" ]] && cp "$REPO/config/presets.yaml" "$BOT_DIR/config/presets.yaml"
  [[ -f "$REPO/config/backtest_presets.yaml" ]] && cp "$REPO/config/backtest_presets.yaml" "$BOT_DIR/config/backtest_presets.yaml"
done

echo "Shared strategies/presets synced to alpha..iota (9 bots)."
