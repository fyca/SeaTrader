#!/usr/bin/env bash
set -euo pipefail
for line in \
  "alpha 8008" "beta 8009" "gamma 8010" \
  "delta 8011" "epsilon 8012" "zeta 8013" \
  "eta 8014" "theta 8015" "iota 8016"
do
  bot=$(echo "$line" | awk '{print $1}')
  port=$(echo "$line" | awk '{print $2}')
  if curl -fsS "http://127.0.0.1:$port/" >/dev/null 2>&1; then
    echo "$bot: UP (http://127.0.0.1:$port)"
  else
    echo "$bot: DOWN (http://127.0.0.1:$port)"
  fi
done
