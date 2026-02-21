#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for b in alpha beta gamma delta epsilon zeta eta theta iota; do
  "$ROOT/scripts/stop_one.sh" "$b" || true
done
