#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TS=$(date -u +"%Y%m%dT%H%M%SZ")
mkdir -p data/config-snapshots
cp config/config.yaml "data/config-snapshots/config-${TS}.yaml"
echo "Saved config snapshot: data/config-snapshots/config-${TS}.yaml"
