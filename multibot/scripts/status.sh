#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - <<'PY'
from urllib.request import urlopen
ports={"alpha":8008,"beta":8009,"gamma":8010,"delta":8011,"epsilon":8012,"zeta":8013,"eta":8014,"theta":8015,"iota":8016}
for b,p in ports.items():
    try:
        with urlopen(f"http://127.0.0.1:{p}/", timeout=1.5):
            print(f"{b}: UP (http://127.0.0.1:{p})")
    except Exception:
        print(f"{b}: DOWN (http://127.0.0.1:{p})")
PY
