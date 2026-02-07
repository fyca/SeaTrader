from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path("data")


def write_artifact(name: str, payload: Any) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = DATA_DIR / name
    obj = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))
    return p


def _json_default(x):
    try:
        return asdict(x)
    except Exception:
        return str(x)
