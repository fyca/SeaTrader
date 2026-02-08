from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PRESETS_PATH = Path("config/backtest_presets.yaml")


def load_presets() -> list[dict]:
    if not PRESETS_PATH.exists():
        return []
    data = yaml.safe_load(PRESETS_PATH.read_text()) or {}
    return list(data.get("presets") or [])


def save_preset(name: str, params: dict[str, Any]) -> None:
    name = str(name).strip()
    if not name:
        raise ValueError("missing preset name")

    presets = load_presets()
    # replace if exists
    out = [p for p in presets if str(p.get("name")) != name]
    out.append({"name": name, "params": params})
    # stable sort by name
    out.sort(key=lambda x: str(x.get("name") or ""))

    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_PATH.write_text(yaml.safe_dump({"presets": out}, sort_keys=False))


def get_preset(name: str) -> dict | None:
    for p in load_presets():
        if str(p.get("name")) == name:
            return p
    return None
