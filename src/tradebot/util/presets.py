from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PRESETS_PATH = Path("config/presets.yaml")
LEGACY_BACKTEST_PRESETS_PATH = Path("config/backtest_presets.yaml")


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_presets() -> list[dict[str, Any]]:
    """Load unified presets.

    Format:
    presets:
      - name: <str>
        bot: <dict>        # patch applied to config/config.yaml
        backtest: <dict>    # backtest params

    Back-compat: if config/presets.yaml doesn't exist yet, we will read
    config/backtest_presets.yaml and return items with backtest=params.
    """
    if PRESETS_PATH.exists():
        data = _read_yaml(PRESETS_PATH)
        return list(data.get("presets") or [])

    # legacy
    if LEGACY_BACKTEST_PRESETS_PATH.exists():
        data = _read_yaml(LEGACY_BACKTEST_PRESETS_PATH)
        out = []
        for p in (data.get("presets") or []):
            out.append({
                "name": p.get("name"),
                "bot": {},
                "backtest": p.get("params") or {},
            })
        return out

    return []


def get_preset(name: str) -> dict[str, Any] | None:
    name = str(name)
    for p in load_presets():
        if str(p.get("name")) == name:
            return p
    return None


def save_preset(*, name: str, bot: dict[str, Any] | None = None, backtest: dict[str, Any] | None = None) -> None:
    name = str(name).strip()
    if not name:
        raise ValueError("missing preset name")

    bot = bot or {}
    backtest = backtest or {}

    presets = load_presets()
    # replace if exists
    out = [p for p in presets if str(p.get("name")) != name]
    out.append({"name": name, "bot": bot, "backtest": backtest})
    out.sort(key=lambda x: str(x.get("name") or ""))

    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_PATH.write_text(yaml.safe_dump({"presets": out}, sort_keys=False))
