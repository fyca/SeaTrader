from __future__ import annotations

import json
from pathlib import Path


USER_DIR = Path("strategies/user")


def list_user_strategies() -> list[dict]:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(USER_DIR.glob("*.json")):
        try:
            obj = json.loads(p.read_text())
            out.append({"id": obj.get("id") or p.stem, "name": obj.get("name") or p.stem})
        except Exception:
            continue
    return out


def load_user_strategy(strategy_id: str) -> dict:
    p = USER_DIR / f"{strategy_id}.json"
    if not p.exists():
        raise FileNotFoundError(strategy_id)
    return json.loads(p.read_text())


def save_user_strategy(strategy_id: str, obj: dict) -> None:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    obj = dict(obj)
    obj.setdefault("id", strategy_id)
    p = USER_DIR / f"{strategy_id}.json"
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))


def delete_user_strategy(strategy_id: str) -> None:
    p = USER_DIR / f"{strategy_id}.json"
    if p.exists():
        p.unlink()
