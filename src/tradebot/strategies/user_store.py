from __future__ import annotations

import json
from pathlib import Path


# Keep user strategies in the repository-level shared folder so all bot instances
# (including multibot dashboards launched from per-bot working dirs) see the same library.
USER_DIR = Path(__file__).resolve().parents[3] / "strategies" / "user"


def list_user_strategies() -> list[dict]:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(USER_DIR.glob("*.json")):
        try:
            obj = json.loads(p.read_text())
            typ = obj.get("type")
            asset = obj.get("asset_class")
            out.append(
                {
                    "id": obj.get("id") or p.stem,
                    "name": obj.get("name") or p.stem,
                    "version": int(obj.get("version") or 1),
                    "type": typ,
                    "asset_class": asset,
                    "legacy_untyped": not bool(typ),
                    "source": "user",
                }
            )
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


def convert_legacy_strategy(
    strategy_id: str,
    *,
    target_type: str,
    target_asset_class: str,
    new_id: str | None = None,
    new_name: str | None = None,
) -> dict:
    obj = load_user_strategy(strategy_id)
    if obj.get("type"):
        raise ValueError("strategy already typed")

    if target_type not in ("entry", "exit"):
        raise ValueError("target_type must be entry or exit")
    if target_asset_class not in ("stocks", "crypto"):
        raise ValueError("target_asset_class must be stocks or crypto")

    out = dict(obj)
    out["id"] = new_id or obj.get("id") or strategy_id
    out["name"] = new_name or obj.get("name") or out["id"]
    out["type"] = target_type
    out["asset_class"] = target_asset_class
    out["version"] = int(out.get("version") or 1)

    save_user_strategy(str(out["id"]), out)
    return out
