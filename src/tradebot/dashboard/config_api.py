from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import HTTPException

from tradebot.util.config import BotConfig


def load_config_file(path: str) -> BotConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    return BotConfig.model_validate(data)


def save_config_file(path: str, cfg: BotConfig) -> None:
    p = Path(path)
    # Preserve key order-ish by not sorting
    p.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))


def validate_config_payload(payload: dict) -> BotConfig:
    try:
        return BotConfig.model_validate(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {e}")
