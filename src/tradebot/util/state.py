from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


STATE_PATH = Path("data/state.json")


@dataclass
class BotState:
    peak_equity: float | None = None


def load_state(path: Path = STATE_PATH) -> BotState:
    if not path.exists():
        return BotState()
    obj = json.loads(path.read_text())
    return BotState(peak_equity=obj.get("peak_equity"))


def save_state(state: BotState, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"peak_equity": state.peak_equity}, indent=2, sort_keys=True))
