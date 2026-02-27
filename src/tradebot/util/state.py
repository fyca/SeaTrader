from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


STATE_PATH = Path("data/state.json")


@dataclass
class BotState:
    peak_equity: float | None = None
    excluded_symbols: list[str] | None = None
    trailing_peaks: dict[str, float] | None = None


def load_state(path: Path = STATE_PATH) -> BotState:
    if not path.exists():
        return BotState()
    obj = json.loads(path.read_text())
    ex = obj.get("excluded_symbols") or []
    if not isinstance(ex, list):
        ex = []
    tp = obj.get("trailing_peaks") or {}
    if not isinstance(tp, dict):
        tp = {}
    cleaned_tp: dict[str, float] = {}
    for k, v in tp.items():
        try:
            cleaned_tp[str(k).upper()] = float(v)
        except Exception:
            continue
    return BotState(peak_equity=obj.get("peak_equity"), excluded_symbols=[str(s).upper() for s in ex], trailing_peaks=cleaned_tp)


def save_state(state: BotState, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "peak_equity": state.peak_equity,
                "excluded_symbols": [str(s).upper() for s in (state.excluded_symbols or [])],
                "trailing_peaks": {str(k).upper(): float(v) for k, v in (state.trailing_peaks or {}).items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
