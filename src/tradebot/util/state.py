from __future__ import annotations

import json
import os
import tempfile
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
    """Save state atomically to prevent corruption on concurrent writes or crashes.
    
    Uses write-to-temp + rename pattern: either the entire file is written or nothing is.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare the data to write
    data = json.dumps(
        {
            "peak_equity": state.peak_equity,
            "excluded_symbols": [str(s).upper() for s in (state.excluded_symbols or [])],
            "trailing_peaks": {str(k).upper(): float(v) for k, v in (state.trailing_peaks or {}).items()},
        },
        indent=2,
        sort_keys=True,
    )
    
    # Write to temporary file in the same directory (ensures same filesystem for atomic rename)
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=str(path.parent),
        delete=False,
        encoding='utf-8'
    ) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    
    # Atomic rename: either succeeds fully or fails (no partial writes)
    try:
        # On Windows, os.rename raises if target exists; on Unix it overwrites atomically
        # Use replace() for cross-platform atomic behavior
        os.replace(tmp_path, str(path))
    except Exception:
        # If rename fails, clean up temp file and re-raise
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise
