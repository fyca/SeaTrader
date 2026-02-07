from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


CACHE_DIR = Path("data/cache/bars")


def _key(prefix: str, symbols: list[str], lookback_days: int, start_iso: str, end_iso: str) -> str:
    h = hashlib.sha256()
    h.update(prefix.encode())
    h.update(str(lookback_days).encode())
    h.update(start_iso.encode())
    h.update(end_iso.encode())
    for s in symbols:
        h.update(s.encode())
        h.update(b"\0")
    return h.hexdigest()[:24]


def load_cached_frames(prefix: str, symbols: list[str], lookback_days: int, start_iso: str, end_iso: str) -> dict[str, pd.DataFrame] | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    k = _key(prefix, symbols, lookback_days, start_iso, end_iso)
    meta = CACHE_DIR / f"{prefix}-{k}.json"
    if not meta.exists():
        return None
    obj = json.loads(meta.read_text())
    out: dict[str, pd.DataFrame] = {}
    for sym, rel in obj.get("files", {}).items():
        p = CACHE_DIR / rel
        if not p.exists():
            return None
        out[sym] = pd.read_parquet(p)
    return out


def save_cached_frames(prefix: str, symbols: list[str], lookback_days: int, start_iso: str, end_iso: str, frames: dict[str, pd.DataFrame]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    k = _key(prefix, symbols, lookback_days, start_iso, end_iso)
    files: dict[str, str] = {}
    for sym, df in frames.items():
        rel = f"{prefix}-{k}-{sym.replace('/', '_')}.parquet"
        (CACHE_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_DIR / rel)
        files[sym] = rel
    (CACHE_DIR / f"{prefix}-{k}.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True))
