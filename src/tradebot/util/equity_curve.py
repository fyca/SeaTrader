from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


CURVE_PATH = Path("data/equity_curve.jsonl")


@dataclass(frozen=True)
class EquityPoint:
    ts: str
    equity: float
    cash: float


def append_equity_point(*, equity: float, cash: float, path: Path = CURVE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pt = EquityPoint(ts=datetime.now(timezone.utc).isoformat(), equity=float(equity), cash=float(cash))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(pt.__dict__, sort_keys=True) + "\n")
