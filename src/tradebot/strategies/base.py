from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class Candidate:
    symbol: str
    score: float
    details: dict


class Strategy(Protocol):
    id: str
    name: str

    def select_equities(self, *, bars: dict[str, pd.DataFrame], cfg) -> tuple[list[str], dict[str, dict]]:
        ...

    def select_crypto(self, *, bars: dict[str, pd.DataFrame], cfg) -> tuple[list[str], dict[str, dict]]:
        ...
