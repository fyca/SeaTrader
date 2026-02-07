from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import GetAssetsRequest


_LEVERAGED_PAT = re.compile(
    r"(\b2x\b|\b3x\b|\bUltra\b|\bLeveraged\b|\bInverse\b|\bBear\b|\bBull\b|Direxion\s+Daily|ProShares\s+Ultra)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EquityUniverseItem:
    symbol: str
    name: str
    exchange: str | None


def is_leveraged_or_inverse_etf(name: str | None) -> bool:
    if not name:
        return False
    return bool(_LEVERAGED_PAT.search(name))


def list_tradable_equities(
    trading_client,
    *,
    exclude_leveraged_etfs: bool,
) -> list[EquityUniverseItem]:
    """Return a broad list of active, tradable US equities/ETFs.

    NOTE: We'll apply price/liquidity filters later.
    """
    req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
    assets = trading_client.get_all_assets(req)

    out: list[EquityUniverseItem] = []
    for a in assets:
        # a.status: 'active'/'inactive'
        if getattr(a, "status", None) != "active":
            continue
        if not getattr(a, "tradable", False):
            continue
        if exclude_leveraged_etfs and is_leveraged_or_inverse_etf(getattr(a, "name", None)):
            continue
        sym = getattr(a, "symbol", None)
        if not sym:
            continue
        out.append(
            EquityUniverseItem(
                symbol=sym,
                name=getattr(a, "name", "") or "",
                exchange=getattr(a, "exchange", None),
            )
        )

    # de-dupe
    seen: set[str] = set()
    deduped: list[EquityUniverseItem] = []
    for it in out:
        if it.symbol in seen:
            continue
        seen.add(it.symbol)
        deduped.append(it)
    return deduped
