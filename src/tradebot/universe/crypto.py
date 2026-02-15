from __future__ import annotations

from dataclasses import dataclass

from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import GetAssetsRequest


_STABLECOIN_BASES = {
    "USDC", "USDT", "DAI", "BUSD", "TUSD", "USDP", "PYUSD", "FDUSD", "GUSD", "LUSD", "FRAX"
}


def _is_stablecoin_symbol(sym: str) -> bool:
    s = str(sym or "").upper().strip()
    if not s:
        return False
    base = s.split("/")[0] if "/" in s else s
    return base in _STABLECOIN_BASES


@dataclass(frozen=True)
class CryptoUniverseItem:
    symbol: str
    name: str


def list_tradable_crypto(trading_client) -> list[CryptoUniverseItem]:
    req = GetAssetsRequest(asset_class=AssetClass.CRYPTO)
    assets = trading_client.get_all_assets(req)

    out: list[CryptoUniverseItem] = []
    for a in assets:
        if getattr(a, "status", None) != "active":
            continue
        if not getattr(a, "tradable", False):
            continue
        sym = getattr(a, "symbol", None)
        if not sym:
            continue
        if _is_stablecoin_symbol(sym):
            continue
        out.append(CryptoUniverseItem(symbol=sym, name=getattr(a, "name", "") or ""))

    # de-dupe
    seen: set[str] = set()
    deduped: list[CryptoUniverseItem] = []
    for it in out:
        if it.symbol in seen:
            continue
        seen.add(it.symbol)
        deduped.append(it)
    return deduped
