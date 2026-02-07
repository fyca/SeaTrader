from __future__ import annotations

from dataclasses import dataclass

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


@dataclass(frozen=True)
class PlacedOrder:
    symbol: str
    side: str
    notional_usd: float
    id: str


def place_notional_market_orders(trading_client, plans) -> list[PlacedOrder]:
    """Place notional market orders (paper/live depending on client)."""
    out: list[PlacedOrder] = []
    for pl in plans:
        req = MarketOrderRequest(
            symbol=pl.symbol,
            notional=round(float(pl.notional_usd), 2),
            side=OrderSide.BUY if pl.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        o = trading_client.submit_order(req)
        out.append(PlacedOrder(symbol=pl.symbol, side=pl.side, notional_usd=float(pl.notional_usd), id=str(getattr(o, "id", ""))))
    return out
