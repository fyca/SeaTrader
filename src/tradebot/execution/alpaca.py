from __future__ import annotations

from dataclasses import dataclass

from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest


@dataclass(frozen=True)
class PlacedOrder:
    symbol: str
    side: str
    notional_usd: float
    id: str
    order_type: str
    expected_price: float | None = None
    limit_price: float | None = None


def place_notional_market_orders(
    trading_client,
    plans,
    *,
    use_limit_orders: bool = False,
    limit_offset_bps: float = 10.0,
    ref_price_by_symbol: dict[str, float] | None = None,
    extended_hours: bool = False,
) -> list[PlacedOrder]:
    """Place notional orders (market by default; optional limit with offset)."""
    out: list[PlacedOrder] = []
    ref_price_by_symbol = ref_price_by_symbol or {}

    for pl in plans:
        side = OrderSide.BUY if pl.side == "buy" else OrderSide.SELL

        if use_limit_orders:
            ref = float(ref_price_by_symbol.get(pl.symbol, 0.0) or 0.0)
            if ref > 0:
                mul = (1 + limit_offset_bps / 10000.0) if pl.side == "buy" else (1 - limit_offset_bps / 10000.0)
                raw_lim = ref * mul
                # Alpaca min pricing increments:
                # - >= $1.00 => max 2 decimals
                # - <  $1.00 => max 4 decimals
                lim = round(raw_lim, 2 if raw_lim >= 1 else 4)
                req = LimitOrderRequest(
                    symbol=pl.symbol,
                    notional=round(float(pl.notional_usd), 2),
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force=TimeInForce.DAY,
                    limit_price=lim,
                    extended_hours=(extended_hours and ("/" not in pl.symbol)),
                )
                o = trading_client.submit_order(req)
                out.append(
                    PlacedOrder(
                        symbol=pl.symbol,
                        side=pl.side,
                        notional_usd=float(pl.notional_usd),
                        id=str(getattr(o, "id", "")),
                        order_type="limit",
                        expected_price=ref,
                        limit_price=lim,
                    )
                )
                continue

        req = MarketOrderRequest(
            symbol=pl.symbol,
            notional=round(float(pl.notional_usd), 2),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        o = trading_client.submit_order(req)
        out.append(
            PlacedOrder(
                symbol=pl.symbol,
                side=pl.side,
                notional_usd=float(pl.notional_usd),
                id=str(getattr(o, "id", "")),
                order_type="market",
                expected_price=(ref_price_by_symbol.get(pl.symbol) if pl.symbol in ref_price_by_symbol else None),
                limit_price=None,
            )
        )
    return out
