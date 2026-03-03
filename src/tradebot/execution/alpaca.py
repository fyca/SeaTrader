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
    asset_class: str
    expected_price: float | None = None
    limit_price: float | None = None
    qty: float | None = None


def place_notional_market_orders(
    trading_client,
    plans,
    *,
    use_limit_orders: bool = False,
    limit_offset_bps: float = 10.0,
    ref_price_by_symbol: dict[str, float] | None = None,
    extended_hours: bool = False,
    symbol_order_type: dict[str, str] | None = None,
    symbol_limit_offset_bps: dict[str, float] | None = None,
    symbol_sell_qty: dict[str, float] | None = None,
) -> list[PlacedOrder]:
    """Place notional orders (market by default; optional limit with offset).

    Per-symbol overrides via symbol_order_type/symbol_limit_offset_bps allow
    different behavior for equities vs crypto.
    
    - For sells: uses exact qty from symbol_sell_qty (position qty)
    - For buys: reduces notional to fit within available buying power
    """
    out: list[PlacedOrder] = []
    ref_price_by_symbol = ref_price_by_symbol or {}
    symbol_order_type = symbol_order_type or {}
    symbol_limit_offset_bps = symbol_limit_offset_bps or {}
    symbol_sell_qty = symbol_sell_qty or {}

    # Get available buying power
    try:
        account = trading_client.get_account()
        available_bp = float(getattr(account, "buying_power", 0.0) or 0.0)
    except Exception:
        available_bp = float('inf')  # If we can't get it, assume unlimited
    
    remaining_bp = available_bp

    for pl in plans:
        side = OrderSide.BUY if pl.side == "buy" else OrderSide.SELL
        # Detect crypto: either has "/" or asset_class is "crypto" (or "unknown" + USD suffix)
        is_crypto = "/" in pl.symbol or pl.asset_class == "crypto" or (pl.asset_class == "unknown" and pl.symbol.endswith("USD"))
        ord_type = str(symbol_order_type.get(pl.symbol, "")).lower()
        use_limit_for_symbol = (ord_type == "limit") if ord_type in ("market", "limit") else bool(use_limit_orders)
        off_bps = float(symbol_limit_offset_bps.get(pl.symbol, limit_offset_bps))

        qty_override = float(symbol_sell_qty.get(pl.symbol, 0.0) or 0.0) if pl.side == "sell" else 0.0

        # For sells: always use exact qty from position
        if pl.side == "sell":
            actual_notional = float(pl.notional_usd)
        else:
            # For buys: reduce notional to fit within remaining buying power
            actual_notional = min(float(pl.notional_usd), remaining_bp)
            if actual_notional <= 0:
                print(f"[order-skip] {pl.symbol} buy: no remaining buying power")
                continue

        if use_limit_for_symbol:
            ref = float(ref_price_by_symbol.get(pl.symbol, 0.0) or 0.0)
            if ref > 0:
                mul = (1 + off_bps / 10000.0) if pl.side == "buy" else (1 - off_bps / 10000.0)
                raw_lim = ref * mul
                # Alpaca min pricing increments:
                # - >= $1.00 => max 2 decimals
                # - <  $1.00 => max 4 decimals
                lim = round(raw_lim, 2 if raw_lim >= 1 else 4)
                
                req_kwargs = dict(
                    symbol=pl.symbol,
                    side=side,
                    type=OrderType.LIMIT,
                    limit_price=lim,
                )
                # Crypto limit orders: use GTC; equities: use DAY
                if is_crypto:
                    req_kwargs["time_in_force"] = TimeInForce.GTC
                else:
                    req_kwargs["time_in_force"] = TimeInForce.DAY
                    req_kwargs["extended_hours"] = (extended_hours and ("/" not in pl.symbol))
                
                # Determine qty vs notional
                if pl.side == "sell" and qty_override > 0:
                    # Sell: use exact qty from position
                    req_kwargs["qty"] = qty_override
                    used_qty = qty_override
                elif is_crypto:
                    # Crypto buy: calculate qty from reference price
                    if ref > 0:
                        calc_qty = actual_notional / ref
                        req_kwargs["qty"] = round(calc_qty, 8)  # 8 decimals for crypto
                        used_qty = calc_qty
                    else:
                        print(f"[order-skip] {pl.symbol} {pl.side} limit: no ref price for crypto qty calc")
                        continue
                else:
                    # Equity buy: use notional
                    req_kwargs["notional"] = round(actual_notional, 2)
                    used_qty = None
                
                req = LimitOrderRequest(**req_kwargs)
                try:
                    o = trading_client.submit_order(req)
                    remaining_bp -= actual_notional
                except Exception as e:
                    msg = str(e).lower()
                    if pl.side == "sell" and qty_override > 0 and ("insufficient qty" in msg or "insufficient" in msg):
                        # Retry with 99.9% qty
                        try:
                            retry_qty = qty_override * 0.999
                            req_kwargs["qty"] = retry_qty
                            req = LimitOrderRequest(**req_kwargs)
                            o = trading_client.submit_order(req)
                            used_qty = retry_qty
                            remaining_bp -= actual_notional
                        except Exception as ex2:
                            print(f"[order-retry-fail] {pl.symbol} {pl.side} limit (qty retry): {ex2}")
                            continue
                    elif pl.side == "buy" and ("insufficient buying power" in msg or "40310000" in msg):
                        # Reduce notional and retry
                        reduced_notional = actual_notional * 0.95  # Try 95% of amount
                        if reduced_notional < 10:  # Skip if too small
                            print(f"[order-skip] {pl.symbol} {pl.side} limit: insufficient buying power (reduced amount too small)")
                            continue
                        try:
                            if is_crypto and ref > 0:
                                reduced_qty = reduced_notional / ref
                                req_kwargs["qty"] = round(reduced_qty, 8)
                            else:
                                req_kwargs["notional"] = round(reduced_notional, 2)
                            req = LimitOrderRequest(**req_kwargs)
                            o = trading_client.submit_order(req)
                            actual_notional = reduced_notional
                            remaining_bp -= reduced_notional
                            print(f"[order-retry-reduced] {pl.symbol} {pl.side} limit: reduced to ${reduced_notional:.2f}")
                        except Exception as ex2:
                            print(f"[order-retry-fail] {pl.symbol} {pl.side} limit (after reduction): {ex2}")
                            continue
                    else:
                        print(f"[order-fail] {pl.symbol} {pl.side} limit: {e}")
                        continue
                
                out.append(
                    PlacedOrder(
                        symbol=pl.symbol,
                        side=pl.side,
                        notional_usd=actual_notional,
                        id=str(getattr(o, "id", "")),
                        order_type="limit",
                        asset_class=pl.asset_class,
                        expected_price=ref,
                        limit_price=lim,
                        qty=used_qty,
                    )
                )
                continue

        # Market orders
        req_kwargs = dict(
            symbol=pl.symbol,
            side=side,
        )
        # time_in_force is required for MarketOrderRequest
        # Equities: DAY (standard trading hours)
        # Crypto: IOC (Immediate or Cancel - no overnight holding)
        if is_crypto:
            req_kwargs["time_in_force"] = TimeInForce.IOC
        else:
            req_kwargs["time_in_force"] = TimeInForce.DAY
        
        # Determine qty vs notional
        if pl.side == "sell" and qty_override > 0:
            # Sell: use exact qty from position
            req_kwargs["qty"] = qty_override
            used_qty = qty_override
        elif is_crypto:
            # Crypto buy: calculate qty from reference price
            ref = float(ref_price_by_symbol.get(pl.symbol, 0.0) or 0.0)
            if ref > 0:
                calc_qty = actual_notional / ref
                req_kwargs["qty"] = round(calc_qty, 8)  # 8 decimals for crypto
                used_qty = calc_qty
            else:
                print(f"[order-skip] {pl.symbol} {pl.side} market: no ref price for crypto qty calc")
                continue
        else:
            # Equity market: use notional
            req_kwargs["notional"] = round(actual_notional, 2)
            used_qty = None
        
        req = MarketOrderRequest(**req_kwargs)
        try:
            o = trading_client.submit_order(req)
            remaining_bp -= actual_notional
        except Exception as e:
            msg = str(e).lower()
            if pl.side == "sell" and qty_override > 0 and ("insufficient qty" in msg or "insufficient" in msg):
                # Retry with 99.9% qty
                try:
                    retry_qty = qty_override * 0.999
                    req_kwargs["qty"] = retry_qty
                    req = MarketOrderRequest(**req_kwargs)
                    o = trading_client.submit_order(req)
                    used_qty = retry_qty
                    remaining_bp -= actual_notional
                except Exception as ex2:
                    print(f"[order-retry-fail] {pl.symbol} {pl.side} market (qty retry): {ex2}")
                    continue
            elif pl.side == "buy" and ("insufficient buying power" in msg or "40310000" in msg):
                # Reduce notional and retry
                reduced_notional = actual_notional * 0.95  # Try 95% of amount
                if reduced_notional < 10:  # Skip if too small
                    print(f"[order-skip] {pl.symbol} {pl.side} market: insufficient buying power (reduced amount too small)")
                    continue
                try:
                    if is_crypto and ref > 0:
                        reduced_qty = reduced_notional / ref
                        req_kwargs["qty"] = round(reduced_qty, 8)
                    else:
                        req_kwargs["notional"] = round(reduced_notional, 2)
                    req = MarketOrderRequest(**req_kwargs)
                    o = trading_client.submit_order(req)
                    actual_notional = reduced_notional
                    remaining_bp -= reduced_notional
                    print(f"[order-retry-reduced] {pl.symbol} {pl.side} market: reduced to ${reduced_notional:.2f}")
                except Exception as ex2:
                    print(f"[order-retry-fail] {pl.symbol} {pl.side} market (after reduction): {ex2}")
                    continue
            else:
                print(f"[order-fail] {pl.symbol} {pl.side} market: {e}")
                continue
        
        out.append(
            PlacedOrder(
                symbol=pl.symbol,
                side=pl.side,
                notional_usd=actual_notional,
                id=str(getattr(o, "id", "")),
                order_type="market",
                asset_class=pl.asset_class,
                expected_price=(ref_price_by_symbol.get(pl.symbol) if pl.symbol in ref_price_by_symbol else None),
                limit_price=None,
                qty=used_qty,
            )
        )
    
    return out
