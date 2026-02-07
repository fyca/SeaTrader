from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    side: str  # buy|sell
    notional_usd: float
    asset_class: str
    reason: str


def diff_to_orders(
    *,
    current_notional: dict[str, float],
    targets: dict[str, float],
    asset_class_by_symbol: dict[str, str],
    min_trade_usd: float = 25.0,
) -> list[OrderPlan]:
    """Compute notional buys/sells to move current -> targets.

    NOTE: This is simplistic: doesn't account for min share qty, price, etc.
    """
    plans: list[OrderPlan] = []

    syms = set(current_notional) | set(targets)
    for s in sorted(syms):
        cur = float(current_notional.get(s, 0.0))
        tgt = float(targets.get(s, 0.0))
        delta = tgt - cur
        if abs(delta) < min_trade_usd:
            continue
        if delta > 0:
            plans.append(OrderPlan(symbol=s, side="buy", notional_usd=delta, asset_class=asset_class_by_symbol.get(s, "unknown"), reason="rebalance"))
        else:
            plans.append(OrderPlan(symbol=s, side="sell", notional_usd=abs(delta), asset_class=asset_class_by_symbol.get(s, "unknown"), reason="rebalance"))

    return plans
