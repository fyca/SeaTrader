from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    symbol: str
    notional_usd: float
    asset_class: str  # 'equity'|'crypto'


def build_equal_weight_targets(
    *,
    equity_symbols: list[str],
    crypto_symbols: list[str],
    equity_budget: float,
    crypto_budget: float,
) -> list[Target]:
    targets: list[Target] = []

    if equity_symbols:
        w = equity_budget / len(equity_symbols)
        for s in equity_symbols:
            targets.append(Target(symbol=s, notional_usd=w, asset_class="equity"))

    if crypto_symbols:
        w = crypto_budget / len(crypto_symbols)
        for s in crypto_symbols:
            targets.append(Target(symbol=s, notional_usd=w, asset_class="crypto"))

    return targets
