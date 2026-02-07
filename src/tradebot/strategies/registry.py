from __future__ import annotations

from tradebot.strategies.baseline_trendvol import BaselineTrendVolStrategy
from tradebot.strategies.regime_filtered import RegimeFilteredTrendVolStrategy
from tradebot.strategies.breakout_trend import BreakoutTrendStrategy
from tradebot.strategies.pullback_trend import PullbackInTrendStrategy


_STRATS = {
    BaselineTrendVolStrategy.id: BaselineTrendVolStrategy(),
    RegimeFilteredTrendVolStrategy.id: RegimeFilteredTrendVolStrategy(),
    BreakoutTrendStrategy.id: BreakoutTrendStrategy(),
    PullbackInTrendStrategy.id: PullbackInTrendStrategy(),
}


def list_strategies() -> list[dict]:
    return [{"id": s.id, "name": s.name} for s in _STRATS.values()]


def get_strategy(strategy_id: str):
    if not strategy_id:
        return _STRATS[BaselineTrendVolStrategy.id]
    if strategy_id not in _STRATS:
        raise KeyError(f"Unknown strategy_id: {strategy_id}")
    return _STRATS[strategy_id]
