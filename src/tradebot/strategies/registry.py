from __future__ import annotations

from tradebot.strategies.baseline_trendvol import BaselineTrendVolStrategy
from tradebot.strategies.regime_filtered import RegimeFilteredTrendVolStrategy
from tradebot.strategies.breakout_trend import BreakoutTrendStrategy
from tradebot.strategies.pullback_trend import PullbackInTrendStrategy
from tradebot.strategies.user_store import list_user_strategies, load_user_strategy
from tradebot.strategies.rule_based import RuleBasedStrategy


_STRATS = {
    BaselineTrendVolStrategy.id: BaselineTrendVolStrategy(),
    RegimeFilteredTrendVolStrategy.id: RegimeFilteredTrendVolStrategy(),
    BreakoutTrendStrategy.id: BreakoutTrendStrategy(),
    PullbackInTrendStrategy.id: PullbackInTrendStrategy(),
}


def list_strategies() -> list[dict]:
    out = [{"id": s.id, "name": s.name, "source": "builtin"} for s in _STRATS.values()]
    for u in list_user_strategies():
        out.append({"id": u["id"], "name": u.get("name") or u["id"], "source": "user"})
    return out


def get_strategy(strategy_id: str):
    if not strategy_id:
        return _STRATS[BaselineTrendVolStrategy.id]
    if strategy_id in _STRATS:
        return _STRATS[strategy_id]
    # user strategy
    spec = load_user_strategy(strategy_id)
    return RuleBasedStrategy(spec)
