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


def list_strategies(*, strategy_type: str | None = None, asset_class: str | None = None, include_legacy: bool = True) -> list[dict]:
    out = [
        {
            "id": s.id,
            "name": s.name,
            "source": "builtin",
            "version": 1,
            "type": None,
            "asset_class": None,
            "legacy_untyped": True,
        }
        for s in _STRATS.values()
    ]
    out.extend(list_user_strategies())

    def _ok(x: dict) -> bool:
        if (not include_legacy) and bool(x.get("legacy_untyped")):
            return False
        if strategy_type and str(x.get("type") or "") != strategy_type:
            return False
        if asset_class and str(x.get("asset_class") or "") != asset_class:
            return False
        return True

    return [x for x in out if _ok(x)]


def get_strategy(strategy_id: str):
    if not strategy_id:
        return _STRATS[BaselineTrendVolStrategy.id]
    if strategy_id in _STRATS:
        return _STRATS[strategy_id]
    # user strategy
    spec = load_user_strategy(strategy_id)
    return RuleBasedStrategy(spec)
