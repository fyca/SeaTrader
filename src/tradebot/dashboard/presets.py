from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# NOTE: Backtest UI uses unified presets file now (config/presets.yaml).
# We keep a thin wrapper here for backwards compatibility with existing endpoints.
from tradebot.util.presets import PRESETS_PATH


def load_presets() -> list[dict]:
    from tradebot.util.presets import load_presets as _lp

    # For backtest UI, return legacy shape: {name, params}
    out = []
    for p in _lp():
        out.append({"name": p.get("name"), "params": p.get("backtest") or {}})
    return out


def _bt_to_bot_patch(params: dict[str, Any]) -> dict[str, Any]:
    """Map backtest params -> live config patch for overlapping knobs."""
    params = params or {}
    bot: dict[str, Any] = {}

    if params.get("strategy_id"):
        bot["strategy_id"] = params.get("strategy_id")

    # risk knobs shared/parity
    if params.get("per_asset_stop_loss_pct") is not None:
        bot.setdefault("risk", {})
        bot["risk"]["per_asset_stop_loss_pct"] = params.get("per_asset_stop_loss_pct")
    if params.get("portfolio_dd_stop") is not None:
        bot.setdefault("risk", {})
        bot["risk"]["portfolio_dd_stop"] = params.get("portfolio_dd_stop")

    # rebalance behavior parity
    bot.setdefault("rebalance", {})
    if params.get("rebalance_mode"):
        bot["rebalance"]["rebalance_mode"] = params.get("rebalance_mode")
    if params.get("rebalance_day"):
        bot.setdefault("scheduling", {})
        bot["scheduling"]["weekly_rebalance_day"] = str(params.get("rebalance_day")).upper()
    if params.get("liquidation_mode"):
        bot["rebalance"]["liquidation_mode"] = params.get("liquidation_mode")
    if params.get("symbol_pnl_floor_pct") is not None:
        bot["rebalance"]["symbol_pnl_floor_pct"] = params.get("symbol_pnl_floor_pct")
    if params.get("symbol_pnl_floor_liquidate") is not None:
        bot["rebalance"]["symbol_pnl_floor_liquidate"] = bool(params.get("symbol_pnl_floor_liquidate"))
    if params.get("symbol_pnl_floor_include_unrealized") is not None:
        bot["rebalance"]["symbol_pnl_floor_include_unrealized"] = bool(params.get("symbol_pnl_floor_include_unrealized"))

    # execution mode mapping for live
    bot.setdefault("execution", {})
    if params.get("use_limit_orders") is not None:
        bot["execution"]["use_limit_orders"] = bool(params.get("use_limit_orders"))
    if params.get("limit_offset_bps") is not None:
        bot["execution"]["limit_offset_bps"] = float(params.get("limit_offset_bps"))
    if params.get("limit_fallback_to_market_open") is not None:
        bot["execution"]["fallback_to_market_at_open"] = bool(params.get("limit_fallback_to_market_open"))
    if params.get("limit_fallback_time_local") is not None:
        bot["execution"]["fallback_time_local"] = str(params.get("limit_fallback_time_local"))

    # per-asset execution overrides
    bot["execution"].setdefault("equities", {})
    bot["execution"].setdefault("crypto", {})
    if params.get("order_type_equities"):
        bot["execution"]["equities"]["order_type"] = str(params.get("order_type_equities"))
    if params.get("order_type_crypto"):
        bot["execution"]["crypto"]["order_type"] = str(params.get("order_type_crypto"))
    if params.get("limit_offset_bps_equities") is not None:
        bot["execution"]["equities"]["limit_offset_bps"] = float(params.get("limit_offset_bps_equities"))
    if params.get("limit_offset_bps_crypto") is not None:
        bot["execution"]["crypto"]["limit_offset_bps"] = float(params.get("limit_offset_bps_crypto"))

    # map backtest risk-check intraday time to live daily risk-check schedule time
    if params.get("risk_check_time_local"):
        bot.setdefault("scheduling", {})
        bot["scheduling"]["daily_risk_check_time_local"] = str(params.get("risk_check_time_local"))
    if params.get("risk_check_time_local_equities"):
        bot.setdefault("scheduling", {})
        bot["scheduling"].setdefault("equities", {})
        bot["scheduling"]["equities"]["risk_check_time_local"] = str(params.get("risk_check_time_local_equities"))
    if params.get("risk_check_time_local_crypto"):
        bot.setdefault("scheduling", {})
        bot["scheduling"].setdefault("crypto", {})
        bot["scheduling"]["crypto"]["risk_check_time_local"] = str(params.get("risk_check_time_local_crypto"))

    # per-asset schedule overrides
    bot.setdefault("scheduling", {})
    bot["scheduling"].setdefault("equities", {})
    bot["scheduling"].setdefault("crypto", {})

    if params.get("rebalance_frequency_equities"):
        bot["scheduling"]["equities"]["rebalance_frequency"] = str(params.get("rebalance_frequency_equities"))
    if params.get("rebalance_day_equities"):
        bot["scheduling"]["equities"]["rebalance_day"] = str(params.get("rebalance_day_equities")).upper()
    if params.get("rebalance_frequency_crypto"):
        bot["scheduling"]["crypto"]["rebalance_frequency"] = str(params.get("rebalance_frequency_crypto"))
    if params.get("rebalance_day_crypto"):
        bot["scheduling"]["crypto"]["rebalance_day"] = str(params.get("rebalance_day_crypto")).upper()
    if params.get("risk_check_frequency_equities"):
        bot["scheduling"]["equities"]["risk_check_frequency"] = str(params.get("risk_check_frequency_equities"))
    if params.get("risk_check_day_equities"):
        bot["scheduling"]["equities"]["risk_check_day"] = str(params.get("risk_check_day_equities")).upper()
    if params.get("risk_check_frequency_crypto"):
        bot["scheduling"]["crypto"]["risk_check_frequency"] = str(params.get("risk_check_frequency_crypto"))
    if params.get("risk_check_day_crypto"):
        bot["scheduling"]["crypto"]["risk_check_day"] = str(params.get("risk_check_day_crypto")).upper()

    # execution timing mapping (best-effort): sets unattended scheduler times.
    exec_mode = params.get("execution_time_mode") or "intraday"
    bot.setdefault("scheduling", {})
    bot["scheduling"].setdefault("equities", {})
    bot["scheduling"].setdefault("crypto", {})

    if exec_mode == "intraday":
        t = params.get("execution_time_local")
        t_eq = params.get("execution_time_local_equities") or t
        t_cr = params.get("execution_time_local_crypto") or t
        if t:
            bot["scheduling"]["weekly_rebalance_time_local"] = str(t)
        if t_eq:
            bot["scheduling"]["equities"]["rebalance_time_local"] = str(t_eq)
        if t_cr:
            bot["scheduling"]["crypto"]["rebalance_time_local"] = str(t_cr)
        bot["scheduling"]["timezone"] = str(params.get("execution_tz") or "America/Los_Angeles")
    else:
        et = params.get("execution_time") or "close"
        t = "06:35" if et == "open" else "12:55"
        bot["scheduling"]["weekly_rebalance_time_local"] = t
        bot["scheduling"].setdefault("equities", {}).setdefault("rebalance_time_local", t)
        bot["scheduling"].setdefault("crypto", {}).setdefault("rebalance_time_local", t)
        bot["scheduling"]["timezone"] = "America/Los_Angeles"

    return bot


def save_preset(name: str, params: dict[str, Any]) -> None:
    # Backtest UI save: persists as unified preset {bot, backtest}
    from tradebot.util.presets import save_preset as _sp

    bot_patch = _bt_to_bot_patch(params)
    _sp(name=name, bot=bot_patch, backtest=params)


def get_preset(name: str) -> dict | None:
    for p in load_presets():
        if str(p.get("name")) == str(name):
            return p
    return None
