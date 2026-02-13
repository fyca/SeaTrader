from __future__ import annotations

import argparse

from rich import print
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from tradebot.adapters.alpaca_client import make_alpaca_clients
from tradebot.adapters.bars import fetch_stock_bars, fetch_crypto_bars
from tradebot.risk.drawdown import update_drawdown_state
from tradebot.risk.exits import trend_break_exit
from tradebot.util.config import load_config
from tradebot.strategies.registry import get_strategy
from tradebot.util.env import load_env
from tradebot.util.state import load_state, save_state
from tradebot.util.artifacts import write_artifact
from tradebot.util.equity_curve import append_equity_point


def cmd_risk_check(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, preset_override=getattr(args, "preset", None))
    run_asset_mode = str(getattr(args, "asset_mode", None) or "both").lower()
    if run_asset_mode not in ("both", "equities", "crypto"):
        run_asset_mode = "both"
    env = load_env()
    clients = make_alpaca_clients(env)

    acct = clients.trading.get_account()
    equity = float(acct.equity)

    state = load_state()
    dd_trigger = cfg.risk.portfolio_dd_stop if cfg.risk.portfolio_dd_stop is not None else cfg.risk.max_drawdown_freeze
    dd_state = update_drawdown_state(prior_peak_equity=state.peak_equity, current_equity=equity, freeze_at=dd_trigger)
    state.peak_equity = dd_state.peak_equity
    save_state(state)

    print(f"Equity: {equity:.2f}  Peak: {dd_state.peak_equity:.2f}  Drawdown: {dd_state.drawdown:.1%}")
    if dd_state.frozen:
        print(f"[red]FROZEN[/red] (>= {cfg.risk.max_drawdown_freeze:.0%}) -> should not open new positions")
    else:
        print("OK")

    # Exits-only logic: if a held position breaks trend OR hits stop-loss, propose a SELL (dry-run only)
    positions = clients.trading.get_all_positions()
    held = [p.symbol for p in positions if float(p.qty) != 0.0]

    eq_syms = [s for s in held if "/" not in s]
    cr_syms = [s for s in held if "/" in s]
    if run_asset_mode == "equities":
        cr_syms = []
    elif run_asset_mode == "crypto":
        eq_syms = []

    exit_plans = []

    stop_pct = cfg.risk.per_asset_stop_loss_pct
    if stop_pct is not None:
        stop_pct = float(stop_pct)

    strat = None
    try:
        strat = get_strategy(cfg.strategy_id)
    except Exception:
        strat = None

    # If user strategy has an exit rule, we'll evaluate it (in addition to stop loss / trend break)
    user_exit = None
    if strat is not None and hasattr(strat, "spec"):
        user_exit = getattr(strat, "spec", {}).get("exit")

    if eq_syms:
        eq_bars = fetch_stock_bars(clients.stocks, eq_syms, lookback_days=cfg.signals.lookback_days)
        for sym, df in eq_bars.items():
            if df is None or len(df) == 0 or "close" not in df.columns:
                continue
            closes = df["close"].dropna()
            if len(closes) == 0:
                continue
            last_px = float(closes.iloc[-1])

            # user exit rule (if present)
            if user_exit:
                try:
                    from tradebot.strategies.rule_engine import EvalContext, eval_rule
                    ctx = EvalContext(closes=closes, ann_factor=252.0)
                    if eval_rule(ctx, user_exit):
                        exit_plans.append({"symbol": sym, "asset_class": "equity", "reason": "user_exit_rule", "last_close": last_px})
                        continue
                except Exception:
                    pass

            # stop-loss check from avg entry
            if stop_pct is not None:
                # pull avg entry from Alpaca position
                pos = next((p for p in positions if p.symbol == sym), None)
                if pos is not None:
                    avg_entry = float(pos.avg_entry_price)
                    stop_level = avg_entry * (1 - stop_pct)
                    if last_px <= stop_level:
                        exit_plans.append({"symbol": sym, "asset_class": "equity", "reason": f"stop_loss_{int(stop_pct*100)}%", "last_close": last_px, "stop_level": stop_level, "avg_entry": avg_entry})
                        continue

            should, reason, last, maL = trend_break_exit(closes, ma_long=cfg.signals.equity.ma_long)
            if should:
                exit_plans.append({"symbol": sym, "asset_class": "equity", "reason": reason, "last_close": last, "ma_long": maL})

    if cr_syms:
        cr_bars = fetch_crypto_bars(clients.crypto, cr_syms, lookback_days=cfg.signals.lookback_days)
        for sym, df in cr_bars.items():
            if df is None or len(df) == 0 or "close" not in df.columns:
                continue
            closes = df["close"].dropna()
            if len(closes) == 0:
                continue
            last_px = float(closes.iloc[-1])

            # user exit rule
            if user_exit:
                try:
                    from tradebot.strategies.rule_engine import EvalContext, eval_rule
                    ctx = EvalContext(closes=closes, ann_factor=365.0)
                    if eval_rule(ctx, user_exit):
                        exit_plans.append({"symbol": sym, "asset_class": "crypto", "reason": "user_exit_rule", "last_close": last_px})
                        continue
                except Exception:
                    pass

            if stop_pct is not None:
                pos = next((p for p in positions if p.symbol == sym), None)
                if pos is not None:
                    avg_entry = float(pos.avg_entry_price)
                    stop_level = avg_entry * (1 - stop_pct)
                    if last_px <= stop_level:
                        exit_plans.append({"symbol": sym, "asset_class": "crypto", "reason": f"stop_loss_{int(stop_pct*100)}%", "last_close": last_px, "stop_level": stop_level, "avg_entry": avg_entry})
                        continue

            should, reason, last, maL = trend_break_exit(closes, ma_long=cfg.signals.crypto.ma_long)
            if should:
                exit_plans.append({"symbol": sym, "asset_class": "crypto", "reason": reason, "last_close": last, "ma_long": maL})

    if exit_plans:
        print("\nExit signals:")
        for e in exit_plans:
            print(f"- SELL {e['symbol']:12s} ({e['asset_class']}) reason={e['reason']}")

    executed_liquidations = []
    if exit_plans and bool(getattr(cfg.risk, "execute_exit_liquidations", False)):
        # de-dup by symbol to avoid double-ordering the same asset
        pos_by_symbol = {p.symbol: p for p in positions}
        seen: set[str] = set()
        for e in exit_plans:
            sym = str(e.get("symbol") or "").strip()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            pos = pos_by_symbol.get(sym)
            if pos is None:
                continue
            try:
                qty = abs(float(pos.qty))
            except Exception:
                qty = 0.0
            if qty <= 0:
                continue
            if bool(getattr(cfg, "dry_run", False)):
                executed_liquidations.append({
                    "symbol": sym,
                    "qty": qty,
                    "side": "sell",
                    "status": "skipped_dry_run",
                    "reason": e.get("reason"),
                })
                continue
            tif = TimeInForce.GTC if "/" in sym else TimeInForce.DAY
            try:
                req = MarketOrderRequest(symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=tif)
                o = clients.trading.submit_order(req)
                executed_liquidations.append({
                    "symbol": sym,
                    "qty": qty,
                    "side": "sell",
                    "status": "submitted",
                    "order_id": str(getattr(o, "id", "")),
                    "reason": e.get("reason"),
                })
                print(f"[green]Submitted SELL[/green] {sym} qty={qty}")
            except Exception as ex:
                executed_liquidations.append({
                    "symbol": sym,
                    "qty": qty,
                    "side": "sell",
                    "status": "error",
                    "error": str(ex),
                    "reason": e.get("reason"),
                })
                print(f"[red]Failed SELL[/red] {sym}: {ex}")

    write_artifact(
        "last_risk_check.json",
        {
            "equity": equity,
            "peak_equity": dd_state.peak_equity,
            "drawdown": dd_state.drawdown,
            "frozen": dd_state.frozen,
            "exit_signals": exit_plans,
            "execute_exit_liquidations": bool(getattr(cfg.risk, "execute_exit_liquidations", False)),
            "executed_liquidations": executed_liquidations,
        },
    )
    append_equity_point(equity=equity, cash=float(getattr(acct, "cash", 0.0) or 0.0))

    return 0
