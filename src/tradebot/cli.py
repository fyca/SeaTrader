from __future__ import annotations

import argparse
from pathlib import Path

from rich import print

from tradebot.util.config import load_config
from tradebot.util.env import load_env
from tradebot.adapters.alpaca_client import make_alpaca_clients
from tradebot.adapters.bars import fetch_stock_bars, fetch_crypto_bars
from tradebot.universe.liquidity import avg_dollar_volume
from tradebot.universe.equities import list_tradable_equities
from tradebot.universe.crypto import list_tradable_crypto
from tradebot.strategies.registry import get_strategy
from tradebot.portfolio.targets import build_equal_weight_targets
from tradebot.execution.plan import OrderPlan, diff_to_orders
from tradebot.execution.alpaca import place_notional_market_orders
from tradebot.execution.guardrails import check_order_guardrails
from tradebot.risk.drawdown import update_drawdown_state
from tradebot.util.state import load_state, save_state
from tradebot.commands.risk_check import cmd_risk_check
from tradebot.commands.dashboard import cmd_dashboard
from tradebot.util.artifacts import write_artifact
from tradebot.util.equity_curve import append_equity_point


def cmd_rebalance(args: argparse.Namespace) -> int:
    # Optional preset override
    cfg = load_config(args.config, preset_override=getattr(args, "preset", None))

    # Optional unattended scheduling: wait until configured/local time (rebalance only).
    if args.wait_until is not None:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import time as _time

        tz = ZoneInfo(getattr(cfg.scheduling, "timezone", "America/Los_Angeles"))
        now = datetime.now(tz)
        hh, mm = [int(x) for x in str(args.wait_until).split(":")]
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target > now:
            secs = (target - now).total_seconds()
            print(f"Waiting until {target.strftime('%Y-%m-%d %H:%M %Z')} (sleep {int(secs)}s)...")
            _time.sleep(secs)
        else:
            print(f"[yellow]Warning[/yellow]: wait_until {args.wait_until} is in the past (now {now.strftime('%H:%M %Z')}); running immediately")
    env = load_env()

    # config can force dry_run; env can also force dry_run
    dry_run = bool(cfg.dry_run or env.dry_run)

    clients = make_alpaca_clients(env)

    acct = clients.trading.get_account()
    equity = float(acct.equity)
    cash = float(acct.cash)

    print(f"Mode: [bold]{cfg.mode}[/bold]  Paper(env): {env.paper}  DRY_RUN: {dry_run}")
    print(f"Account: {acct.account_number}  Equity: {equity:.2f}  Cash: {cash:.2f}")

    # Drawdown / freeze state
    state = load_state()
    dd_trigger = cfg.risk.portfolio_dd_stop if cfg.risk.portfolio_dd_stop is not None else cfg.risk.max_drawdown_freeze
    dd_state = update_drawdown_state(prior_peak_equity=state.peak_equity, current_equity=equity, freeze_at=dd_trigger)
    state.peak_equity = dd_state.peak_equity
    save_state(state)
    if dd_state.frozen:
        print(f"[red]DD STOP[/red]: drawdown={dd_state.drawdown:.1%} (>= {dd_trigger:.0%}) behavior={cfg.risk.dd_stop_behavior}")
    elif dd_state.drawdown >= cfg.risk.warn_drawdown:
        print(f"[yellow]Warning[/yellow]: drawdown={dd_state.drawdown:.1%}")

    # Save a lightweight snapshot for the dashboard
    write_artifact(
        "last_account.json",
        {
            "account_number": acct.account_number,
            "equity": equity,
            "cash": cash,
            "paper": env.paper,
            "dry_run": dry_run,
            "drawdown": dd_state.drawdown,
            "frozen": dd_state.frozen,
        },
    )
    append_equity_point(equity=equity, cash=cash)

    # 1) Universes
    eq_univ = list_tradable_equities(clients.trading, exclude_leveraged_etfs=cfg.universe.exclude_leveraged_etfs)
    cr_univ = list_tradable_crypto(clients.trading)

    # Prefer a stable, liquid universe: S&P 500 intersection with Alpaca-tradable assets.
    try:
        from tradebot.universe.sp500 import get_sp500_symbols

        sp500 = set(get_sp500_symbols())
    except Exception as e:
        print(f"[yellow]Warning:[/yellow] failed to fetch S&P 500 list ({e}); falling back to Alpaca assets")
        sp500 = set()

    eq_all = [x.symbol for x in eq_univ]
    if sp500:
        eq_symbols = [s for s in eq_all if s in sp500]
    else:
        eq_symbols = eq_all

    # Crypto: default to USD-quoted pairs only
    cr_all = cfg.universe.crypto_symbols_allowlist or [x.symbol for x in cr_univ]
    cr_symbols = [s for s in cr_all if s.endswith("/USD")]

    # Limit data pulls to manageable sizes
    eq_symbols = eq_symbols[:500]
    cr_symbols = cr_symbols[:200]

    print(f"Universe candidates: equities={len(eq_symbols)} crypto={len(cr_symbols)}")

    # 2) Data
    eq_bars = fetch_stock_bars(clients.stocks, eq_symbols, lookback_days=cfg.signals.lookback_days)
    cr_bars = fetch_crypto_bars(clients.crypto, cr_symbols, lookback_days=cfg.signals.lookback_days)

    # 3) Strategy selection
    strat = get_strategy(cfg.strategy_id)

    eq_sel, eq_sig_details = strat.select_equities(bars=eq_bars, cfg=cfg)
    cr_sel, cr_sig_details = strat.select_crypto(bars=cr_bars, cfg=cfg)

    print(f"Selected: equities={len(eq_sel)} crypto={len(cr_sel)}")
    if eq_sel:
        print("Equities:", ", ".join(eq_sel))
    if cr_sel:
        print("Crypto:", ", ".join(cr_sel))

    # Apply symbol exclusion floor (parity with backtest, unrealized-based in live).
    excluded = set([str(s).upper() for s in (state.excluded_symbols or [])])
    floor = cfg.rebalance.symbol_pnl_floor_pct
    if floor is not None:
        for p in clients.trading.get_all_positions():
            sym = str(getattr(p, "symbol", "") or "").upper()
            if not sym:
                continue
            plpc = float(getattr(p, "unrealized_plpc", 0.0) or 0.0) if cfg.rebalance.symbol_pnl_floor_include_unrealized else 0.0
            if plpc <= float(floor):
                excluded.add(sym)

    if excluded:
        eq_sel = [s for s in eq_sel if str(s).upper() not in excluded]
        cr_sel = [s for s in cr_sel if str(s).upper() not in excluded]

    # Persist exclusions across runs
    state.excluded_symbols = sorted(excluded)
    save_state(state)

    # 4) Targets
    equity_budget = equity * cfg.allocation.equities
    crypto_budget = equity * cfg.allocation.crypto
    targets = build_equal_weight_targets(
        equity_symbols=eq_sel,
        crypto_symbols=cr_sel,
        equity_budget=equity_budget,
        crypto_budget=crypto_budget,
    )
    target_map = {t.symbol: float(t.notional_usd) for t in targets}
    class_map = {t.symbol: t.asset_class for t in targets}

    # 5) Current positions (notional)
    current_positions = clients.trading.get_all_positions()
    current_map: dict[str, float] = {}
    avg_entry_map: dict[str, float] = {}
    cur_price_map: dict[str, float] = {}
    for p in current_positions:
        sym = getattr(p, "symbol", "")
        mv = float(getattr(p, "market_value", 0.0) or 0.0)
        if sym:
            current_map[sym] = mv
            try:
                avg_entry_map[sym] = float(getattr(p, "avg_entry_price", 0.0) or 0.0)
            except Exception:
                pass
            try:
                cur_price_map[sym] = float(getattr(p, "current_price", 0.0) or 0.0)
            except Exception:
                pass

    # liquidation_mode parity
    if cfg.rebalance.liquidation_mode == "hold_until_exit":
        for sym, cur in current_map.items():
            if sym not in target_map:
                target_map[sym] = float(cur)
                class_map.setdefault(sym, "crypto" if "/" in sym else "equity")

    # immediate liquidation for excluded symbols
    if cfg.rebalance.symbol_pnl_floor_liquidate and excluded:
        for sym in excluded:
            if sym in current_map:
                target_map[sym] = 0.0
                class_map.setdefault(sym, "crypto" if "/" in sym else "equity")

    # 6) Plan orders
    plans = diff_to_orders(current_notional=current_map, targets=target_map, asset_class_by_symbol=class_map)

    # rebalance_mode parity: no_add_to_losers
    if cfg.rebalance.rebalance_mode == "no_add_to_losers":
        keep: list = []
        for pl in plans:
            if pl.side != "buy":
                keep.append(pl)
                continue
            try:
                cur_mv = float(current_map.get(pl.symbol, 0.0) or 0.0)
                avg = float(avg_entry_map.get(pl.symbol, 0.0) or 0.0)
                px = float(cur_price_map.get(pl.symbol, 0.0) or 0.0)
                # if position exists and current price below avg entry, skip add
                if cur_mv > 0 and avg > 0 and px > 0 and px < avg:
                    continue
            except Exception:
                pass
            keep.append(pl)
        plans = keep

    print("\nOrder plan (notional USD):")
    if not plans:
        print("(no trades)")
    for pl in plans:
        print(f"- {pl.side.upper():4s} {pl.symbol:12s} ${pl.notional_usd:,.2f}  ({pl.asset_class})")

    write_artifact(
        "last_rebalance.json",
        {
            "universe": {"equities": len(eq_symbols), "crypto": len(cr_symbols)},
            "selected": {"equities": eq_sel, "crypto": cr_sel},
            "entry_signals": {
                "equities": {s: eq_sig_details.get(s) for s in eq_sel},
                "crypto": {s: cr_sig_details.get(s) for s in cr_sel},
            },
            "plans": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "notional_usd": p.notional_usd,
                    "asset_class": p.asset_class,
                    "reason": p.reason,
                }
                for p in plans
            ],
            "frozen": dd_state.frozen,
        },
    )

    if dd_state.frozen:
        if cfg.risk.dd_stop_behavior == "liquidate_to_cash":
            # Override plan: liquidate everything.
            plans = []
            for sym, cur in sorted(current_map.items()):
                if cur <= 0:
                    continue
                plans.append(
                    OrderPlan(
                        symbol=sym,
                        side="sell",
                        notional_usd=float(cur),
                        asset_class=("crypto" if "/" in sym else "equity"),
                        reason="dd_stop_liquidate",
                    )
                )
            print("\nDD stop mode: liquidate_to_cash (all positions set to SELL).")
        else:
            # freeze: allow sells, block buys
            plans = [p for p in plans if p.side == "sell"]
            print("\nDD stop mode: freeze (filtered order plan to SELLS only).")

    if dry_run or not args.place_orders:
        print("\nNo orders placed (dry-run or --place-orders not set).")
        return 0

    # Hard safety: only allow placing when both config + env indicate paper
    if cfg.mode != "paper" or not env.paper:
        raise RuntimeError("Refusing to place orders unless cfg.mode=paper and APCA_PAPER=true")

    gr = check_order_guardrails(
        plans,
        max_orders=cfg.execution.max_orders_per_run,
        max_single_notional=cfg.execution.max_single_order_notional_usd,
        max_total_notional=cfg.execution.max_total_notional_usd,
    )
    if not gr.ok:
        raise RuntimeError(f"Guardrail blocked order placement: {gr.reason}")

    # Reference prices for optional limit orders
    ref_px: dict[str, float] = {}
    for sym, df in {**eq_bars, **cr_bars}.items():
        try:
            if df is not None and len(df) and "close" in df.columns:
                ref_px[sym] = float(df["close"].dropna().iloc[-1])
        except Exception:
            pass

    placed = place_notional_market_orders(
        clients.trading,
        plans,
        use_limit_orders=bool(cfg.execution.use_limit_orders),
        limit_offset_bps=float(cfg.execution.limit_offset_bps),
        ref_price_by_symbol=ref_px,
    )

    write_artifact(
        "last_placed_orders.json",
        {
            "count": len(placed),
            "orders": [o.__dict__ for o in placed],
        },
    )

    print("\nPlaced orders:")
    for o in placed:
        print(f"- {o.side.upper():4s} {o.symbol:12s} ${o.notional_usd:,.2f}  id={o.id}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="tradebot")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("rebalance", help="Run a (dry-run) rebalance")
    pr.add_argument("--config", default=str(Path("config/config.yaml")), help="Path to config YAML")
    pr.add_argument("--place-orders", action="store_true", help="Actually place orders (requires dry_run=false in config and DRY_RUN=false env)")
    pr.add_argument(
        "--wait-until",
        default=None,
        help="Optional local time HH:MM to sleep until before running (uses config.scheduling.timezone). Useful for unattended open/close runs.",
    )
    pr.add_argument("--preset", default=None, help="Override config.active_preset for this run")
    pr.set_defaults(func=cmd_rebalance)

    pc = sub.add_parser("risk-check", help="Run drawdown/freeze check (no trades)")
    pc.add_argument("--config", default=str(Path("config/config.yaml")), help="Path to config YAML")
    pc.add_argument("--preset", default=None, help="Override config.active_preset for this run")
    pc.set_defaults(func=cmd_risk_check)

    pd = sub.add_parser("dashboard", help="Run local HTML dashboard server")
    pd.add_argument("--config", default=str(Path("config/config.yaml")), help="Path to config YAML")
    pd.add_argument("--host", default="127.0.0.1")
    pd.add_argument("--port", type=int, default=8008)
    pd.set_defaults(func=cmd_dashboard)

    args = p.parse_args()
    return int(args.func(args))
