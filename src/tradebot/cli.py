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
from tradebot.execution.plan import diff_to_orders
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
    dd_state = update_drawdown_state(prior_peak_equity=state.peak_equity, current_equity=equity, freeze_at=cfg.risk.max_drawdown_freeze)
    state.peak_equity = dd_state.peak_equity
    save_state(state)
    if dd_state.frozen:
        print(f"[red]FROZEN[/red]: drawdown={dd_state.drawdown:.1%} (>= {cfg.risk.max_drawdown_freeze:.0%}) -> no new entries")
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
    for p in current_positions:
        sym = getattr(p, "symbol", "")
        mv = float(getattr(p, "market_value", 0.0) or 0.0)
        if sym:
            current_map[sym] = mv

    # 6) Plan orders
    plans = diff_to_orders(current_notional=current_map, targets=target_map, asset_class_by_symbol=class_map)

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
        # Allow sells (exits), block buys
        plans = [p for p in plans if p.side == "sell"]
        print("\nFrozen mode: filtered order plan to SELLS only.")

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

    placed = place_notional_market_orders(clients.trading, plans)

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
