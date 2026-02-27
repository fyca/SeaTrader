#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]  # tradebot/multibot
REPO = ROOT.parents[0]                      # tradebot
BOTS_DIR = ROOT / "bots"
RUN_BOT = ROOT / "scripts" / "run_bot.sh"

MARK_BEGIN = "# >>> SEATRADER_MULTIBOT_CRON_BEGIN >>>"
MARK_END = "# <<< SEATRADER_MULTIBOT_CRON_END <<<"

BOT_ORDER = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota"]
DOW_MAP = {"SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6}


def sh(cmd: list[str], stdin: str | None = None) -> tuple[int, str, str]:
    p = subprocess.run(cmd, input=stdin, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr


def read_crontab() -> str:
    code, out, err = sh(["crontab", "-l"])
    if code != 0:
        # No crontab typically returns non-zero + stderr text
        if "no crontab" in (err or "").lower():
            return ""
        raise RuntimeError(f"crontab -l failed: {err or out}")
    return out


def write_crontab(text: str) -> None:
    code, out, err = sh(["crontab", "-"], stdin=text)
    if code != 0:
        raise RuntimeError(f"crontab install failed: {err or out}")


def strip_block(cron_text: str) -> str:
    pattern = re.compile(rf"\n?{re.escape(MARK_BEGIN)}.*?{re.escape(MARK_END)}\n?", re.S)
    cleaned = re.sub(pattern, "\n", cron_text).strip("\n")
    return (cleaned + "\n") if cleaned else ""


def hhmm_to_min_hour(hhmm: str) -> tuple[int, int]:
    hh, mm = hhmm.split(":")
    return int(mm), int(hh)


def cron_time(freq: str, day: str | None, hhmm: str, minute_of_hour: int | None = None) -> str:
    m, h = hhmm_to_min_hour(hhmm)
    f = (freq or "").lower()
    if f == "daily":
        return f"{m} {h} * * *"
    if f == "weekly":
        d = DOW_MAP.get((day or "MON").upper(), 1)
        return f"{m} {h} * * {d}"
    if f == "hourly":
        mm = m if minute_of_hour is None else max(0, min(59, int(minute_of_hour)))
        return f"{mm} * * * *"
    raise ValueError(f"Unsupported frequency: {freq!r} (supported: daily, weekly, hourly)")


def _daily_hhmm_cron(hhmm: str) -> str:
    m, h = hhmm_to_min_hour(hhmm)
    return f"{m} {h} * * *"


def bot_cron_lines(bot: str, alias: str | None = None, config_path: Path | None = None) -> list[str]:
    cfg_p = config_path or (BOTS_DIR / bot / "config" / "config.yaml")
    if not cfg_p.exists():
        raise FileNotFoundError(f"Missing config: {cfg_p}")

    cfg = yaml.safe_load(cfg_p.read_text()) or {}
    sched = cfg.get("scheduling") or {}
    eq = sched.get("equities") or {}
    cr = sched.get("crypto") or {}
    ex = cfg.get("execution") or {}
    ex_eq = ex.get("equities") or {}
    ex_cr = ex.get("crypto") or {}

    lines: list[str] = []
    prefix = f"cd {REPO} && source .venv/bin/activate"
    tag_bot = (alias or bot).upper()

    eq_reb = cron_time(eq.get("rebalance_frequency", "weekly"), eq.get("rebalance_day", "MON"), eq.get("rebalance_time_local", "06:30"))
    eq_risk = cron_time(eq.get("risk_check_frequency", "daily"), eq.get("risk_check_day", "MON"), eq.get("risk_check_time_local", "06:31"), eq.get("risk_check_minute_of_hour"))
    cr_reb = cron_time(cr.get("rebalance_frequency", "daily"), cr.get("rebalance_day", "MON"), cr.get("rebalance_time_local", "00:00"))
    cr_risk = cron_time(cr.get("risk_check_frequency", "daily"), cr.get("risk_check_day", "MON"), cr.get("risk_check_time_local", "00:01"), cr.get("risk_check_minute_of_hour"))

    lines.append(
        f"{eq_reb} /bin/bash -lc '{prefix} && {RUN_BOT} {bot} rebalance --place-orders --asset-mode equities' # STMB_{tag_bot}_REB_EQ"
    )
    lines.append(
        f"{eq_risk} /bin/bash -lc '{prefix} && {RUN_BOT} {bot} risk-check --asset-mode equities' # STMB_{tag_bot}_RISK_EQ"
    )
    lines.append(
        f"{cr_reb} /bin/bash -lc '{prefix} && {RUN_BOT} {bot} rebalance --place-orders --asset-mode crypto' # STMB_{tag_bot}_REB_CR"
    )
    lines.append(
        f"{cr_risk} /bin/bash -lc '{prefix} && {RUN_BOT} {bot} risk-check --asset-mode crypto' # STMB_{tag_bot}_RISK_CR"
    )

    # Dedicated fallback sweeps: ensure stale limit orders can still convert even when
    # rebalance/risk-check timing does not align with fallback window.
    try:
        if bool(ex_eq.get("fallback_to_market_at_open", False)):
            t = str(ex_eq.get("fallback_time_local", "06:30"))
            lines.append(
                f"{_daily_hhmm_cron(t)} /bin/bash -lc '{prefix} && {RUN_BOT} {bot} risk-check --asset-mode equities' # STMB_{tag_bot}_FALL_EQ"
            )
    except Exception:
        pass

    try:
        if bool(ex_cr.get("fallback_to_market_at_open", False)):
            t = str(ex_cr.get("fallback_time_local", "00:02"))
            lines.append(
                f"{_daily_hhmm_cron(t)} /bin/bash -lc '{prefix} && {RUN_BOT} {bot} risk-check --asset-mode crypto' # STMB_{tag_bot}_FALL_CR"
            )
    except Exception:
        pass

    return lines


def _clean_alias(name: str | None) -> str | None:
    if not name:
        return None
    s = re.sub(r"[^A-Za-z0-9_\-]", "_", str(name).strip())
    return s[:40] if s else None


def build_block() -> str:
    lines = [MARK_BEGIN, "# Generated by multibot/scripts/manage_cron.py --install"]
    for bot in BOT_ORDER:
        lines.append(f"# --- {bot} ---")
        lines.extend(bot_cron_lines(bot))
    lines.append(MARK_END)
    return "\n".join(lines) + "\n"


def install_bot(bot: str, alias: str | None = None, config_path: str | None = None) -> list[str]:
    bot = bot.lower().strip()
    alias = _clean_alias(alias)
    cpath = Path(config_path).resolve() if config_path else None
    new_lines = bot_cron_lines(bot, alias=alias, config_path=cpath)

    existing = read_crontab().splitlines()
    tag = (alias or bot).upper()
    runbot_hint = f"run_bot.sh {bot} "

    filtered = []
    for ln in existing:
        if f"STMB_{tag}_" in ln:
            continue
        if runbot_hint in ln and "STMB_" in ln:
            continue
        filtered.append(ln)

    filtered.extend(new_lines)
    payload = "\n".join(x for x in filtered if x is not None).strip("\n") + "\n"
    write_crontab(payload)
    return new_lines


def remove_bot(bot: str, alias: str | None = None) -> int:
    bot = bot.lower().strip()
    alias = _clean_alias(alias)
    existing = read_crontab().splitlines()
    tag = (alias or bot).upper()
    runbot_hint = f"run_bot.sh {bot} "

    kept = []
    removed = 0
    for ln in existing:
        if f"STMB_{tag}_" in ln or (runbot_hint in ln and "STMB_" in ln):
            removed += 1
            continue
        kept.append(ln)
    payload = "\n".join(kept).strip("\n")
    write_crontab((payload + "\n") if payload else "")
    return removed


def install() -> str:
    current = read_crontab()
    base = strip_block(current)
    block = build_block()
    merged = (base + "\n" + block).replace("\n\n\n", "\n\n")
    write_crontab(merged)
    return block


def remove() -> None:
    current = read_crontab()
    base = strip_block(current)
    write_crontab(base)


def preview() -> str:
    return build_block()


def current_block() -> str:
    txt = read_crontab()
    start = txt.find(MARK_BEGIN)
    end = txt.find(MARK_END)
    if start == -1 or end == -1 or end < start:
        return ""
    end += len(MARK_END)
    return txt[start:end] + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Manage per-bot SeaTrader cron entries")
    ap.add_argument("action", choices=["install", "remove", "preview", "show", "install-bot", "remove-bot"])
    ap.add_argument("--bot", help="bot folder name (alpha..iota)")
    ap.add_argument("--alias", help="optional selectable bot label used in STMB tags")
    ap.add_argument("--config", help="optional explicit config.yaml path for install-bot")
    args = ap.parse_args()

    if args.action == "install":
        blk = install()
        print("Installed per-bot cron block:\n")
        print(blk)
        return 0
    if args.action == "remove":
        remove()
        print("Removed per-bot cron block")
        return 0
    if args.action == "preview":
        print(preview())
        return 0
    if args.action == "show":
        print(current_block() or "(no per-bot cron block installed)")
        return 0
    if args.action == "install-bot":
        if not args.bot:
            raise SystemExit("--bot is required for install-bot")
        lines = install_bot(args.bot, alias=args.alias, config_path=args.config)
        print("Installed cron lines for bot:\n")
        print("\n".join(lines))
        return 0
    if args.action == "remove-bot":
        if not args.bot:
            raise SystemExit("--bot is required for remove-bot")
        n = remove_bot(args.bot, alias=args.alias)
        print(f"Removed {n} cron line(s) for bot {args.bot}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
