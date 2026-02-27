#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from alpaca.trading.client import TradingClient

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bot")
    args = ap.parse_args()

    bot = args.bot.strip().lower()
    bot_dir = ROOT / "bots" / bot
    if not bot_dir.exists():
        raise SystemExit(f"Unknown bot: {bot}")

    env_file = bot_dir / ".env"
    if not env_file.exists():
        raise SystemExit(f"Missing env file: {env_file}")

    env = os.environ.copy()
    env.update(load_env_file(env_file))

    key = env.get("APCA_API_KEY_ID")
    secret = env.get("APCA_API_SECRET_KEY")
    paper = str(env.get("APCA_PAPER", "true")).lower() == "true"
    if not key or not secret:
        raise SystemExit("Missing APCA_API_KEY_ID/APCA_API_SECRET_KEY in bot .env")

    c = TradingClient(key, secret, paper=paper)
    try:
        c.cancel_orders()
    except Exception:
        pass
    try:
        c.close_all_positions(cancel_orders=True)
    except Exception:
        pass
    print("broker account flatten requested")

    for d in (bot_dir / "data", bot_dir / "logs"):
        d.mkdir(parents=True, exist_ok=True)
        for p in list(d.iterdir()):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)

    print(f"wipe complete for {bot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
