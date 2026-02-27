#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

PORTS = {
    "alpha": 8008,
    "beta": 8009,
    "gamma": 8010,
    "delta": 8011,
    "epsilon": 8012,
    "zeta": 8013,
    "eta": 8014,
    "theta": 8015,
    "iota": 8016,
}


def venv_python() -> str:
    if os.name == "nt":
        p = REPO / ".venv" / "Scripts" / "python.exe"
    else:
        p = REPO / ".venv" / "bin" / "python"
    if p.exists():
        return str(p)
    return sys.executable


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
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
    ap.add_argument("cmd", choices=["rebalance", "risk-check", "dashboard"])
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    bot = args.bot.strip().lower()
    bot_dir = ROOT / "bots" / bot
    if not bot_dir.exists():
        raise SystemExit(f"Unknown bot: {bot}")

    cfg = bot_dir / "config" / "config.yaml"
    py = venv_python()

    env = os.environ.copy()
    env.update(load_env_file(bot_dir / ".env"))

    base = [py, "-m", "tradebot.cli"]
    if args.cmd == "rebalance":
        cmd = base + ["rebalance", "--config", str(cfg)] + args.extra
    elif args.cmd == "risk-check":
        cmd = base + ["risk-check", "--config", str(cfg)] + args.extra
    else:
        port = int(env.get("PORT") or PORTS.get(bot, 8008))
        cmd = base + ["dashboard", "--config", str(cfg), "--host", "127.0.0.1", "--port", str(port)] + args.extra

    p = subprocess.run(cmd, cwd=str(bot_dir), env=env)
    return int(p.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
