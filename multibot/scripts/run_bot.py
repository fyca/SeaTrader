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
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    bot = args.bot.strip().lower()
    bot_dir = ROOT / "bots" / bot
    if not bot_dir.exists():
        raise SystemExit(f"Unknown bot: {bot}")

    cfg = bot_dir / "config" / "config.yaml"
    py = venv_python()

    env = os.environ.copy()
    env_file = bot_dir / ".env"
    file_env = load_env_file(env_file)
    env.update(file_env)

    base = [py, "-m", "tradebot.cli"]
    if args.cmd == "rebalance":
        cmd = base + ["rebalance", "--config", str(cfg)] + args.extra
    elif args.cmd == "risk-check":
        cmd = base + ["risk-check", "--config", str(cfg)] + args.extra
    else:
        port = int(env.get("PORT") or PORTS.get(bot, 8008))
        cmd = base + ["dashboard", "--config", str(cfg), "--host", "127.0.0.1", "--port", str(port)] + args.extra

    if args.debug:
        print(f"[run_bot] bot={bot} cmd={args.cmd}")
        print(f"[run_bot] cwd={bot_dir}")
        print(f"[run_bot] python={py}")
        print(f"[run_bot] config={cfg} exists={cfg.exists()}")
        print(f"[run_bot] env_file={env_file} exists={env_file.exists()}")
        print(f"[run_bot] loaded_env_keys={sorted(file_env.keys())}")
        print(f"[run_bot] APCA_API_KEY_ID present={bool(env.get('APCA_API_KEY_ID'))}")
        print(f"[run_bot] APCA_API_SECRET_KEY present={bool(env.get('APCA_API_SECRET_KEY'))}")
        print(f"[run_bot] exec={' '.join(cmd)}")

    if args.cmd == "dashboard" and (not env.get("APCA_API_KEY_ID") or not env.get("APCA_API_SECRET_KEY")):
        print("[run_bot] ERROR: Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY after loading bot .env")
        return 2

    p = subprocess.run(cmd, cwd=str(bot_dir), env=env)
    if args.debug:
        print(f"[run_bot] exit_code={p.returncode}")
    return int(p.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
