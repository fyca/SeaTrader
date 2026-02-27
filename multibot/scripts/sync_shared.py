#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # tradebot/multibot
REPO = ROOT.parent
BOTS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota"]


def sync_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        for p in src.rglob("*"):
            rel = p.relative_to(src)
            out = dst / rel
            if p.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)


def main() -> int:
    for bot in BOTS:
        bot_dir = ROOT / "bots" / bot
        (bot_dir / "config").mkdir(parents=True, exist_ok=True)
        (bot_dir / "strategies" / "user").mkdir(parents=True, exist_ok=True)

        sync_tree(REPO / "strategies" / "user", bot_dir / "strategies" / "user")

        for name in ("presets.yaml", "backtest_presets.yaml"):
            src = REPO / "config" / name
            if src.exists():
                shutil.copy2(src, bot_dir / "config" / name)

    print("Shared strategies/presets synced to alpha..iota (9 bots).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
