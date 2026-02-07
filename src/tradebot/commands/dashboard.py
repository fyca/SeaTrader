from __future__ import annotations

import argparse

import uvicorn

from tradebot.dashboard.app import create_app


def cmd_dashboard(args: argparse.Namespace) -> int:
    app = create_app(config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
