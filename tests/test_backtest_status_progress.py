from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from tradebot.backtest.engine import BacktestResult
from tradebot.backtest import job
from tradebot.dashboard.app import create_app


class _DummyAsset:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.tradable = True
        self.status = "active"


class _DummyTrading:
    def get_all_assets(self):
        return [_DummyAsset("SPY")]


class _DummyClients:
    def __init__(self):
        self.trading = _DummyTrading()
        self.stocks = object()
        self.crypto = object()


def _wait_for_done(job_id: str, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = job.get_status(job_id)
        if st.get("state") in {"done", "error", "stopped", "missing"}:
            return st
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish in time")


def test_backtest_done_status_preserves_progress_and_writes_result(monkeypatch):
    monkeypatch.setattr(job, "load_config", lambda _p: type("Cfg", (), {
        "signals": type("S", (), {"lookback_days": 5})(),
        "limits": type("L", (), {
            "min_avg_crypto_dollar_volume_20d": 5_000_000,
            "max_equity_positions": 10,
            "max_crypto_positions": 5,
        })(),
    })())
    monkeypatch.setattr(job, "load_env", lambda: object())
    monkeypatch.setattr(job, "make_alpaca_clients", lambda _e: _DummyClients())
    monkeypatch.setattr(job, "get_sp500_symbols", lambda: ["SPY"])
    monkeypatch.setattr(job, "list_tradable_crypto", lambda _t: [type("A", (), {"symbol": "BTC/USD"})()])
    monkeypatch.setattr(job, "load_cached_frames", lambda *args, **kwargs: {})
    monkeypatch.setattr(job, "save_cached_frames", lambda *args, **kwargs: None)

    def _fake_run_backtest(**kwargs):
        progress_cb = kwargs["progress_cb"]
        progress_cb(1, 5, 100_100.0)
        progress_cb(5, 5, 101_234.56)
        return BacktestResult(
            params={"asset_mode": "crypto"},
            equity_curve=[{"date": "2026-01-01", "equity": 100_000.0, "cash": 100_000.0}],
            metrics={"return": 0.012, "trade_count": 1},
            trades=[],
            events=[],
            open_positions=[],
            realized_pnl_by_symbol={},
            excluded_symbols=[],
        )

    monkeypatch.setattr(job, "run_backtest", _fake_run_backtest)

    job_id = job.start_backtest(
        config_path="config/config.yaml",
        params={
            "start": "2026-01-01",
            "end": "2026-01-31",
            "asset_mode": "crypto",
            "risk_check_frequency_crypto": "hourly",
        },
    )

    st = _wait_for_done(job_id)
    assert st["state"] == "done"
    assert st["progress"] == 5
    assert st["total"] == 5
    assert st["result_ready"] is True
    assert abs(float(st.get("current_equity")) - 101_234.56) < 1e-6

    result_path = Path("data/backtests") / job_id / "result.json"
    assert result_path.exists()
    payload = json.loads(result_path.read_text())
    assert payload.get("job_id") == job_id
    assert payload.get("metrics", {}).get("timing", {}).get("write_seconds") is not None


def test_status_stream_emits_terminal_end_event(monkeypatch):
    app = create_app(config_path="config/config.yaml")

    seq = [
        {"state": "running", "progress": 3, "total": 5},
        {"state": "done", "progress": 5, "total": 5, "result_ready": True},
    ]

    monkeypatch.setattr("tradebot.dashboard.app.get_latest_job_id", lambda: "job-1")

    def _fake_bt_status(_job_id):
        return seq.pop(0) if seq else {"state": "done", "progress": 5, "total": 5}

    monkeypatch.setattr("tradebot.dashboard.app.bt_status", _fake_bt_status)

    client = TestClient(app)
    with client.stream("GET", "/api/backtest/status/stream?job_id=job-1") as resp:
        text = "".join(chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else chunk for chunk in resp.iter_raw())

    assert "data: {\"progress\": 3, \"state\": \"running\", \"total\": 5}" in text
    assert "data: {\"progress\": 5, \"result_ready\": true, \"state\": \"done\", \"total\": 5}" in text
    assert "event: end\ndata: done" in text
