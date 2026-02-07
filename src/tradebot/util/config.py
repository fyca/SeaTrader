from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Allocation(BaseModel):
    equities: float = 0.5
    crypto: float = 0.5


class Limits(BaseModel):
    max_equity_positions: int = 10
    max_crypto_positions: int = 5
    min_stock_price: float = 5.0
    min_avg_dollar_volume_20d: float = 20_000_000
    min_avg_crypto_dollar_volume_20d: float = 5_000_000


class Risk(BaseModel):
    max_drawdown_freeze: float = 0.20
    warn_drawdown: float = 0.10
    per_asset_stop_loss_pct: float | None = None


class SignalParams(BaseModel):
    ma_long: int
    ma_short: int
    vol_lookback: int
    max_ann_vol: float


class Signals(BaseModel):
    timeframe: str = "1D"
    lookback_days: int = 420
    equity: SignalParams = SignalParams(ma_long=200, ma_short=50, vol_lookback=20, max_ann_vol=0.80)
    crypto: SignalParams = SignalParams(ma_long=120, ma_short=30, vol_lookback=20, max_ann_vol=2.50)


class Execution(BaseModel):
    use_limit_orders: bool = False
    limit_offset_bps: int = 10
    max_orders_per_run: int = 25
    max_single_order_notional_usd: float = 2500
    max_total_notional_usd: float = 15000


class Scheduling(BaseModel):
    weekly_rebalance_day: str = "MON"
    daily_risk_check_time_local: str = "18:05"


class Universe(BaseModel):
    equity_benchmark_symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    crypto_symbols_allowlist: list[str] = Field(default_factory=list)
    exclude_leveraged_etfs: bool = True


class BotConfig(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    strategy_id: str = "baseline_trendvol"
    dry_run: bool = True
    allocation: Allocation = Allocation()
    limits: Limits = Limits()
    risk: Risk = Risk()
    signals: Signals = Signals()
    execution: Execution = Execution()
    scheduling: Scheduling = Scheduling()
    universe: Universe = Universe()


def load_config(path: str | Path) -> BotConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    return BotConfig.model_validate(data)
