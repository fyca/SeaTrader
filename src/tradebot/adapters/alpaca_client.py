from __future__ import annotations

from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient

from tradebot.util.env import Env


@dataclass(frozen=True)
class AlpacaClients:
    trading: TradingClient
    stocks: StockHistoricalDataClient
    crypto: CryptoHistoricalDataClient


def make_alpaca_clients(env: Env) -> AlpacaClients:
    # alpaca-py automatically targets paper via TradingClient(paper=True)
    trading = TradingClient(env.key_id, env.secret_key, paper=env.paper)
    stocks = StockHistoricalDataClient(env.key_id, env.secret_key)
    crypto = CryptoHistoricalDataClient(env.key_id, env.secret_key)
    return AlpacaClients(trading=trading, stocks=stocks, crypto=crypto)
