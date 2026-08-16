"""Thin wrapper around alpaca-py: account/position lookups, historical bars, and
market order submission for both equities and crypto.
"""
import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestTradeRequest,
    StockBarsRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from bot.portfolio import Position
from config import Config

logger = logging.getLogger(__name__)

CRYPTO_ASSET_CLASS = "crypto"


class AlpacaClient:
    def __init__(self, cfg: Config):
        self.trading = TradingClient(cfg.api_key, cfg.secret_key, paper=cfg.paper)
        self.stock_data = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)
        self.crypto_data = CryptoHistoricalDataClient(cfg.api_key, cfg.secret_key)

    def get_account(self):
        return self.trading.get_account()

    def get_open_positions(self) -> Dict[str, dict]:
        """Returns {symbol: {"side": "long"|"short", "qty": float, "avg_entry_price": float}}."""
        result = {}
        for pos in self.trading.get_all_positions():
            symbol = _to_display_symbol(pos.symbol)
            side = "long" if str(pos.side).lower().endswith("long") else "short"
            result[symbol] = {
                "side": side,
                "qty": abs(float(pos.qty)),
                "avg_entry_price": float(pos.avg_entry_price),
            }
        return result

    def get_bars(self, symbol: str, timeframe: TimeFrame, limit: int, asset_class: str) -> pd.DataFrame:
        if asset_class == CRYPTO_ASSET_CLASS:
            req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=timeframe, limit=limit)
            raw = self.crypto_data.get_crypto_bars(req).df
        else:
            req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=timeframe, limit=limit)
            raw = self.stock_data.get_stock_bars(req).df

        if raw is None or raw.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.xs(symbol, level=0)

        return raw.sort_index()

    def get_bars_range(
        self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime, asset_class: str
    ) -> pd.DataFrame:
        """Historical bars for a fixed [start, end] date range, used by the backtester
        (bot/backtest.py). The live loop only ever calls get_bars() with a `limit`, so
        this is purely additive - it doesn't change get_bars()'s behavior at all.
        alpaca-py paginates large start/end ranges internally.
        """
        if asset_class == CRYPTO_ASSET_CLASS:
            req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=timeframe, start=start, end=end)
            raw = self.crypto_data.get_crypto_bars(req).df
        else:
            req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=timeframe, start=start, end=end)
            raw = self.stock_data.get_stock_bars(req).df

        if raw is None or raw.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.xs(symbol, level=0)

        return raw.sort_index()

    def get_latest_price(self, symbol: str, asset_class: str) -> float:
        if asset_class == CRYPTO_ASSET_CLASS:
            req = CryptoLatestTradeRequest(symbol_or_symbols=[symbol])
            trade = self.crypto_data.get_crypto_latest_trade(req)[symbol]
        else:
            req = StockLatestTradeRequest(symbol_or_symbols=[symbol])
            trade = self.stock_data.get_stock_latest_trade(req)[symbol]
        return float(trade.price)

    def submit_entry_order(self, symbol: str, qty: float, side: str, asset_class: str):
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        tif = TimeInForce.GTC if asset_class == CRYPTO_ASSET_CLASS else TimeInForce.DAY
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=order_side, time_in_force=tif)
        logger.info("Submitting %s order: %s qty=%s (%s)", order_side, symbol, qty, side)
        return self.trading.submit_order(req)

    def close_position(self, symbol: str, position: Position, asset_class: str):
        order_side = OrderSide.SELL if position.side == "long" else OrderSide.BUY
        tif = TimeInForce.GTC if asset_class == CRYPTO_ASSET_CLASS else TimeInForce.DAY
        req = MarketOrderRequest(symbol=symbol, qty=position.qty, side=order_side, time_in_force=tif)
        logger.info("Closing %s position in %s: %s qty=%s", position.side, symbol, order_side, position.qty)
        return self.trading.submit_order(req)


def _to_display_symbol(broker_symbol: str) -> str:
    """Alpaca returns crypto symbols without the slash (e.g. BTCUSD); normalize back
    to our config's "BTC/USD" form."""
    if broker_symbol == "BTCUSD":
        return "BTC/USD"
    return broker_symbol
