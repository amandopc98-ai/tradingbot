"""Entry point: polls each configured symbol, enforces stop-losses every tick, and
evaluates strategy signals whenever a new bar closes for that symbol's timeframe.
"""
import logging
import logging.handlers
import os
import time
from typing import Dict

import pandas as pd

from bot import indicators
from bot import risk_manager as risk
from bot.alpaca_client import AlpacaClient
from bot.portfolio import Portfolio
from bot.strategies import STRATEGY_REGISTRY, Signal, Strategy
from config import Config, SymbolConfig, load_config

logger = logging.getLogger("tradingbot")


def setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        "logs/tradingbot.log", maxBytes=5_000_000, backupCount=3
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(console)


def build_strategies(cfg: Config) -> Dict[str, Strategy]:
    strategies = {}
    for symbol, scfg in cfg.symbols.items():
        strategy_cls = STRATEGY_REGISTRY[scfg.strategy]
        strategies[symbol] = strategy_cls(symbol, **scfg.params)
    return strategies


def shorting_allowed(symbol: str, scfg: SymbolConfig, cfg: Config) -> bool:
    if scfg.asset_class == "crypto":
        return False  # Alpaca crypto is spot-only, no short selling
    return cfg.allow_shorting


def process_symbol(
    symbol: str,
    strategy: Strategy,
    scfg: SymbolConfig,
    cfg: Config,
    client: AlpacaClient,
    portfolio: Portfolio,
    last_bar_ts: Dict[str, object],
) -> None:
    position = portfolio.get(symbol)

    # 1. Stop-loss / trailing-stop check runs every tick regardless of bar closes,
    #    so a fast intra-bar move against us doesn't wait for the next bar close.
    if position is not None:
        try:
            current_price = client.get_latest_price(symbol, scfg.asset_class)
        except Exception:
            logger.exception("Failed to fetch latest price for %s", symbol)
            return

        bars_for_atr = client.get_bars(symbol, scfg.timeframe, limit=20, asset_class=scfg.asset_class)
        atr_value = 0.0
        if len(bars_for_atr) >= 15:
            atr_series = indicators.atr(bars_for_atr, period=14)
            if not atr_series.empty and not atr_series.isna().all():
                atr_value = float(atr_series.iloc[-1])

        risk.update_trailing_stop(position, current_price, atr_value, scfg.trail_atr_mult)

        if risk.stop_triggered(position, current_price):
            try:
                client.close_position(symbol, position, scfg.asset_class)
            except Exception:
                logger.exception("Failed to submit close order for %s", symbol)
                return
            logger.info(
                "%s: stop triggered at %.4f (stop=%.4f), closed %s position",
                symbol, current_price, position.stop_price, position.side,
            )
            portfolio.remove(symbol)
            return

    # 2. Strategy signal, evaluated only once a new bar has closed for this timeframe.
    bars = client.get_bars(symbol, scfg.timeframe, limit=strategy.min_bars + 5, asset_class=scfg.asset_class)
    if bars.empty or len(bars) < strategy.min_bars:
        return

    latest_ts = bars.index[-1]
    if last_bar_ts.get(symbol) == latest_ts:
        return
    last_bar_ts[symbol] = latest_ts

    position = portfolio.get(symbol)  # re-fetch: may have just been closed above
    signal = strategy.generate_signal(bars, position.side if position else None)

    atr_series = indicators.atr(bars, period=14)
    atr_value = float(atr_series.iloc[-1]) if not atr_series.empty and not pd.isna(atr_series.iloc[-1]) else 0.0
    last_close = float(bars["close"].iloc[-1])

    if signal == Signal.EXIT and position is not None:
        try:
            client.close_position(symbol, position, scfg.asset_class)
        except Exception:
            logger.exception("Failed to submit close order for %s", symbol)
            return
        logger.info("%s: strategy exit signal, closed %s position", symbol, position.side)
        portfolio.remove(symbol)
        return

    if signal in (Signal.LONG, Signal.SHORT) and position is None:
        side = "long" if signal == Signal.LONG else "short"

        if side == "short" and not shorting_allowed(symbol, scfg, cfg):
            logger.info("%s: short signal ignored (shorting disabled for this asset)", symbol)
            return

        if risk.correlation_blocked(symbol, side, portfolio):
            logger.info("%s: long signal blocked by SPY/QQQ correlation filter", symbol)
            return

        if atr_value <= 0:
            logger.warning("%s: ATR unavailable, skipping entry", symbol)
            return

        try:
            account = client.get_account()
        except Exception:
            logger.exception("Failed to fetch account info")
            return

        qty = risk.position_size(float(account.equity), atr_value, last_close, cfg.risk_pct, scfg.asset_class)
        qty = risk.cap_by_buying_power(
            qty, float(account.buying_power), last_close, scfg.max_allocation_pct, scfg.asset_class
        )
        if qty <= 0:
            logger.warning("%s: computed order quantity is 0, skipping entry", symbol)
            return

        try:
            client.submit_entry_order(symbol, qty, side, scfg.asset_class)
        except Exception:
            logger.exception("Failed to submit entry order for %s", symbol)
            return

        stop_price = risk.initial_stop_price(last_close, atr_value, side)
        portfolio.open(symbol, side, qty, last_close, atr_value, stop_price)
        logger.info(
            "%s: opened %s %.6f @ ~%.4f, ATR=%.4f, hard stop=%.4f",
            symbol, side, qty, last_close, atr_value, stop_price,
        )


def run() -> None:
    setup_logging()
    cfg = load_config()
    client = AlpacaClient(cfg)
    portfolio = Portfolio(state_path=cfg.state_file)

    try:
        broker_positions = client.get_open_positions()
        portfolio.reconcile_with_broker(broker_positions)
    except Exception:
        logger.exception("Failed to reconcile with broker on startup")

    strategies = build_strategies(cfg)
    last_bar_ts: Dict[str, object] = {}

    logger.info(
        "Starting trading bot (paper=%s) for symbols: %s",
        cfg.paper, ", ".join(cfg.symbols.keys()),
    )

    while True:
        for symbol, strategy in strategies.items():
            scfg = cfg.symbols[symbol]
            try:
                process_symbol(symbol, strategy, scfg, cfg, client, portfolio, last_bar_ts)
            except Exception:
                logger.exception("Unhandled error processing %s", symbol)
        portfolio.save()
        time.sleep(cfg.poll_interval_seconds)


if __name__ == "__main__":
    run()
