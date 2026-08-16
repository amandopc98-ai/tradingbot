"""Backtests the live bot's strategies over historical Alpaca bar data.

This module never touches bot/main.py or changes any live behavior. It reuses the
exact same building blocks the live bot uses so a backtest and a live run are
guaranteed to make identical decisions given the same price history:

  - bot.main.build_strategies() / bot.main.shorting_allowed() - imported, not
    reimplemented, so strategy construction and the short-selling gate can never
    drift out of sync with the live bot.
  - bot.strategies.* - the same Strategy subclasses (mean reversion, momentum
    breakout, trend following) generate signals from the same generate_signal(bars,
    position_side) call the live bot makes.
  - bot.risk_manager - the same position sizing, hard/trailing stop, and
    SPY+QQQ-blocks-BTC correlation filter functions.
  - bot.portfolio.Portfolio / Position - the same in-memory position/stop
    bookkeeping (never persisted to disk here - portfolio.save() is never called,
    so a backtest run can't clobber the live bot's portfolio_state.json).
  - bot.indicators.atr - the same ATR calculation.

Fidelity notes (read before trusting the numbers):

  - The live bot fetches only the most recent `strategy.min_bars + 5` bars for
    signal generation and only the most recent 20 bars for the stop-loss ATR
    (see bot/main.py's process_symbol). This backtest replays that exact windowing
    bar-by-bar rather than handing each strategy the full history, since that's
    what actually happens live - including the EMA "restarts its warm-up on every
    truncated window" quirk that implies for the trend-following strategy. This is
    intentional: the goal is fidelity to live behavior, not a "corrected" backtest.
  - The live bot checks stops on every poll tick using the latest trade price; this
    backtest only has OHLC bar data, so it checks stops once per closed bar using
    that bar's close price as the stand-in for "the price observed at that check".
    A fast intra-bar move through the stop and back may not be caught.
  - "Buying power" is approximated as the symbol's own running equity (no margin,
    no shared-account modeling across symbols).
  - Orders fill at the bar close with no slippage or commission.
  - When multiple --symbol values are backtested together, all of them share one
    Portfolio so the SPY/QQQ-blocks-BTC correlation filter behaves correctly across
    symbols; bars from every symbol are replayed in true chronological order for
    that reason, even though each symbol still gets its own equity curve.

Strategy overrides (--strategy): a backtest can explore a different strategy for a
symbol than config.py's live mapping without changing that mapping at all - e.g.
`--symbol GLD --strategy opening_range_breakout` backtests GLD with the opening-range
strategy while bot/main.py keeps trading GLD with trend_following, untouched. Some
strategies (currently opening_range_breakout) manage their own stop-loss/take-profit
internally instead of the shared ATR-based one in bot/risk_manager.py, because their
exit levels are fixed price levels, not ATR multiples. Those strategies declare a
`self_managed_exits = True` attribute; this engine checks for it via getattr() and,
only when present, (a) skips the generic ATR stop-loss check in Phase 1 below, and
(b) sizes the entry using the strategy's own get_entry_stop_price() instead of ATR.
Every existing strategy is unaffected - the attribute is simply absent on them, so
getattr() returns the old default and behavior is unchanged.
"""
import argparse
import csv
import os
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from bot import indicators
from bot import risk_manager as risk
from bot.alpaca_client import AlpacaClient
from bot.main import build_strategies, shorting_allowed
from bot.portfolio import Portfolio
from bot.strategies import STRATEGY_REGISTRY, Signal, Strategy
from config import Config, SymbolConfig, load_config

DEFAULT_INITIAL_EQUITY = 100_000.0
STOP_CHECK_WINDOW = 20  # mirrors main.py's client.get_bars(..., limit=20) for the stop-loss ATR

# Only consulted for strategies passed via --strategy that aren't already some
# symbol's live default (see config.py's _default_symbols()) - i.e. a strategy this
# backtest is exploring "off label" for a symbol. Purely additive; doesn't touch
# config.py or change any symbol's live timeframe.
BACKTEST_STRATEGY_DEFAULT_TIMEFRAMES = {
    "opening_range_breakout": TimeFrame.Minute,
}

_TIMEFRAME_UNIT_SECONDS = {
    TimeFrameUnit.Minute: 60,
    TimeFrameUnit.Hour: 3600,
    TimeFrameUnit.Day: 86400,
}


@dataclass
class Trade:
    symbol: str
    side: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    qty: float
    pnl: float
    exit_reason: str  # "signal" | "stop" | "end_of_backtest"


@dataclass
class BacktestResult:
    symbol: str
    initial_equity: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)  # [{"timestamp": iso str, "equity": float}]

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate_pct(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades) * 100.0

    @property
    def total_return_pct(self) -> float:
        if not self.equity_curve or self.initial_equity <= 0:
            return 0.0
        final_equity = self.equity_curve[-1]["equity"]
        return (final_equity / self.initial_equity - 1.0) * 100.0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]["equity"]
        max_dd = 0.0
        for point in self.equity_curve:
            equity = point["equity"]
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)
        return max_dd * 100.0


@dataclass
class _SymbolWindow:
    symbol: str
    scfg: SymbolConfig
    strategy: Strategy
    bars: pd.DataFrame
    entry_window: int  # strategy.min_bars + 5, mirrors main.py's get_bars(limit=...)


def _lookback_start(start: datetime, timeframe: TimeFrame, bars_needed: int, asset_class: str) -> datetime:
    """How far before `start` to fetch bars so strategies already have a full
    window (e.g. 200+ bars for trend following) right at `start`, instead of
    spending the first chunk of the requested period stuck on HOLD."""
    unit_seconds = _TIMEFRAME_UNIT_SECONDS.get(timeframe.unit, 86400)
    bar_seconds = timeframe.amount * unit_seconds
    raw_seconds = bars_needed * bar_seconds
    # Stocks only trade ~6.5h of a 24h day, ~5 of 7 days a week, so calendar time
    # needed to accumulate N *trading* bars is much longer than N * bar_seconds.
    calendar_multiplier = 1.0 if asset_class == "crypto" else 4.5
    padded_seconds = raw_seconds * calendar_multiplier + 3 * 86400
    return start - timedelta(seconds=padded_seconds)


def _latest_atr(window: pd.DataFrame) -> float:
    if len(window) < 15:
        return 0.0
    series = indicators.atr(window, period=14)
    if series.empty or pd.isna(series.iloc[-1]):
        return 0.0
    return float(series.iloc[-1])


class Backtester:
    def __init__(
        self,
        cfg: Config,
        client: AlpacaClient,
        initial_equity: float = DEFAULT_INITIAL_EQUITY,
        strategy_overrides: Optional[Dict[str, str]] = None,
        strategy_override_params: Optional[dict] = None,
    ):
        self.cfg = cfg
        self.client = client
        self.initial_equity = initial_equity
        self.strategy_overrides = strategy_overrides or {}
        # Extra constructor kwargs (e.g. trend_filter_enabled/_period) applied to
        # every overridden strategy in this run - see --trend-filter/--trend-filter-period.
        self.strategy_override_params = strategy_override_params or {}
        self._entry_ts: Dict[str, str] = {}

    def run(self, symbols: List[str], start: datetime, end: datetime) -> Dict[str, BacktestResult]:
        strategies = build_strategies(self.cfg)  # same construction the live bot uses, for every configured symbol
        equity_by_symbol = {s: self.initial_equity for s in symbols}
        results = {s: BacktestResult(symbol=s, initial_equity=self.initial_equity) for s in symbols}

        # A throwaway, guaranteed-nonexistent path: Portfolio._load() is a no-op if
        # the file doesn't exist, and we never call .save(), so nothing is ever
        # written to disk and the live bot's portfolio_state.json is never touched.
        scratch_state_path = os.path.join(tempfile.gettempdir(), f"backtest_portfolio_{uuid.uuid4().hex}.json")
        portfolio = Portfolio(state_path=scratch_state_path)

        windows: Dict[str, _SymbolWindow] = {}
        events = []  # (timestamp, symbol, row_index), merged and time-sorted across all requested symbols

        for symbol in symbols:
            scfg = self.cfg.symbols[symbol]
            strategy = strategies[symbol]

            override_name = self.strategy_overrides.get(symbol)
            if override_name is not None:
                strategy = STRATEGY_REGISTRY[override_name](symbol, **self.strategy_override_params)
                override_timeframe = BACKTEST_STRATEGY_DEFAULT_TIMEFRAMES.get(override_name, scfg.timeframe)
                scfg = replace(scfg, strategy=override_name, timeframe=override_timeframe)

            entry_window = strategy.min_bars + 5

            lookback_start = _lookback_start(
                start, scfg.timeframe, max(strategy.min_bars, STOP_CHECK_WINDOW) + 5, scfg.asset_class
            )
            bars = self.client.get_bars_range(symbol, scfg.timeframe, lookback_start, end, scfg.asset_class)
            if bars.empty:
                raise RuntimeError(f"No historical bars returned for {symbol} between {lookback_start} and {end}")

            windows[symbol] = _SymbolWindow(symbol, scfg, strategy, bars, entry_window)
            for i, ts in enumerate(bars.index):
                if start <= ts <= end:
                    events.append((ts, symbol, i))

        events.sort(key=lambda e: e[0])

        for ts, symbol, i in events:
            self._process_bar(symbol, i, windows[symbol], portfolio, results[symbol], equity_by_symbol)

        # Force-close anything still open at the end of its data, so trade stats
        # and the equity curve reflect a fully realized outcome.
        for symbol in symbols:
            position = portfolio.get(symbol)
            if position is not None:
                w = windows[symbol]
                last_close = float(w.bars["close"].iloc[-1])
                last_ts = w.bars.index[-1]
                self._close_position(symbol, position, last_close, last_ts, "end_of_backtest", portfolio, results[symbol], equity_by_symbol)
                results[symbol].equity_curve.append(
                    {"timestamp": last_ts.isoformat(), "equity": equity_by_symbol[symbol]}
                )

        return results

    def _process_bar(self, symbol, i, w: _SymbolWindow, portfolio: Portfolio, result: BacktestResult, equity_by_symbol: Dict[str, float]) -> None:
        ts = w.bars.index[i]
        close = float(w.bars["close"].iloc[i])
        position = portfolio.get(symbol)
        stopped_out = False

        # Phase 1: stop-loss / trailing-stop check - mirrors the "every tick" check
        # in main.py's process_symbol, evaluated once per closed bar here since only
        # OHLC bar data is available (see module docstring). Skipped entirely for
        # self_managed_exits strategies, which handle their own stop/target/EOD
        # closes through Signal.EXIT in Phase 2 instead.
        self_managed = getattr(w.strategy, "self_managed_exits", False)
        if position is not None and not self_managed:
            stop_window = w.bars.iloc[max(0, i + 1 - STOP_CHECK_WINDOW):i + 1]
            atr_value = _latest_atr(stop_window)
            risk.update_trailing_stop(position, close, atr_value, w.scfg.trail_atr_mult)
            if risk.stop_triggered(position, close):
                self._close_position(symbol, position, close, ts, "stop", portfolio, result, equity_by_symbol)
                stopped_out = True

        # Phase 2: strategy signal on the newly closed bar - mirrors main.py's
        # bounded get_bars(limit=strategy.min_bars + 5) window exactly.
        if not stopped_out:
            signal_window = w.bars.iloc[max(0, i + 1 - w.entry_window):i + 1]
            position = portfolio.get(symbol)  # re-fetch: may have just been closed above
            signal = w.strategy.generate_signal(signal_window, position.side if position else None)
            atr_value = _latest_atr(signal_window)

            if signal == Signal.EXIT and position is not None:
                self._close_position(symbol, position, close, ts, "signal", portfolio, result, equity_by_symbol)

            elif signal in (Signal.LONG, Signal.SHORT) and position is None:
                side = "long" if signal == Signal.LONG else "short"

                if self_managed:
                    stop_price = w.strategy.get_entry_stop_price(signal_window, side)
                    sizing_distance = abs(close - stop_price) if stop_price is not None else 0.0
                else:
                    sizing_distance = atr_value
                    stop_price = risk.initial_stop_price(close, atr_value, side)

                blocked = (
                    (side == "short" and not shorting_allowed(symbol, w.scfg, self.cfg))
                    or risk.correlation_blocked(symbol, side, portfolio)
                    or sizing_distance <= 0
                )
                if not blocked:
                    equity = equity_by_symbol[symbol]
                    qty = risk.position_size(equity, sizing_distance, close, self.cfg.risk_pct, w.scfg.asset_class)
                    qty = risk.cap_by_buying_power(qty, equity, close, w.scfg.max_allocation_pct, w.scfg.asset_class)
                    if qty > 0:
                        portfolio.open(symbol, side, qty, close, atr_value, stop_price)
                        self._entry_ts[symbol] = ts.isoformat()

        result.equity_curve.append(
            {"timestamp": ts.isoformat(), "equity": self._mark_to_market(symbol, close, portfolio, equity_by_symbol)}
        )

    def _mark_to_market(self, symbol: str, close: float, portfolio: Portfolio, equity_by_symbol: Dict[str, float]) -> float:
        equity = equity_by_symbol[symbol]
        position = portfolio.get(symbol)
        if position is None:
            return equity
        if position.side == "long":
            unrealized = (close - position.entry_price) * position.qty
        else:
            unrealized = (position.entry_price - close) * position.qty
        return equity + unrealized

    def _close_position(self, symbol, position, exit_price, exit_ts, reason, portfolio: Portfolio, result: BacktestResult, equity_by_symbol: Dict[str, float]) -> None:
        if position.side == "long":
            pnl = (exit_price - position.entry_price) * position.qty
        else:
            pnl = (position.entry_price - exit_price) * position.qty

        equity_by_symbol[symbol] += pnl
        result.trades.append(Trade(
            symbol=symbol,
            side=position.side,
            entry_time=self._entry_ts.get(symbol, position.opened_at),
            entry_price=position.entry_price,
            exit_time=exit_ts.isoformat(),
            exit_price=exit_price,
            qty=position.qty,
            pnl=pnl,
            exit_reason=reason,
        ))
        portfolio.remove(symbol)


def _print_report(symbol: str, strategy_name: str, start: datetime, end: datetime, result: BacktestResult) -> None:
    print(f"\n=== Backtest: {symbol} ({strategy_name}) ===")
    print(f"Period: {start.date()} -> {end.date()}")
    print(f"Bars evaluated: {len(result.equity_curve)}")
    print(f"Trades: {result.num_trades}")
    print(f"Win rate: {result.win_rate_pct:.1f}%")
    print(f"Total return: {result.total_return_pct:+.2f}%")
    print(f"Max drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Initial equity: ${result.initial_equity:,.2f}")
    if result.equity_curve:
        print(f"Final equity:   ${result.equity_curve[-1]['equity']:,.2f}")


def _save_equity_curve(result: BacktestResult, symbol: str, start: datetime, end: datetime, output_dir: str) -> str:
    safe_symbol = symbol.replace("/", "-")
    filename = f"{safe_symbol}_{start.date()}_{end.date()}_equity_curve.csv"
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "equity"])
        for point in result.equity_curve:
            writer.writerow([point["timestamp"], f"{point['equity']:.2f}"])
    return path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the bot's live strategies over historical Alpaca bar data.",
        epilog="Example: python -m bot.backtest --symbol SPY --start 2023-01-01 --end 2024-01-01",
    )
    parser.add_argument(
        "--symbol", action="append", dest="symbols", metavar="SYMBOL",
        help="Symbol to backtest (e.g. SPY, BTC/USD). Repeatable. Omit to backtest all 5 configured symbols.",
    )
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    parser.add_argument(
        "--equity", type=float, default=DEFAULT_INITIAL_EQUITY,
        help=f"Starting equity per symbol for the simulation (default: {DEFAULT_INITIAL_EQUITY:,.0f})",
    )
    parser.add_argument(
        "--output-dir", default="backtests",
        help="Directory to write equity-curve CSVs into (default: backtests/)",
    )
    parser.add_argument(
        "--strategy", dest="strategy_override", metavar="STRATEGY_NAME",
        help=(
            "Override which strategy to backtest, applied to every --symbol given in "
            "this run (e.g. --symbol GLD --strategy opening_range_breakout). Does not "
            "change config.py's live symbol->strategy mapping - only this one "
            "backtest invocation. Available: " + ", ".join(sorted(STRATEGY_REGISTRY))
        ),
    )
    parser.add_argument(
        "--trend-filter", dest="trend_filter", action="store_true",
        help=(
            "Enable opening_range_breakout's optional trend filter for this run "
            "(only has an effect together with --strategy opening_range_breakout; "
            "off by default, matching the strategy's own default)."
        ),
    )
    parser.add_argument(
        "--trend-filter-period", dest="trend_filter_period", type=int, metavar="MINUTES",
        help="Trend filter lookback in minutes/bars (default: 120). Implies nothing about --trend-filter itself.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config()
    client = AlpacaClient(cfg)

    symbols = args.symbols or list(cfg.symbols.keys())
    unknown = [s for s in symbols if s not in cfg.symbols]
    if unknown:
        raise SystemExit(f"Unknown symbol(s): {', '.join(unknown)}. Configured symbols: {', '.join(cfg.symbols)}")

    strategy_overrides: Dict[str, str] = {}
    if args.strategy_override:
        if args.strategy_override not in STRATEGY_REGISTRY:
            raise SystemExit(
                f"Unknown --strategy '{args.strategy_override}'. Available: {', '.join(sorted(STRATEGY_REGISTRY))}"
            )
        strategy_overrides = {s: args.strategy_override for s in symbols}

    strategy_override_params = {}
    if args.trend_filter:
        strategy_override_params["trend_filter_enabled"] = True
    if args.trend_filter_period is not None:
        strategy_override_params["trend_filter_period"] = args.trend_filter_period

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    if end <= start:
        raise SystemExit("--end must be after --start")

    os.makedirs(args.output_dir, exist_ok=True)

    engine = Backtester(
        cfg, client, initial_equity=args.equity,
        strategy_overrides=strategy_overrides, strategy_override_params=strategy_override_params,
    )
    results = engine.run(symbols, start, end)

    for symbol in symbols:
        effective_strategy_name = strategy_overrides.get(symbol, cfg.symbols[symbol].strategy)
        _print_report(symbol, effective_strategy_name, start, end, results[symbol])
        path = _save_equity_curve(results[symbol], symbol, start, end, args.output_dir)
        print(f"Equity curve saved to: {path}")


if __name__ == "__main__":
    main()
