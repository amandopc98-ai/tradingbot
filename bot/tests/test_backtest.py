from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from bot.backtest import Backtester, BacktestResult, DEFAULT_INITIAL_EQUITY, Trade, _lookback_start, _save_equity_curve, parse_args
from config import Config, SymbolConfig


class _FakeClient:
    """Stands in for AlpacaClient: get_bars_range ignores the requested
    start/end/timeframe and just returns whichever pre-built DataFrame was handed
    to it, so these tests never touch the network."""

    def __init__(self, bars_by_symbol):
        self.bars_by_symbol = bars_by_symbol

    def get_bars_range(self, symbol, timeframe, start, end, asset_class):
        return self.bars_by_symbol[symbol]


def _make_cfg(symbols: dict) -> Config:
    return Config(
        api_key="x", secret_key="y", paper=True,
        poll_interval_seconds=60, risk_pct=0.01, allow_shorting=True,
        state_file="unused.json", symbols=symbols,
    )


def _make_bars(idx, closes, volumes=None, spread=1.0):
    volumes = volumes or [1000.0] * len(closes)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + spread for c in closes],
            "low": [c - spread for c in closes],
            "close": closes,
            "volume": volumes,
        },
        index=pd.DatetimeIndex(idx),
    )


SPY_CFG = SymbolConfig(
    strategy="mean_reversion", timeframe=TimeFrame(15, TimeFrameUnit.Minute),
    asset_class="stock", params={"period": 20, "z_threshold": 1.5}, trail_atr_mult=None,
)
QQQ_CFG = SymbolConfig(
    strategy="mean_reversion", timeframe=TimeFrame(15, TimeFrameUnit.Minute),
    asset_class="stock", params={"period": 20, "z_threshold": 1.8}, trail_atr_mult=None,
)
BTC_CFG = SymbolConfig(
    strategy="momentum_breakout", timeframe=TimeFrame.Hour,
    asset_class="crypto", params={"period": 20, "volume_mult": 1.5}, trail_atr_mult=2.0,
)
GLD_CFG = SymbolConfig(  # GLD's real live mapping: trend_following, 4h - unrelated to the opening-range override tests below
    strategy="trend_following", timeframe=TimeFrame(4, TimeFrameUnit.Hour),
    asset_class="stock", params={"fast": 50, "slow": 200}, trail_atr_mult=3.0,
)


# ---------------------------------------------------------------------------
# BacktestResult metrics (pure math, no engine involved)
# ---------------------------------------------------------------------------

def test_backtest_result_metrics():
    result = BacktestResult(symbol="SPY", initial_equity=100_000.0)
    result.equity_curve = [
        {"timestamp": "t0", "equity": 100_000.0},
        {"timestamp": "t1", "equity": 110_000.0},  # peak
        {"timestamp": "t2", "equity": 99_000.0},   # trough: (110000-99000)/110000 = 10%
        {"timestamp": "t3", "equity": 105_000.0},
    ]
    result.trades = [
        Trade("SPY", "long", "t0", 100.0, "t1", 110.0, 10, 100.0, "signal"),
        Trade("SPY", "long", "t1", 110.0, "t2", 100.0, 10, -100.0, "signal"),
    ]
    assert result.num_trades == 2
    assert result.win_rate_pct == 50.0
    assert round(result.total_return_pct, 2) == 5.0
    assert round(result.max_drawdown_pct, 4) == round((110_000 - 99_000) / 110_000 * 100, 4)


def test_backtest_result_handles_no_trades():
    result = BacktestResult(symbol="SPY", initial_equity=100_000.0)
    assert result.num_trades == 0
    assert result.win_rate_pct == 0.0
    assert result.total_return_pct == 0.0
    assert result.max_drawdown_pct == 0.0


# ---------------------------------------------------------------------------
# _lookback_start
# ---------------------------------------------------------------------------

def test_lookback_start_pads_more_for_stocks_than_crypto():
    start = datetime(2023, 6, 1, tzinfo=timezone.utc)
    stock_lookback = _lookback_start(start, TimeFrame(15, TimeFrameUnit.Minute), 200, "stock")
    crypto_lookback = _lookback_start(start, TimeFrame(15, TimeFrameUnit.Minute), 200, "crypto")
    assert stock_lookback < crypto_lookback < start


# ---------------------------------------------------------------------------
# Engine: mean reversion long entry + exit at the mean
# ---------------------------------------------------------------------------

def test_backtest_engine_opens_and_exits_long_on_mean_reversion_dip():
    base = datetime(2023, 1, 3, tzinfo=timezone.utc)
    n = 60
    idx = [base + timedelta(minutes=15 * i) for i in range(n)]
    closes = [99.0 if i % 2 == 0 else 101.0 for i in range(n)]
    dip_i = 45
    closes[dip_i] = 60.0       # triggers a long entry (z far below -1.5)
    closes[dip_i + 3] = 100.0  # reverts to the mean -> strategy exit

    bars = _make_bars(idx, closes)
    cfg = _make_cfg({"SPY": SPY_CFG})
    client = _FakeClient({"SPY": bars})
    engine = Backtester(cfg, client, initial_equity=DEFAULT_INITIAL_EQUITY)

    results = engine.run(["SPY"], idx[40], idx[-1])
    result = results["SPY"]

    assert result.num_trades == 1
    trade = result.trades[0]
    assert trade.side == "long"
    assert trade.exit_reason == "signal"
    assert trade.pnl > 0
    assert len(result.equity_curve) > 0
    assert result.total_return_pct > 0


# ---------------------------------------------------------------------------
# Engine: hard ATR stop-loss fires when price keeps falling instead of reverting
# ---------------------------------------------------------------------------

def test_backtest_engine_stop_loss_closes_before_signal_exit():
    base = datetime(2023, 1, 3, tzinfo=timezone.utc)
    pattern = [99.0, 101.0] * 20  # 40 bars, mean ~100, real dispersion
    tail = [60.0, 55.0, 50.0]     # sharp drop that keeps falling - never reverts to the mean
    closes = pattern + tail
    idx = [base + timedelta(minutes=15 * i) for i in range(len(closes))]

    bars = _make_bars(idx, closes)
    cfg = _make_cfg({"SPY": SPY_CFG})
    client = _FakeClient({"SPY": bars})
    engine = Backtester(cfg, client, initial_equity=DEFAULT_INITIAL_EQUITY)

    results = engine.run(["SPY"], idx[35], idx[-1])
    result = results["SPY"]

    assert result.num_trades == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.pnl < 0


# ---------------------------------------------------------------------------
# Engine: correlation filter blocks a new BTC long when SPY and QQQ are both long,
# but only when they're actually backtested together (shared Portfolio).
# ---------------------------------------------------------------------------

def _correlation_scenario_bars():
    base = datetime(2023, 1, 3, tzinfo=timezone.utc)
    n = 40
    idx = [base + timedelta(minutes=15 * i) for i in range(n)]

    spy_closes = [99.0, 101.0] * 15 + [60.0] * 10   # dips and stays long from bar 30 on
    qqq_closes = [99.0, 101.0] * 15 + [50.0] * 10   # dips harder (needed for its 1.8 threshold) and stays long
    btc_closes = [100.0] * 35 + [110.0] * 5         # breaks out at bar 35, after SPY/QQQ are already long
    btc_vols = [1000.0] * 35 + [2000.0] * 5         # volume confirmation on the breakout bar onward

    return idx, _make_bars(idx, spy_closes), _make_bars(idx, qqq_closes), _make_bars(idx, btc_closes, btc_vols)


def test_correlation_filter_blocks_btc_when_backtested_with_spy_and_qqq():
    idx, spy_bars, qqq_bars, btc_bars = _correlation_scenario_bars()
    cfg = _make_cfg({"SPY": SPY_CFG, "QQQ": QQQ_CFG, "BTC/USD": BTC_CFG})
    client = _FakeClient({"SPY": spy_bars, "QQQ": qqq_bars, "BTC/USD": btc_bars})
    engine = Backtester(cfg, client, initial_equity=DEFAULT_INITIAL_EQUITY)

    results = engine.run(["SPY", "QQQ", "BTC/USD"], idx[25], idx[-1])

    assert results["SPY"].num_trades == 1
    assert results["QQQ"].num_trades == 1
    assert results["BTC/USD"].num_trades == 0  # blocked by the correlation filter


def test_correlation_filter_does_not_apply_when_btc_backtested_alone():
    idx, _, _, btc_bars = _correlation_scenario_bars()
    cfg = _make_cfg({"BTC/USD": BTC_CFG})
    client = _FakeClient({"BTC/USD": btc_bars})
    engine = Backtester(cfg, client, initial_equity=DEFAULT_INITIAL_EQUITY)

    results = engine.run(["BTC/USD"], idx[25], idx[-1])

    assert results["BTC/USD"].num_trades == 1  # nothing to block it this time


# ---------------------------------------------------------------------------
# Equity-curve CSV export
# ---------------------------------------------------------------------------

def test_save_equity_curve_writes_csv(tmp_path):
    result = BacktestResult(symbol="BTC/USD", initial_equity=100_000.0)
    result.equity_curve = [
        {"timestamp": "2023-01-01T00:00:00+00:00", "equity": 100_000.0},
        {"timestamp": "2023-01-01T01:00:00+00:00", "equity": 101_000.0},
    ]
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = datetime(2023, 1, 2, tzinfo=timezone.utc)

    path = _save_equity_curve(result, "BTC/USD", start, end, str(tmp_path))

    assert path.endswith("BTC-USD_2023-01-01_2023-01-02_equity_curve.csv")
    contents = open(path).read()
    assert "timestamp,equity" in contents
    assert "101000.00" in contents


# ---------------------------------------------------------------------------
# --strategy override: opening_range_breakout for GLD without touching GLD's live
# trend_following mapping in config.py
# ---------------------------------------------------------------------------

_ORB_DAY = "2026-07-09"

_ORB_LONG_SETUP = {
    "15:40": {"high": 101.0},
    "15:50": {"low": 99.0},
    "16:05": {"open": 100.0, "high": 106.0, "low": 100.0, "close": 105.0},
    "16:10": {"open": 105.0, "high": 105.5, "low": 101.5, "close": 102.0},
    "16:15": {"open": 102.0, "high": 107.5, "low": 102.0, "close": 107.0},
}


def _make_orb_day_bars(overrides: dict, end_time: str = "22:00", base_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(f"{_ORB_DAY} 15:29", f"{_ORB_DAY} {end_time}", freq="1min", tz="Europe/Berlin")
    df = pd.DataFrame(
        {"open": base_price, "high": base_price, "low": base_price, "close": base_price, "volume": 1000.0},
        index=idx,
    )
    for time_str, ohlc in overrides.items():
        ts = pd.Timestamp(f"{_ORB_DAY} {time_str}", tz="Europe/Berlin")
        for col, val in ohlc.items():
            df.loc[ts, col] = val
    return df


def test_strategy_override_uses_opening_range_breakout_for_gld():
    overrides = dict(_ORB_LONG_SETUP)
    overrides["16:20"] = {"open": 107.0, "high": 107.0, "low": 97.0, "close": 98.0}  # range-based stop (99) hit
    bars = _make_orb_day_bars(overrides)

    cfg = _make_cfg({"GLD": GLD_CFG})
    client = _FakeClient({"GLD": bars})
    engine = Backtester(
        cfg, client, initial_equity=DEFAULT_INITIAL_EQUITY, strategy_overrides={"GLD": "opening_range_breakout"}
    )

    start = datetime(2026, 7, 9, tzinfo=timezone.utc)
    end = datetime(2026, 7, 10, tzinfo=timezone.utc)
    results = engine.run(["GLD"], start, end)
    result = results["GLD"]

    assert result.num_trades == 1
    trade = result.trades[0]
    assert trade.side == "long"
    assert trade.entry_price == 107.0
    # Exits exactly at the range-based stop's triggering bar (98.0, at 16:20) - if the
    # generic ATR stop from Phase 1 had NOT been skipped, the tiny ATR from the mostly
    # flat pre-breakout data would have closed this at/near 107 on the very next bar
    # instead, so this pins down that self_managed_exits was actually honored.
    assert trade.exit_price == 98.0
    assert trade.exit_reason == "signal"
    assert trade.pnl < 0


def test_strategy_override_does_not_mutate_gld_live_config():
    cfg = _make_cfg({"GLD": GLD_CFG})
    client = _FakeClient({"GLD": _make_orb_day_bars(_ORB_LONG_SETUP)})
    engine = Backtester(cfg, client, strategy_overrides={"GLD": "opening_range_breakout"})

    engine.run(["GLD"], datetime(2026, 7, 9, tzinfo=timezone.utc), datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert cfg.symbols["GLD"].strategy == "trend_following"
    assert cfg.symbols["GLD"].timeframe == GLD_CFG.timeframe


def test_parse_args_strategy_override():
    args = parse_args(["--symbol", "GLD", "--strategy", "opening_range_breakout", "--start", "2026-07-06", "--end", "2026-07-10"])
    assert args.strategy_override == "opening_range_breakout"


def test_parse_args_strategy_override_defaults_to_none():
    args = parse_args(["--symbol", "GLD", "--start", "2026-07-06", "--end", "2026-07-10"])
    assert args.strategy_override is None


def test_parse_args_trend_filter_flags():
    args = parse_args([
        "--symbol", "GLD", "--strategy", "opening_range_breakout",
        "--trend-filter", "--trend-filter-period", "30",
        "--start", "2026-07-06", "--end", "2026-07-10",
    ])
    assert args.trend_filter is True
    assert args.trend_filter_period == 30


def test_parse_args_trend_filter_flags_default_off():
    args = parse_args(["--symbol", "GLD", "--start", "2026-07-06", "--end", "2026-07-10"])
    assert args.trend_filter is False
    assert args.trend_filter_period is None


def _make_prior_day_tail(day: str, n_bars: int, price: float, end_time: str = "22:00") -> pd.DataFrame:
    end_ts = pd.Timestamp(f"{day} {end_time}", tz="Europe/Berlin")
    idx = pd.date_range(end=end_ts, periods=n_bars, freq="1min", tz="Europe/Berlin")
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1000.0}, index=idx)


def test_backtester_threads_strategy_override_params_into_the_strategy():
    # Same "blocked" scenario as the strategy-level trend-filter tests, but driven
    # through the full Backtester.run() pipeline this time, to prove --trend-filter /
    # --trend-filter-period (via strategy_override_params) actually reach the
    # constructed strategy instance rather than being silently dropped.
    prior = _make_prior_day_tail("2026-07-08", n_bars=10, price=300.0)
    today = _make_orb_day_bars(_ORB_LONG_SETUP)
    bars = pd.concat([prior, today]).sort_index()

    cfg = _make_cfg({"GLD": GLD_CFG})
    start = datetime(2026, 7, 6, tzinfo=timezone.utc)
    end = datetime(2026, 7, 10, tzinfo=timezone.utc)

    without_filter = Backtester(
        cfg, _FakeClient({"GLD": bars}),
        strategy_overrides={"GLD": "opening_range_breakout"},
    )
    with_filter = Backtester(
        cfg, _FakeClient({"GLD": bars}),
        strategy_overrides={"GLD": "opening_range_breakout"},
        strategy_override_params={"trend_filter_enabled": True, "trend_filter_period": 50},
    )

    assert without_filter.run(["GLD"], start, end)["GLD"].num_trades == 1
    assert with_filter.run(["GLD"], start, end)["GLD"].num_trades == 0
