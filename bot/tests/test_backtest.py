from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from bot.backtest import Backtester, BacktestResult, DEFAULT_INITIAL_EQUITY, Trade, _lookback_start, _save_equity_curve
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
