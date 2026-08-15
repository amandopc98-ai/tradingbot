import numpy as np
import pandas as pd

from bot import indicators


def make_ohlcv(closes, high_offset=0.5, low_offset=0.5, volume=1000):
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + high_offset,
            "low": closes - low_offset,
            "close": closes,
            "volume": [volume] * len(closes),
        }
    )


def test_sma_matches_manual_average():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = indicators.sma(s, 3)
    assert np.isclose(result.iloc[-1], (3 + 4 + 5) / 3)
    assert pd.isna(result.iloc[0])


def test_rolling_std_zero_for_constant_series():
    s = pd.Series([10.0] * 25)
    result = indicators.rolling_std(s, 20)
    assert result.iloc[-1] == 0


def test_ema_reacts_faster_than_longer_span():
    s = pd.Series(list(range(1, 101)), dtype=float)
    fast = indicators.ema(s, 5)
    slow = indicators.ema(s, 50)
    # in a steady uptrend the fast EMA should track closer to the latest price
    assert abs(fast.iloc[-1] - s.iloc[-1]) < abs(slow.iloc[-1] - s.iloc[-1])


def test_rolling_high_low():
    s = pd.Series([1, 5, 2, 8, 3], dtype=float)
    assert indicators.rolling_high(s, 3).iloc[-1] == 8
    assert indicators.rolling_low(s, 3).iloc[-1] == 2


def test_avg_volume():
    v = pd.Series([100, 200, 300], dtype=float)
    assert indicators.avg_volume(v, 3).iloc[-1] == 200


def test_atr_positive_for_moving_series():
    df = make_ohlcv(np.linspace(100, 120, 30))
    result = indicators.atr(df, period=14)
    assert result.iloc[-1] > 0
    assert pd.isna(result.iloc[0])  # not enough data yet


def test_atr_reflects_wider_true_range():
    tight = make_ohlcv(np.full(30, 100.0), high_offset=0.1, low_offset=0.1)
    wide = make_ohlcv(np.full(30, 100.0), high_offset=5.0, low_offset=5.0)
    assert indicators.atr(wide, 14).iloc[-1] > indicators.atr(tight, 14).iloc[-1]
