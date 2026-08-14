import numpy as np
import pandas as pd

from strategies.base import Signal
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.trend_following import TrendFollowingStrategy


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

def _mr_bars(tail_close):
    # Mildly oscillating base (mean 100, real dispersion) rather than a flat series,
    # so a single extra bar doesn't collapse the rolling std to ~0 and blow up z.
    closes = [99.0, 101.0] * 10 + [tail_close]
    return pd.DataFrame({"close": closes})


def test_mean_reversion_long_on_extreme_dip():
    strat = MeanReversionStrategy("SPY", period=20, z_threshold=1.5)
    bars = _mr_bars(60.0)
    assert strat.generate_signal(bars, position_side=None) == Signal.LONG


def test_mean_reversion_short_on_extreme_spike():
    strat = MeanReversionStrategy("SPY", period=20, z_threshold=1.5)
    bars = _mr_bars(140.0)
    assert strat.generate_signal(bars, position_side=None) == Signal.SHORT


def test_mean_reversion_holds_within_band():
    strat = MeanReversionStrategy("SPY", period=20, z_threshold=1.5)
    bars = _mr_bars(100.5)
    assert strat.generate_signal(bars, position_side=None) == Signal.HOLD


def test_mean_reversion_exits_long_at_mean():
    strat = MeanReversionStrategy("SPY", period=20, z_threshold=1.5)
    bars = _mr_bars(100.1)  # last close lands right at/above the rolling mean
    assert strat.generate_signal(bars, position_side="long") == Signal.EXIT


def test_mean_reversion_higher_threshold_needs_bigger_move():
    bars = _mr_bars(102.0)  # z ~= 1.7: clears a 1.5 threshold but not a 1.8 one
    loose = MeanReversionStrategy("SPY", period=20, z_threshold=1.5)
    tight = MeanReversionStrategy("QQQ", period=20, z_threshold=1.8)
    assert loose.generate_signal(bars, position_side=None) == Signal.SHORT
    assert tight.generate_signal(bars, position_side=None) == Signal.HOLD


def test_mean_reversion_insufficient_bars_holds():
    strat = MeanReversionStrategy("SPY", period=20, z_threshold=1.5)
    bars = pd.DataFrame({"close": [100.0] * 5})
    assert strat.generate_signal(bars, position_side=None) == Signal.HOLD


# ---------------------------------------------------------------------------
# Momentum breakout
# ---------------------------------------------------------------------------

def _momentum_bars(last_close, last_high, last_low, last_volume):
    n = 21
    closes = [100.0] * n
    highs = [101.0] * n
    lows = [99.0] * n
    volumes = [1000.0] * n
    closes.append(last_close)
    highs.append(last_high)
    lows.append(last_low)
    volumes.append(last_volume)
    return pd.DataFrame({"close": closes, "high": highs, "low": lows, "volume": volumes})


def test_momentum_breakout_long_on_volume_confirmed_high_break():
    strat = MomentumBreakoutStrategy("BTC/USD", period=20, volume_mult=1.5)
    bars = _momentum_bars(last_close=105, last_high=105, last_low=104, last_volume=2000)
    assert strat.generate_signal(bars, position_side=None) == Signal.LONG


def test_momentum_breakout_holds_without_volume_confirmation():
    strat = MomentumBreakoutStrategy("BTC/USD", period=20, volume_mult=1.5)
    bars = _momentum_bars(last_close=105, last_high=105, last_low=104, last_volume=1100)
    assert strat.generate_signal(bars, position_side=None) == Signal.HOLD


def test_momentum_breakout_short_on_volume_confirmed_low_break():
    strat = MomentumBreakoutStrategy("BTC/USD", period=20, volume_mult=1.5)
    bars = _momentum_bars(last_close=95, last_high=96, last_low=94, last_volume=2000)
    assert strat.generate_signal(bars, position_side=None) == Signal.SHORT


def test_momentum_breakout_exits_long_on_breakdown():
    strat = MomentumBreakoutStrategy("BTC/USD", period=20, volume_mult=1.5)
    bars = _momentum_bars(last_close=95, last_high=96, last_low=94, last_volume=2000)
    assert strat.generate_signal(bars, position_side="long") == Signal.EXIT


def test_momentum_breakout_insufficient_bars_holds():
    strat = MomentumBreakoutStrategy("BTC/USD", period=20, volume_mult=1.5)
    bars = pd.DataFrame({"close": [100.0] * 5, "high": [101.0] * 5, "low": [99.0] * 5, "volume": [1000.0] * 5})
    assert strat.generate_signal(bars, position_side=None) == Signal.HOLD


# ---------------------------------------------------------------------------
# Trend following
# ---------------------------------------------------------------------------

def _find_crossover(closes: pd.Series, fast: int, slow: int, direction: str):
    """Returns the index (as an integer position) of the first bar where the fast/slow
    EMA crossover of `direction` ("up" or "down") occurs, matching the strategy's own
    cross_up/cross_down definition."""
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    for i in range(1, len(closes)):
        f_now, f_prev = ema_fast.iloc[i], ema_fast.iloc[i - 1]
        s_now, s_prev = ema_slow.iloc[i], ema_slow.iloc[i - 1]
        if direction == "up" and f_prev <= s_prev and f_now > s_now:
            return i
        if direction == "down" and f_prev >= s_prev and f_now < s_now:
            return i
    raise AssertionError(f"no {direction} crossover found in synthetic series")


def test_trend_following_long_on_golden_cross():
    fast, slow = 5, 20
    strat = TrendFollowingStrategy("GLD", fast=fast, slow=slow)
    closes = pd.Series([50.0] * 40 + [150.0] * 40)
    idx = _find_crossover(closes, fast, slow, "up")
    bars = pd.DataFrame({"close": closes.iloc[: idx + 1]})
    assert strat.generate_signal(bars, position_side=None) == Signal.LONG


def test_trend_following_short_on_death_cross():
    fast, slow = 5, 20
    strat = TrendFollowingStrategy("GLD", fast=fast, slow=slow)
    closes = pd.Series([150.0] * 40 + [50.0] * 40)
    idx = _find_crossover(closes, fast, slow, "down")
    bars = pd.DataFrame({"close": closes.iloc[: idx + 1]})
    assert strat.generate_signal(bars, position_side=None) == Signal.SHORT


def test_trend_following_exits_long_on_death_cross():
    fast, slow = 5, 20
    strat = TrendFollowingStrategy("GLD", fast=fast, slow=slow)
    closes = pd.Series([150.0] * 40 + [50.0] * 40)
    idx = _find_crossover(closes, fast, slow, "down")
    bars = pd.DataFrame({"close": closes.iloc[: idx + 1]})
    assert strat.generate_signal(bars, position_side="long") == Signal.EXIT


def test_trend_following_insufficient_bars_holds():
    strat = TrendFollowingStrategy("GLD", fast=50, slow=200)
    bars = pd.DataFrame({"close": [100.0] * 10})
    assert strat.generate_signal(bars, position_side=None) == Signal.HOLD
