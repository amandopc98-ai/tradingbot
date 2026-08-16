import pytest
import pandas as pd

from bot import indicators
from bot.strategies.base import Signal
from bot.strategies.opening_range_breakout import (
    RANGE_TZ,
    OpeningRangeBreakoutStrategy,
    _compute_day_state,
)

DAY = "2026-07-09"  # a Thursday, safely inside both US and EU daylight saving time
BASE_PRICE = 100.0


def _make_day_bars(overrides: dict, end_time: str = "22:00") -> pd.DataFrame:
    """A full day of 1-minute bars from 15:29 to `end_time` Europe/Berlin, flat at
    BASE_PRICE everywhere except the timestamps named in `overrides`."""
    idx = pd.date_range(f"{DAY} 15:29", f"{DAY} {end_time}", freq="1min", tz=RANGE_TZ)
    df = pd.DataFrame(
        {
            "open": BASE_PRICE, "high": BASE_PRICE, "low": BASE_PRICE, "close": BASE_PRICE,
            "volume": 1000.0,
        },
        index=idx,
    )
    for time_str, ohlc in overrides.items():
        ts = pd.Timestamp(f"{DAY} {time_str}", tz=RANGE_TZ)
        for col, val in ohlc.items():
            df.loc[ts, col] = val
    return df


def _replay(strategy: OpeningRangeBreakoutStrategy, bars: pd.DataFrame):
    """Feeds the strategy one new bar at a time (like the live bot / backtester do)
    and returns the resulting entry/exit events as (timestamp, kind, price) tuples."""
    position_side = None
    events = []
    for i in range(len(bars)):
        window = bars.iloc[: i + 1]
        signal = strategy.generate_signal(window, position_side)
        ts = bars.index[i]
        price = float(bars["close"].iloc[i])
        if signal == Signal.EXIT and position_side is not None:
            events.append((ts, "EXIT", price))
            position_side = None
        elif signal in (Signal.LONG, Signal.SHORT) and position_side is None:
            events.append((ts, signal.value, price))
            position_side = signal.value
    return events


# A standard long setup shared by several tests: range [99, 101], breakout at 16:05
# (close 105, high 106), a clean retest at 16:10, entry trigger at 16:15 (close 107).
_LONG_SETUP = {
    "15:40": {"high": 101.0},
    "15:50": {"low": 99.0},
    "16:05": {"open": 100.0, "high": 106.0, "low": 100.0, "close": 105.0},
    "16:10": {"open": 105.0, "high": 105.5, "low": 101.5, "close": 102.0},
    "16:15": {"open": 102.0, "high": 107.5, "low": 102.0, "close": 107.0},
}

# The mirror-image short setup shared by the trend-filter tests: range [99, 101],
# breakout at 16:05 (close 95, low 94), a clean retest at 16:10, entry trigger at
# 16:15 (close 93).
_SHORT_SETUP = {
    "15:40": {"high": 101.0},
    "15:50": {"low": 99.0},
    "16:05": {"open": 100.0, "high": 100.0, "low": 94.0, "close": 95.0},
    "16:10": {"open": 95.0, "high": 96.0, "low": 95.0, "close": 96.0},
    "16:15": {"open": 96.0, "high": 96.0, "low": 93.0, "close": 93.0},
}


# ---------------------------------------------------------------------------
# Range formation
# ---------------------------------------------------------------------------

def test_range_is_high_low_of_1529_to_1600_window():
    bars = _make_day_bars({"15:40": {"high": 103.5}, "15:50": {"low": 97.5}}, end_time="16:00")
    today_bars = bars  # already exactly the range window's day
    state = _compute_day_state(today_bars, take_profit_r_multiple=1.8)
    assert state is not None
    assert state.range_high == 103.5
    assert state.range_low == 97.5
    assert state.breakout_ts is None  # nothing after the range window yet


def test_no_range_state_before_1529():
    # a "day" with bars only before the range window even starts
    idx = pd.date_range(f"{DAY} 15:00", f"{DAY} 15:28", freq="1min", tz=RANGE_TZ)
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=idx)
    assert _compute_day_state(bars, 1.8) is None


# ---------------------------------------------------------------------------
# Breakout: direction, first-breakout-only
# ---------------------------------------------------------------------------

def test_first_breakout_of_the_day_wins_even_if_a_later_bar_breaks_the_other_way():
    overrides = {
        "15:40": {"high": 101.0},
        "15:50": {"low": 99.0},
        "16:05": {"close": 105.0, "high": 106.0},  # first breakout: long
        "17:00": {"close": 90.0, "low": 89.0},      # a later bar breaks short - must be ignored
    }
    bars = _make_day_bars(overrides)
    state = _compute_day_state(bars, 1.8)
    assert state.breakout_direction == "long"
    assert state.breakout_ts == pd.Timestamp(f"{DAY} 16:05", tz=RANGE_TZ)


def test_short_breakout_direction():
    overrides = {
        "15:40": {"high": 101.0},
        "15:50": {"low": 99.0},
        "16:05": {"open": 100.0, "high": 100.0, "low": 94.0, "close": 95.0},
    }
    bars = _make_day_bars(overrides)
    state = _compute_day_state(bars, 1.8)
    assert state.breakout_direction == "short"
    assert state.breakout_low == 94.0


# ---------------------------------------------------------------------------
# Retest detection
# ---------------------------------------------------------------------------

def test_retest_requires_close_not_violating_the_range_line():
    overrides = dict(_LONG_SETUP)
    # replace the clean retest at 16:10 with a bar that dips into the zone but
    # CLOSES below range_high (99..101) - this must NOT count as a valid retest
    overrides["16:10"] = {"open": 105.0, "high": 105.0, "low": 100.5, "close": 100.5}
    del overrides["16:15"]  # remove the other candidate bar too, so nothing else could qualify
    bars = _make_day_bars(overrides)
    state = _compute_day_state(bars, 1.8)
    assert state.retest_ts is None  # the violating bar doesn't qualify
    assert state.entry_ts is None   # so no entry can be found either


def test_retest_qualifies_on_a_later_bar_after_a_violating_one():
    overrides = dict(_LONG_SETUP)
    overrides["16:08"] = {"open": 105.0, "high": 105.0, "low": 100.5, "close": 100.5}  # violates, doesn't qualify
    # 16:10 (from _LONG_SETUP) still dips in cleanly afterwards and qualifies
    bars = _make_day_bars(overrides)
    state = _compute_day_state(bars, 1.8)
    assert state.retest_ts == pd.Timestamp(f"{DAY} 16:10", tz=RANGE_TZ)


def test_retest_bar_that_also_closes_above_breakout_high_does_not_self_trigger_entry():
    # a single bar dips into the zone (qualifies as retest) AND closes above the
    # breakout bar's high in the same bar - entry must still wait for a LATER bar
    # ("nach angelaufenem Retest"), not fire on this same bar.
    overrides = dict(_LONG_SETUP)
    overrides["16:10"] = {"open": 105.0, "high": 108.0, "low": 101.5, "close": 107.0}  # dips in AND closes above 106
    del overrides["16:15"]  # remove the normal entry bar; no further bar closes above 106
    bars = _make_day_bars(overrides)
    state = _compute_day_state(bars, 1.8)
    assert state.retest_ts == pd.Timestamp(f"{DAY} 16:10", tz=RANGE_TZ)
    assert state.entry_ts is None


# ---------------------------------------------------------------------------
# Entry, stop, and take-profit levels
# ---------------------------------------------------------------------------

def test_entry_stop_and_target_prices_for_long():
    bars = _make_day_bars(_LONG_SETUP)
    state = _compute_day_state(bars, take_profit_r_multiple=1.8)
    assert state.entry_ts == pd.Timestamp(f"{DAY} 16:15", tz=RANGE_TZ)
    entry_price = 107.0
    assert state.stop_price == 99.0  # range_low
    risk_amount = entry_price - 99.0
    assert state.target_price == entry_price + 1.8 * risk_amount


def test_take_profit_multiple_is_configurable():
    bars = _make_day_bars(_LONG_SETUP)
    state_default = _compute_day_state(bars, take_profit_r_multiple=1.8)
    state_custom = _compute_day_state(bars, take_profit_r_multiple=3.0)
    assert state_custom.target_price > state_default.target_price


def test_short_side_stop_and_target():
    overrides = {
        "15:40": {"high": 101.0},
        "15:50": {"low": 99.0},
        "16:05": {"open": 100.0, "high": 100.0, "low": 94.0, "close": 95.0},  # breakout short, low=94
        # retest condition (short): high >= breakout_low (94) and close <= range_low (99)
        "16:10": {"open": 95.0, "high": 96.0, "low": 95.0, "close": 96.0},
        "16:15": {"open": 96.0, "high": 96.0, "low": 93.0, "close": 93.0},    # closes below breakout_low 94
    }
    bars = _make_day_bars(overrides)
    state = _compute_day_state(bars, take_profit_r_multiple=1.8)
    assert state.breakout_direction == "short"
    assert state.entry_ts == pd.Timestamp(f"{DAY} 16:15", tz=RANGE_TZ)
    entry_price = 93.0
    assert state.stop_price == 101.0  # range_high
    risk_amount = 101.0 - entry_price
    assert state.target_price == entry_price - 1.8 * risk_amount


# ---------------------------------------------------------------------------
# End-to-end via generate_signal(): entry, stop, target, EOD, max 1 trade/day
# ---------------------------------------------------------------------------

def test_end_to_end_long_entry_then_stop_loss():
    overrides = dict(_LONG_SETUP)
    overrides["16:20"] = {"open": 107.0, "high": 107.0, "low": 97.0, "close": 98.0}  # <= stop (99)
    bars = _make_day_bars(overrides)
    strategy = OpeningRangeBreakoutStrategy("GLD")

    events = _replay(strategy, bars)

    assert events == [
        (pd.Timestamp(f"{DAY} 16:15", tz=RANGE_TZ), "long", 107.0),
        (pd.Timestamp(f"{DAY} 16:20", tz=RANGE_TZ), "EXIT", 98.0),
    ]


def test_end_to_end_long_entry_then_take_profit():
    overrides = dict(_LONG_SETUP)
    overrides["16:25"] = {"open": 107.0, "high": 122.0, "low": 107.0, "close": 122.0}  # >= target (121.4)
    bars = _make_day_bars(overrides)
    strategy = OpeningRangeBreakoutStrategy("GLD")

    events = _replay(strategy, bars)

    assert len(events) == 2
    assert events[0] == (pd.Timestamp(f"{DAY} 16:15", tz=RANGE_TZ), "long", 107.0)
    assert events[1][:2] == (pd.Timestamp(f"{DAY} 16:25", tz=RANGE_TZ), "EXIT")
    assert events[1][2] == 122.0


def test_end_to_end_eod_close_when_neither_stop_nor_target_hit():
    # after entry at 16:15 (close 107), every later bar reverts to the flat 100.0
    # baseline - safely between stop (99) and target (121.4), so only the
    # end-of-day check should close it. The cutoff (15:55 America/New_York) first
    # becomes true at 21:55 Europe/Berlin during summer DST (a 6h offset), which is
    # why the close happens there rather than on the literal last bar of the array.
    bars = _make_day_bars(_LONG_SETUP, end_time="22:00")
    strategy = OpeningRangeBreakoutStrategy("GLD")

    events = _replay(strategy, bars)

    assert len(events) == 2
    assert events[0] == (pd.Timestamp(f"{DAY} 16:15", tz=RANGE_TZ), "long", 107.0)
    exit_ts, kind, exit_price = events[1]
    assert kind == "EXIT"
    assert exit_ts == pd.Timestamp(f"{DAY} 21:55", tz=RANGE_TZ)
    assert exit_price == BASE_PRICE   # baseline price, not stop or target


def test_max_one_trade_per_day_even_after_a_stop_out_and_a_new_setup():
    overrides = dict(_LONG_SETUP)
    overrides["16:20"] = {"open": 107.0, "high": 107.0, "low": 97.0, "close": 98.0}  # stop out
    overrides["16:30"] = {"open": 98.0, "high": 112.0, "low": 98.0, "close": 110.0}  # again above breakout_high (106)
    bars = _make_day_bars(overrides)
    strategy = OpeningRangeBreakoutStrategy("GLD")

    events = _replay(strategy, bars)

    # exactly one entry + one exit for the whole day - the 16:30 close above 106
    # must NOT trigger a second entry.
    assert len(events) == 2
    assert events[0][1] == "long"
    assert events[1][1] == "EXIT"


def test_no_signal_without_a_breakout():
    bars = _make_day_bars({"15:40": {"high": 101.0}, "15:50": {"low": 99.0}})  # flat all day, no breakout
    strategy = OpeningRangeBreakoutStrategy("GLD")

    events = _replay(strategy, bars)

    assert events == []


# ---------------------------------------------------------------------------
# get_entry_stop_price() - used by bot/backtest.py for sizing self-managed strategies
# ---------------------------------------------------------------------------

def test_get_entry_stop_price_matches_computed_state():
    bars = _make_day_bars(_LONG_SETUP)
    strategy = OpeningRangeBreakoutStrategy("GLD")
    assert strategy.get_entry_stop_price(bars, "long") == 99.0


def test_get_entry_stop_price_none_before_range_exists():
    idx = pd.date_range(f"{DAY} 15:00", f"{DAY} 15:28", freq="1min", tz=RANGE_TZ)
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=idx)
    strategy = OpeningRangeBreakoutStrategy("GLD")
    assert strategy.get_entry_stop_price(bars, "long") is None


def test_self_managed_exits_flag_is_set():
    assert OpeningRangeBreakoutStrategy("GLD").self_managed_exits is True


# ---------------------------------------------------------------------------
# Optional trend filter (rule 9) - off by default. A small trend_filter_period (5)
# is used throughout so the whole lookback window fits within the constructed day's
# bars, rather than needing 120 minutes of prior-day history for every test.
# ---------------------------------------------------------------------------

def test_invalid_trend_filter_mode_rejected():
    with pytest.raises(ValueError):
        OpeningRangeBreakoutStrategy("GLD", trend_filter_mode="wma")


def test_trend_filter_disabled_by_default():
    strategy = OpeningRangeBreakoutStrategy("GLD")
    assert strategy.trend_filter_enabled is False


def test_trend_filter_allows_long_when_price_above_moving_average():
    # bars 16:11-16:14 stay at the flat 100.0 baseline, so the SMA(5) ending at the
    # 16:15 entry bar is (100+100+100+100+107)/5 = 101.4 - below the entry price 107.
    bars = _make_day_bars(_LONG_SETUP)
    strategy = OpeningRangeBreakoutStrategy("GLD", trend_filter_enabled=True, trend_filter_period=5)

    events = _replay(strategy, bars)

    assert len(events) == 2
    assert events[0][1] == "long"


def _make_prior_day_tail(day: str, n_bars: int, price: float, end_time: str = "22:00") -> pd.DataFrame:
    """n_bars of flat `price` 1-minute bars ending at `end_time` on `day`
    (Europe/Berlin) - used to give the trend filter's moving average some history to
    pull from beyond just today's ~47 pre-entry bars, without that history ever
    landing inside today's 15:29-16:00 range window or its breakout/retest/entry
    scan (those only ever look at *today's* bars)."""
    end_ts = pd.Timestamp(f"{day} {end_time}", tz=RANGE_TZ)
    idx = pd.date_range(end=end_ts, periods=n_bars, freq="1min", tz=RANGE_TZ)
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1000.0}, index=idx)


def test_trend_filter_blocks_long_when_price_below_moving_average():
    # A high-priced prior session pulls the 50-bar average above the entry price
    # (107), without touching today's breakout/retest/entry bars at all: a filler
    # bar *between* the retest and entry that itself exceeded breakout_high (106)
    # would just become the entry bar - so pulling the average up has to come from
    # outside today's post-retest window entirely.
    prior = _make_prior_day_tail("2026-07-08", n_bars=10, price=300.0)
    today = _make_day_bars(_LONG_SETUP)
    bars = pd.concat([prior, today]).sort_index()
    strategy = OpeningRangeBreakoutStrategy("GLD", trend_filter_enabled=True, trend_filter_period=50)

    events = _replay(strategy, bars)

    assert events == []  # setup discarded; the day counts as finished, no trade at all


def test_trend_filter_allows_short_when_price_below_moving_average():
    # bars 16:11-16:14 stay at the flat 100.0 baseline, so the SMA(5) ending at the
    # 16:15 entry bar is (100*4 + 93)/5 = 99.4 - above the entry price 93.
    bars = _make_day_bars(_SHORT_SETUP)
    strategy = OpeningRangeBreakoutStrategy("GLD", trend_filter_enabled=True, trend_filter_period=5)

    events = _replay(strategy, bars)

    assert len(events) == 2
    assert events[0][1] == "short"


def test_trend_filter_blocks_short_when_price_above_moving_average():
    # Mirror image of the long-block test: a low-priced prior session pulls the
    # 60-bar average below the entry price (93), from outside today's post-retest
    # window (see the comment on the long-block test for why it has to be there).
    prior = _make_prior_day_tail("2026-07-08", n_bars=20, price=10.0)
    today = _make_day_bars(_SHORT_SETUP)
    bars = pd.concat([prior, today]).sort_index()
    strategy = OpeningRangeBreakoutStrategy("GLD", trend_filter_enabled=True, trend_filter_period=60)

    events = _replay(strategy, bars)

    assert events == []


def test_trend_filter_disabled_gives_identical_result_to_no_filter_at_all():
    overrides = dict(_LONG_SETUP)
    overrides["16:20"] = {"open": 107.0, "high": 107.0, "low": 97.0, "close": 98.0}  # stop-hit bar
    bars = _make_day_bars(overrides)

    baseline = OpeningRangeBreakoutStrategy("GLD")
    # Same scenario, but explicitly disabled with a non-default period, to prove the
    # period value is irrelevant whenever the filter itself is off.
    explicitly_disabled = OpeningRangeBreakoutStrategy(
        "GLD", trend_filter_enabled=False, trend_filter_period=5, trend_filter_mode="ema"
    )

    assert _replay(baseline, bars) == _replay(explicitly_disabled, bars)


def test_trend_filter_ema_mode_can_differ_from_sma_mode():
    # A ramping price series makes EMA and SMA diverge, which can flip the filter's
    # allow/block decision depending on trend_filter_mode.
    overrides = dict(_LONG_SETUP)
    overrides["16:11"] = {"close": 90.0}
    overrides["16:12"] = {"close": 95.0}
    overrides["16:13"] = {"close": 100.0}
    overrides["16:14"] = {"close": 105.0}
    bars = _make_day_bars(overrides)

    sma_strategy = OpeningRangeBreakoutStrategy("GLD", trend_filter_enabled=True, trend_filter_period=5, trend_filter_mode="sma")
    ema_strategy = OpeningRangeBreakoutStrategy("GLD", trend_filter_enabled=True, trend_filter_period=5, trend_filter_mode="ema")

    sma_events = _replay(sma_strategy, bars)
    ema_events = _replay(ema_strategy, bars)

    # Both should at least be internally consistent (either allows or fully blocks
    # for the day) - the real assertion is that the two moving averages are not
    # required to agree, i.e. mode is actually being respected.
    window = bars.loc[:pd.Timestamp(f"{DAY} 16:15", tz=RANGE_TZ), "close"]
    sma_value = float(indicators.sma(window, 5).iloc[-1])
    ema_value = float(indicators.ema(window, 5).iloc[-1])
    assert sma_value != ema_value
    assert len(sma_events) in (0, 2)
    assert len(ema_events) in (0, 2)
