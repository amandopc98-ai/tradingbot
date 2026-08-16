"""Opening-range breakout-and-retest strategy (GLD-oriented, but not GLD-specific).

Rules (as specified):
  1. Range: the highest high / lowest low of the 1-minute bars between 15:29 and
     16:00 Europe/Berlin time on a given trading day.
  2. Breakout: the first subsequent 1-minute bar that *closes* beyond either range
     boundary sets that day's direction (long/short) and "breakout bar". Only the
     first breakout of the day counts - a later close through the opposite boundary
     is ignored.
  3. Retest zone: for a long breakout, [range_high, breakout_bar.high]; for a short
     breakout, [breakout_bar.low, range_low].
  4. Retest: a later bar whose low dips into the zone (long) / high pokes into the
     zone (short) *and* whose close doesn't violate the broken range line (long:
     close >= range_high; short: close <= range_low). This is a per-bar check, not
     a running invalidation - a single bar either qualifies as "the retest" or it
     doesn't.
  5. Entry: the first bar *after* the retest bar whose close breaks back above the
     breakout bar's high (long) / below its low (short).
  6. Stop-loss: the opposite range boundary (long: range_low, short: range_high) -
     fixed, not ATR-based.
  7. Take-profit: entry +/- `take_profit_r_multiple` x (entry - stop distance),
     default 1.8.
  8. At most one trade per day; any open position is force-closed at end of day.
  9. Optional trend filter (off by default - trend_filter_enabled=False preserves
     the exact behavior above): if enabled, at the moment an entry would otherwise
     fire, compute an SMA or EMA (trend_filter_mode) over the last
     trend_filter_period 1-minute bars (default 120) ending at the entry bar. A long
     entry requires price above that average, a short entry requires price below it;
     otherwise the setup is discarded for the day - like a rejected entry, this does
     not roll forward to a later bar (see "stateless" note below for why).

Interface: identical to every other strategy - generate_signal(bars, position_side)
-> Signal, using only the shared Signal enum (LONG/SHORT/EXIT/HOLD). No changes to
bot/strategies/base.py were needed.

Why this needs bot/backtest.py's opt-in "self_managed_exits" hook: the stop-loss and
take-profit here are fixed price levels derived from the day's range, not an ATR
multiple - they don't fit bot/risk_manager.py's ATR-based hard-stop model at all. So
this strategy manages its own exits entirely through its own Signal.EXIT (computed
fresh from the day's bars on every call, exactly like every other decision here), and
declares `self_managed_exits = True` so bot/backtest.py skips the generic ATR stop
check for its positions instead of letting a wrong ATR-based stop fire first. See
get_entry_stop_price() below for the other half of that hook (sizing at entry).

Day handling is entirely stateless, like the other strategies: every call re-derives
today's range/breakout/retest/entry from the bars it's given, by filtering to
"today" (the local date of the most recent bar) and scanning forward in time. This
has a useful side effect for rule 8: the day's entry bar is a deterministic function
of that day's price history, so a LONG/SHORT signal only ever fires on the exact bar
where the entry condition first becomes true - never again afterwards, even if the
setup conditions are technically still "true" on a later bar. No explicit
"already traded today" flag is needed.

Two independent local times are used deliberately:
  - Europe/Berlin for the range window and for grouping bars into "days", matching
    how the range is specified and because the regular US trading session never
    crosses a Berlin midnight.
  - America/New_York for the end-of-day cutoff, because "market close" is a fixed
    16:00 in the exchange's own local time; anchoring it to a fixed Berlin clock
    time would drift during the ~1-3 weeks each spring/autumn when US and EU DST
    transitions don't land on the same date.
"""
from dataclasses import dataclass
from datetime import time as dt_time
from typing import Optional

import pandas as pd

from bot import indicators
from bot.strategies.base import Signal, Strategy

RANGE_TZ = "Europe/Berlin"
RANGE_WINDOW_START = "15:29"
RANGE_WINDOW_END = "16:00"

EOD_TZ = "America/New_York"
EOD_CUTOFF_TIME = dt_time(15, 55)  # a few minutes ahead of the 16:00 ET close, as a safety margin

DEFAULT_TAKE_PROFIT_R_MULTIPLE = 1.8
DEFAULT_TREND_FILTER_PERIOD = 120  # minutes == bars, since this strategy is 1-minute only
DEFAULT_TREND_FILTER_MODE = "sma"
VALID_TREND_FILTER_MODES = ("sma", "ema")

# Not a "warm-up" requirement like trend_following's EMA - this strategy only ever
# looks at *today's* bars. It needs to be large enough that bot/backtest.py's bounded
# replay window (min_bars + 5, mirroring main.py's get_bars(limit=...)) still spans
# the *entire* trading day (~391 one-minute bars) even late in the session, so the
# 15:29-16:00 range never falls outside the window.
MIN_BARS = 800


@dataclass
class _DayState:
    range_high: float
    range_low: float
    breakout_ts: Optional[pd.Timestamp] = None
    breakout_direction: Optional[str] = None  # "long" | "short"
    breakout_high: Optional[float] = None
    breakout_low: Optional[float] = None
    retest_ts: Optional[pd.Timestamp] = None
    entry_ts: Optional[pd.Timestamp] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None


def _compute_day_state(today_bars: pd.DataFrame, take_profit_r_multiple: float) -> Optional[_DayState]:
    """today_bars: bars already filtered to a single local trading day, tz-aware,
    sorted ascending. Returns None if the 15:29-16:00 range isn't established yet."""
    range_window = today_bars.between_time(RANGE_WINDOW_START, RANGE_WINDOW_END)
    if range_window.empty:
        return None

    range_high = float(range_window["high"].max())
    range_low = float(range_window["low"].min())
    state = _DayState(range_high=range_high, range_low=range_low)

    after_range = today_bars[today_bars.index > range_window.index.max()]
    if after_range.empty:
        return state

    long_breaks = after_range[after_range["close"] > range_high]
    short_breaks = after_range[after_range["close"] < range_low]
    first_long_ts = long_breaks.index.min() if not long_breaks.empty else None
    first_short_ts = short_breaks.index.min() if not short_breaks.empty else None

    if first_long_ts is None and first_short_ts is None:
        return state  # no breakout yet today

    if first_short_ts is None or (first_long_ts is not None and first_long_ts < first_short_ts):
        breakout_ts, direction = first_long_ts, "long"
    else:
        breakout_ts, direction = first_short_ts, "short"

    breakout_bar = after_range.loc[breakout_ts]
    state.breakout_ts = breakout_ts
    state.breakout_direction = direction
    state.breakout_high = float(breakout_bar["high"])
    state.breakout_low = float(breakout_bar["low"])

    after_breakout = after_range[after_range.index > breakout_ts]
    if after_breakout.empty:
        return state

    if direction == "long":
        retest_mask = (after_breakout["low"] <= state.breakout_high) & (after_breakout["close"] >= range_high)
    else:
        retest_mask = (after_breakout["high"] >= state.breakout_low) & (after_breakout["close"] <= range_low)

    retest_bars = after_breakout[retest_mask]
    if retest_bars.empty:
        return state
    retest_ts = retest_bars.index.min()
    state.retest_ts = retest_ts

    # Entry is only looked for strictly after the retest bar (rule 5: "nach
    # angelaufenem Retest") - a bar can't simultaneously sit inside the retest zone
    # and close beyond the breakout bar's high/low, so this never overlaps anyway.
    after_retest = after_breakout[after_breakout.index > retest_ts]
    if after_retest.empty:
        return state

    if direction == "long":
        entry_bars = after_retest[after_retest["close"] > state.breakout_high]
    else:
        entry_bars = after_retest[after_retest["close"] < state.breakout_low]

    if entry_bars.empty:
        return state
    entry_ts = entry_bars.index.min()
    state.entry_ts = entry_ts

    entry_price = float(entry_bars.loc[entry_ts, "close"])
    if direction == "long":
        stop_price = range_low
        risk_amount = entry_price - stop_price
        target_price = entry_price + take_profit_r_multiple * risk_amount
    else:
        stop_price = range_high
        risk_amount = stop_price - entry_price
        target_price = entry_price - take_profit_r_multiple * risk_amount

    if risk_amount > 0:
        state.stop_price = stop_price
        state.target_price = target_price

    return state


class OpeningRangeBreakoutStrategy(Strategy):
    self_managed_exits = True  # see module docstring: fixed range-based stop/target, not ATR

    def __init__(
        self,
        symbol: str,
        take_profit_r_multiple: float = DEFAULT_TAKE_PROFIT_R_MULTIPLE,
        trend_filter_enabled: bool = False,
        trend_filter_period: int = DEFAULT_TREND_FILTER_PERIOD,
        trend_filter_mode: str = DEFAULT_TREND_FILTER_MODE,
    ):
        if trend_filter_mode not in VALID_TREND_FILTER_MODES:
            raise ValueError(f"trend_filter_mode must be one of {VALID_TREND_FILTER_MODES}, got {trend_filter_mode!r}")
        super().__init__(
            symbol,
            take_profit_r_multiple=take_profit_r_multiple,
            trend_filter_enabled=trend_filter_enabled,
            trend_filter_period=trend_filter_period,
            trend_filter_mode=trend_filter_mode,
        )
        self.take_profit_r_multiple = take_profit_r_multiple
        self.trend_filter_enabled = trend_filter_enabled
        self.trend_filter_period = trend_filter_period
        self.trend_filter_mode = trend_filter_mode

    @property
    def min_bars(self) -> int:
        # Comfortably covers a full trading day (see MIN_BARS above); also widened
        # for an unusually large trend_filter_period so its lookback is never
        # truncated out of the replay window bot/backtest.py hands us.
        return max(MIN_BARS, self.trend_filter_period + 5)

    def _to_local(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars.index.tz is None:
            bars = bars.tz_localize("UTC")
        return bars.tz_convert(RANGE_TZ)

    def _today_bars(self, bars_local: pd.DataFrame) -> pd.DataFrame:
        today = bars_local.index[-1].date()
        return bars_local[bars_local.index.date == today]

    def _is_end_of_day(self, bars: pd.DataFrame) -> bool:
        if bars.index.tz is None:
            bars = bars.tz_localize("UTC")
        last_local = bars.tz_convert(EOD_TZ).index[-1]
        return last_local.time() >= EOD_CUTOFF_TIME

    def _passes_trend_filter(self, bars_local: pd.DataFrame, entry_ts: pd.Timestamp, direction: str) -> bool:
        """Rule 9. bars_local: the full (possibly multi-day) bars already converted
        to RANGE_TZ, so its index is directly comparable to entry_ts. Deliberately
        allowed to look across a day boundary into prior-session bars when
        trend_filter_period is larger than how much of today has traded so far -
        "the last N *traded* minutes", not "the last N minutes of today only"."""
        if not self.trend_filter_enabled:
            return True

        window = bars_local[bars_local.index <= entry_ts]
        if len(window) < self.trend_filter_period:
            return False  # not enough history to compute the filter yet - block conservatively

        closes = window["close"]
        if self.trend_filter_mode == "ema":
            ma_series = indicators.ema(closes, self.trend_filter_period)
        else:
            ma_series = indicators.sma(closes, self.trend_filter_period)

        ma_value = float(ma_series.iloc[-1])
        if pd.isna(ma_value):
            return False

        price = float(closes.iloc[-1])
        return price > ma_value if direction == "long" else price < ma_value

    def get_entry_stop_price(self, bars: pd.DataFrame, side: str) -> Optional[float]:
        """Used by bot/backtest.py (only for self_managed_exits strategies) to size
        the position using this strategy's own stop distance instead of an ATR one.
        Expected to be called with the same `bars` right after generate_signal
        returned LONG/SHORT for them."""
        bars_local = self._to_local(bars)
        today_bars = self._today_bars(bars_local)
        state = _compute_day_state(today_bars, self.take_profit_r_multiple)
        if state is None:
            return None
        return state.stop_price

    def generate_signal(self, bars: pd.DataFrame, position_side: Optional[str]) -> Signal:
        if bars.empty:
            return Signal.HOLD

        bars_local = self._to_local(bars)
        today_bars = self._today_bars(bars_local)
        state = _compute_day_state(today_bars, self.take_profit_r_multiple)
        current_ts = today_bars.index[-1] if not today_bars.empty else None
        current_close = float(bars["close"].iloc[-1])

        if position_side is not None:
            if state is not None and state.stop_price is not None:
                if position_side == "long":
                    if current_close <= state.stop_price:
                        return Signal.EXIT
                    if state.target_price is not None and current_close >= state.target_price:
                        return Signal.EXIT
                else:
                    if current_close >= state.stop_price:
                        return Signal.EXIT
                    if state.target_price is not None and current_close <= state.target_price:
                        return Signal.EXIT

            if self._is_end_of_day(bars):
                return Signal.EXIT

            return Signal.HOLD

        # Flat: only signal on the exact bar where today's entry condition first
        # fires - on every later bar that same entry_ts is in the past, so this
        # naturally enforces "at most one trade per day" with no extra state. A
        # trend-filter rejection reuses the same mechanism: entry_ts doesn't change
        # on a later call, so once rejected it's rejected for the rest of the day.
        if state is None or state.entry_ts is None or state.entry_ts != current_ts:
            return Signal.HOLD

        direction = state.breakout_direction
        if not self._passes_trend_filter(bars_local, state.entry_ts, direction):
            return Signal.HOLD

        return Signal.LONG if direction == "long" else Signal.SHORT
