from typing import Optional

import pandas as pd

from strategies.base import Signal, Strategy


class TrendFollowingStrategy(Strategy):
    """EMA(fast)/EMA(slow) crossover trend following.

    Flat -> LONG when the fast EMA crosses above the slow EMA. Flat -> SHORT (or exit
    a long) on the opposite cross. The trailing ATR stop is enforced separately by
    risk.py so it can react between bar closes.
    """

    def __init__(self, symbol: str, fast: int = 50, slow: int = 200):
        super().__init__(symbol, fast=fast, slow=slow)
        self.fast = fast
        self.slow = slow

    @property
    def min_bars(self) -> int:
        return self.slow + 2

    def generate_signal(self, bars: pd.DataFrame, position_side: Optional[str]) -> Signal:
        closes = bars["close"]
        if len(closes) < self.slow + 2:
            return Signal.HOLD

        ema_fast = closes.ewm(span=self.fast, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow, adjust=False).mean()

        fast_now, fast_prev = ema_fast.iloc[-1], ema_fast.iloc[-2]
        slow_now, slow_prev = ema_slow.iloc[-1], ema_slow.iloc[-2]

        cross_up = fast_prev <= slow_prev and fast_now > slow_now
        cross_down = fast_prev >= slow_prev and fast_now < slow_now

        if position_side is None:
            if cross_up:
                return Signal.LONG
            if cross_down:
                return Signal.SHORT
            return Signal.HOLD

        if position_side == "long":
            return Signal.EXIT if cross_down else Signal.HOLD

        if position_side == "short":
            return Signal.EXIT if cross_up else Signal.HOLD

        return Signal.HOLD
