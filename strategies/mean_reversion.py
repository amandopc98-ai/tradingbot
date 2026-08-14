from typing import Optional

import pandas as pd

from strategies.base import Signal, Strategy


class MeanReversionStrategy(Strategy):
    """Bollinger-style z-score mean reversion.

    Flat -> LONG when price is more than `z_threshold` standard deviations below the
    rolling mean; flat -> SHORT when more than `z_threshold` above. Exits when price
    returns to the mean.
    """

    def __init__(self, symbol: str, period: int = 20, z_threshold: float = 1.5):
        super().__init__(symbol, period=period, z_threshold=z_threshold)
        self.period = period
        self.z_threshold = z_threshold

    @property
    def min_bars(self) -> int:
        return self.period + 1

    def generate_signal(self, bars: pd.DataFrame, position_side: Optional[str]) -> Signal:
        closes = bars["close"]
        if len(closes) < self.period:
            return Signal.HOLD

        mean = closes.rolling(self.period).mean().iloc[-1]
        std = closes.rolling(self.period).std().iloc[-1]
        last = closes.iloc[-1]

        if pd.isna(std) or std == 0:
            return Signal.HOLD

        z = (last - mean) / std

        if position_side is None:
            if z <= -self.z_threshold:
                return Signal.LONG
            if z >= self.z_threshold:
                return Signal.SHORT
            return Signal.HOLD

        if position_side == "long":
            return Signal.EXIT if last >= mean else Signal.HOLD

        if position_side == "short":
            return Signal.EXIT if last <= mean else Signal.HOLD

        return Signal.HOLD
