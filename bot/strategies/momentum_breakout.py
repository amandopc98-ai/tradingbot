from typing import Optional

import pandas as pd

from bot.strategies.base import Signal, Strategy


class MomentumBreakoutStrategy(Strategy):
    """20-period high/low breakout confirmed by volume.

    Flat -> LONG when the close breaks above the prior `period`-bar high with volume
    at least `volume_mult`x the prior `period`-bar average volume. Flat -> SHORT on
    the mirror-image breakdown (callers may disable short entries, e.g. for spot
    crypto). Exits (or reverses) on a breakdown/breakout back through the opposite
    extreme; the trailing ATR stop itself is enforced separately by risk.py since it
    needs to run every tick, not just on new bars.
    """

    def __init__(self, symbol: str, period: int = 20, volume_mult: float = 1.5):
        super().__init__(symbol, period=period, volume_mult=volume_mult)
        self.period = period
        self.volume_mult = volume_mult

    @property
    def min_bars(self) -> int:
        return self.period + 2

    def generate_signal(self, bars: pd.DataFrame, position_side: Optional[str]) -> Signal:
        if len(bars) < self.period + 2:
            return Signal.HOLD

        highs, lows, vols, closes = bars["high"], bars["low"], bars["volume"], bars["close"]

        # Exclude the current (just-closed) bar from the reference window so the
        # breakout is measured against the prior `period` bars, not itself.
        prior_high = highs.iloc[-self.period - 1:-1].max()
        prior_low = lows.iloc[-self.period - 1:-1].min()
        avg_vol = vols.iloc[-self.period - 1:-1].mean()

        last_close = closes.iloc[-1]
        last_vol = vols.iloc[-1]
        volume_confirmed = avg_vol > 0 and last_vol >= self.volume_mult * avg_vol

        if position_side is None:
            if last_close > prior_high and volume_confirmed:
                return Signal.LONG
            if last_close < prior_low and volume_confirmed:
                return Signal.SHORT
            return Signal.HOLD

        if position_side == "long":
            return Signal.EXIT if last_close < prior_low else Signal.HOLD

        if position_side == "short":
            return Signal.EXIT if last_close > prior_high else Signal.HOLD

        return Signal.HOLD
