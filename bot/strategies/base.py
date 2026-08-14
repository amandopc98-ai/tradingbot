from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import pandas as pd


class Signal(str, Enum):
    LONG = "long"
    SHORT = "short"
    EXIT = "exit"
    HOLD = "hold"


class Strategy(ABC):
    """Generates entry/exit signals from a closed-bar OHLCV DataFrame.

    Strategies never place orders or size positions themselves - they only look at
    price history plus the current position side (or None if flat) and return a
    Signal. Sizing, stops, and order execution are handled by risk.py / main.py so
    the same risk model applies uniformly across strategies.
    """

    def __init__(self, symbol: str, **params):
        self.symbol = symbol
        self.params = params

    @property
    @abstractmethod
    def min_bars(self) -> int:
        """Minimum number of closed bars required before a signal can be generated."""

    @abstractmethod
    def generate_signal(self, bars: pd.DataFrame, position_side: Optional[str]) -> Signal:
        """bars: DataFrame with open/high/low/close/volume columns, oldest first.
        position_side: "long", "short", or None if currently flat.
        """
