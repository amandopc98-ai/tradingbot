from bot.strategies.base import Signal, Strategy
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.momentum_breakout import MomentumBreakoutStrategy
from bot.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from bot.strategies.trend_following import TrendFollowingStrategy

STRATEGY_REGISTRY = {
    "mean_reversion": MeanReversionStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
    "trend_following": TrendFollowingStrategy,
    "opening_range_breakout": OpeningRangeBreakoutStrategy,
}

__all__ = [
    "Signal",
    "Strategy",
    "MeanReversionStrategy",
    "MomentumBreakoutStrategy",
    "TrendFollowingStrategy",
    "OpeningRangeBreakoutStrategy",
    "STRATEGY_REGISTRY",
]
