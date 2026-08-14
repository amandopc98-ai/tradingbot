from strategies.base import Signal, Strategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.trend_following import TrendFollowingStrategy

STRATEGY_REGISTRY = {
    "mean_reversion": MeanReversionStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
    "trend_following": TrendFollowingStrategy,
}

__all__ = [
    "Signal",
    "Strategy",
    "MeanReversionStrategy",
    "MomentumBreakoutStrategy",
    "TrendFollowingStrategy",
    "STRATEGY_REGISTRY",
]
