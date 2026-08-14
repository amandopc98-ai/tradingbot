import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv


@dataclass
class SymbolConfig:
    strategy: str
    timeframe: TimeFrame
    asset_class: str  # "stock" or "crypto"
    params: dict
    trail_atr_mult: Optional[float]  # None = no trailing stop, just the hard ATR stop
    max_allocation_pct: float = 0.30


@dataclass
class Config:
    api_key: str
    secret_key: str
    paper: bool
    poll_interval_seconds: int
    risk_pct: float
    allow_shorting: bool
    state_file: str
    symbols: Dict[str, SymbolConfig] = field(default_factory=dict)


def _default_symbols() -> Dict[str, SymbolConfig]:
    return {
        "SPY": SymbolConfig(
            strategy="mean_reversion",
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            asset_class="stock",
            params={"period": 20, "z_threshold": 1.5},
            trail_atr_mult=None,
        ),
        "QQQ": SymbolConfig(
            strategy="mean_reversion",
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            asset_class="stock",
            params={"period": 20, "z_threshold": 1.8},
            trail_atr_mult=None,
        ),
        "BTC/USD": SymbolConfig(
            strategy="momentum_breakout",
            timeframe=TimeFrame.Hour,
            asset_class="crypto",
            params={"period": 20, "volume_mult": 1.5},
            trail_atr_mult=2.0,
        ),
        "GLD": SymbolConfig(
            strategy="trend_following",
            timeframe=TimeFrame(4, TimeFrameUnit.Hour),
            asset_class="stock",
            params={"fast": 50, "slow": 200},
            trail_atr_mult=3.0,
        ),
        "USO": SymbolConfig(
            strategy="trend_following",
            timeframe=TimeFrame(4, TimeFrameUnit.Hour),
            asset_class="stock",
            params={"fast": 50, "slow": 200},
            trail_atr_mult=3.0,
        ),
    }


def load_config() -> Config:
    load_dotenv()

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set (see .env.example)."
        )

    return Config(
        api_key=api_key,
        secret_key=secret_key,
        paper=os.environ.get("ALPACA_PAPER", "true").lower() != "false",
        poll_interval_seconds=int(os.environ.get("POLL_INTERVAL_SECONDS", "60")),
        risk_pct=float(os.environ.get("RISK_PCT_PER_TRADE", "0.01")),
        allow_shorting=os.environ.get("ALLOW_SHORTING", "true").lower() == "true",
        state_file=os.environ.get("PORTFOLIO_STATE_FILE", "portfolio_state.json"),
        symbols=_default_symbols(),
    )
