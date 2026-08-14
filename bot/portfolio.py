"""Local tracking of open positions: side, entry, stop, and trailing-stop state.

Alpaca is the source of truth for whether a position exists and its quantity, but it
has no concept of "our" trailing stop or entry ATR, so we persist that supplementary
state to disk between ticks (and process restarts) here.
"""
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    qty: float
    entry_price: float
    entry_atr: float
    stop_price: float
    extreme_price: float  # highest price since entry (long) / lowest (short), for trailing stops
    opened_at: str  # ISO timestamp


class Portfolio:
    def __init__(self, state_path: str = "portfolio_state.json"):
        self.state_path = state_path
        self.positions: Dict[str, Position] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r") as f:
                raw = json.load(f)
            self.positions = {sym: Position(**data) for sym, data in raw.items()}
        except (json.JSONDecodeError, TypeError, OSError):
            logger.exception("Failed to load portfolio state from %s; starting empty", self.state_path)
            self.positions = {}

    def save(self) -> None:
        with open(self.state_path, "w") as f:
            json.dump({sym: asdict(pos) for sym, pos in self.positions.items()}, f, indent=2)

    def get(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def is_long(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos is not None and pos.side == "long"

    def is_short(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos is not None and pos.side == "short"

    def open(self, symbol: str, side: str, qty: float, entry_price: float, entry_atr: float, stop_price: float) -> Position:
        pos = Position(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            entry_atr=entry_atr,
            stop_price=stop_price,
            extreme_price=entry_price,
            opened_at=datetime.now(timezone.utc).isoformat(),
        )
        self.positions[symbol] = pos
        return pos

    def remove(self, symbol: str) -> None:
        self.positions.pop(symbol, None)

    def reconcile_with_broker(self, broker_positions: Dict[str, dict]) -> None:
        """Drop local positions the broker no longer shows (e.g. closed manually or
        stopped out on the broker side), and warn about broker positions we have no
        local stop/trailing state for (best-effort: they'll get a fresh hard stop on
        the next tick rather than losing risk management entirely).
        """
        for symbol in list(self.positions):
            if symbol not in broker_positions:
                logger.warning("Dropping local position for %s: no longer open at broker", symbol)
                self.remove(symbol)

        for symbol in broker_positions:
            if symbol not in self.positions:
                logger.warning(
                    "Broker has an open position in %s with no local stop-loss state "
                    "(bot restarted mid-trade?). It will be re-initialized with a fresh "
                    "ATR-based hard stop on the next tick.",
                    symbol,
                )
