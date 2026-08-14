"""Position sizing, stop-loss, and cross-symbol exposure rules.

Sizing rule: every symbol is sized so that a 1x ATR adverse move equals exactly
`risk_pct` of account equity. A quiet instrument therefore gets a larger position
and a volatile one gets a smaller one, holding dollar risk constant across symbols.

Stop rule: the initial hard stop sits exactly 1x ATR from entry, which by
construction is a `risk_pct` loss of equity - no exceptions, regardless of strategy.
A strategy-specific trailing multiple (e.g. 2x ATR for momentum, 3x ATR for trend
following) only ever tightens the stop as a trade moves favorably; it never widens
risk beyond the initial hard stop.
"""
import math
from typing import Optional

from bot.portfolio import Portfolio, Position

CRYPTO_ASSET_CLASS = "crypto"


def position_size(equity: float, atr: float, price: float, risk_pct: float, asset_class: str) -> float:
    if atr <= 0 or price <= 0 or equity <= 0:
        return 0.0
    risk_amount = equity * risk_pct
    qty = risk_amount / atr
    if asset_class != CRYPTO_ASSET_CLASS:
        qty = math.floor(qty)
    else:
        qty = round(qty, 6)
    return max(qty, 0.0)


def cap_by_buying_power(qty: float, buying_power: float, price: float, max_allocation_pct: float, asset_class: str) -> float:
    if qty <= 0 or price <= 0:
        return 0.0
    max_notional = buying_power * max_allocation_pct
    notional = qty * price
    if notional > max_notional and max_notional > 0:
        qty = max_notional / price
        if asset_class != CRYPTO_ASSET_CLASS:
            qty = math.floor(qty)
        else:
            qty = round(qty, 6)
    return max(qty, 0.0)


def initial_stop_price(entry_price: float, atr: float, side: str) -> float:
    """1x ATR hard stop - by construction (see position_size) this equals a
    risk_pct loss of equity if hit."""
    return entry_price - atr if side == "long" else entry_price + atr


def update_trailing_stop(position: Position, current_price: float, atr: float, trail_atr_mult: Optional[float]) -> float:
    """Ratchets position.stop_price toward the current price using a trailing
    multiple of ATR, but only ever in the favorable direction. Returns the
    (possibly updated) stop price. If trail_atr_mult is None, the strategy has no
    trailing-stop component and the hard stop is left untouched.
    """
    if position.side == "long":
        position.extreme_price = max(position.extreme_price, current_price)
        if trail_atr_mult is not None and atr > 0:
            candidate = position.extreme_price - trail_atr_mult * atr
            position.stop_price = max(position.stop_price, candidate)
    else:
        position.extreme_price = min(position.extreme_price, current_price)
        if trail_atr_mult is not None and atr > 0:
            candidate = position.extreme_price + trail_atr_mult * atr
            position.stop_price = min(position.stop_price, candidate)
    return position.stop_price


def stop_triggered(position: Position, current_price: float) -> bool:
    if position.side == "long":
        return current_price <= position.stop_price
    return current_price >= position.stop_price


def correlation_blocked(symbol: str, side: str, portfolio: Portfolio) -> bool:
    """If SPY and QQQ are both already long, block new BTC/USD long entries to
    avoid doubling up on risk-on exposure."""
    if symbol == "BTC/USD" and side == "long":
        return portfolio.is_long("SPY") and portfolio.is_long("QQQ")
    return False
