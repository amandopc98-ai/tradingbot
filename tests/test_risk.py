import risk
from portfolio import Portfolio


def test_position_size_scales_inversely_with_atr():
    equity = 100_000.0
    quiet = risk.position_size(equity, atr=1.0, price=100.0, risk_pct=0.01, asset_class="stock")
    volatile = risk.position_size(equity, atr=10.0, price=100.0, risk_pct=0.01, asset_class="stock")
    assert quiet > volatile
    # 1% of 100k = 1000 risk; ATR=1 -> 1000 shares; ATR=10 -> 100 shares
    assert quiet == 1000
    assert volatile == 100


def test_position_size_rounds_stocks_down_to_whole_shares():
    qty = risk.position_size(equity=10_000.0, atr=3.0, price=50.0, risk_pct=0.01, asset_class="stock")
    assert qty == 33  # 100 / 3 = 33.33 -> floor to 33
    assert float(qty).is_integer()


def test_position_size_allows_fractional_crypto():
    qty = risk.position_size(equity=10_000.0, atr=3.0, price=50.0, risk_pct=0.01, asset_class="crypto")
    assert round(qty, 2) == 33.33


def test_position_size_zero_when_atr_is_zero():
    assert risk.position_size(10_000.0, atr=0.0, price=50.0, risk_pct=0.01, asset_class="stock") == 0.0


def test_cap_by_buying_power_reduces_oversized_qty():
    # 1000 shares @ $100 = $100k notional, but only $10k buying power at 30% max allocation = $3k cap
    qty = risk.cap_by_buying_power(qty=1000, buying_power=10_000.0, price=100.0, max_allocation_pct=0.30, asset_class="stock")
    assert qty == 30  # $3000 / $100 = 30 shares


def test_cap_by_buying_power_leaves_small_orders_untouched():
    qty = risk.cap_by_buying_power(qty=5, buying_power=10_000.0, price=100.0, max_allocation_pct=0.30, asset_class="stock")
    assert qty == 5


def test_initial_stop_price_long_and_short():
    assert risk.initial_stop_price(entry_price=100.0, atr=5.0, side="long") == 95.0
    assert risk.initial_stop_price(entry_price=100.0, atr=5.0, side="short") == 105.0


def test_trailing_stop_only_tightens_never_loosens_for_long():
    from portfolio import Position

    pos = Position(
        symbol="BTC/USD", side="long", qty=1.0, entry_price=100.0, entry_atr=5.0,
        stop_price=95.0, extreme_price=100.0, opened_at="2026-01-01T00:00:00+00:00",
    )
    # price rises: trailing stop (2x ATR) should ratchet up
    risk.update_trailing_stop(pos, current_price=120.0, atr=5.0, trail_atr_mult=2.0)
    assert pos.stop_price == 110.0  # 120 - 2*5

    # price dips but stays above the trailing stop: stop must never move down
    risk.update_trailing_stop(pos, current_price=112.0, atr=5.0, trail_atr_mult=2.0)
    assert pos.stop_price == 110.0


def test_trailing_stop_none_leaves_hard_stop_untouched():
    from portfolio import Position

    pos = Position(
        symbol="SPY", side="long", qty=1.0, entry_price=100.0, entry_atr=5.0,
        stop_price=95.0, extreme_price=100.0, opened_at="2026-01-01T00:00:00+00:00",
    )
    risk.update_trailing_stop(pos, current_price=130.0, atr=5.0, trail_atr_mult=None)
    assert pos.stop_price == 95.0  # mean-reversion has no trailing component


def test_stop_triggered_long_and_short():
    from portfolio import Position

    long_pos = Position("SPY", "long", 1.0, 100.0, 5.0, 95.0, 100.0, "2026-01-01T00:00:00+00:00")
    assert risk.stop_triggered(long_pos, current_price=94.0) is True
    assert risk.stop_triggered(long_pos, current_price=96.0) is False

    short_pos = Position("SPY", "short", 1.0, 100.0, 5.0, 105.0, 100.0, "2026-01-01T00:00:00+00:00")
    assert risk.stop_triggered(short_pos, current_price=106.0) is True
    assert risk.stop_triggered(short_pos, current_price=104.0) is False


def test_correlation_filter_blocks_btc_long_when_spy_and_qqq_both_long(tmp_path):
    portfolio = Portfolio(state_path=str(tmp_path / "state.json"))
    portfolio.open("SPY", "long", 10, 500.0, 5.0, 495.0)
    portfolio.open("QQQ", "long", 10, 400.0, 4.0, 396.0)

    assert risk.correlation_blocked("BTC/USD", "long", portfolio) is True


def test_correlation_filter_allows_btc_when_only_one_of_spy_qqq_long(tmp_path):
    portfolio = Portfolio(state_path=str(tmp_path / "state.json"))
    portfolio.open("SPY", "long", 10, 500.0, 5.0, 495.0)

    assert risk.correlation_blocked("BTC/USD", "long", portfolio) is False


def test_correlation_filter_does_not_apply_to_other_symbols(tmp_path):
    portfolio = Portfolio(state_path=str(tmp_path / "state.json"))
    portfolio.open("SPY", "long", 10, 500.0, 5.0, 495.0)
    portfolio.open("QQQ", "long", 10, 400.0, 4.0, 396.0)

    assert risk.correlation_blocked("GLD", "long", portfolio) is False
