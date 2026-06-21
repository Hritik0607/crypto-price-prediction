"""
tests/test_candle.py

Unit tests for candle aggregation logic in services/candles/run.py

Tests the two pure functions:
  - init_candle(): initializes a candle from the first trade
  - update_candle(): updates candle with each subsequent trade

These functions have no external dependencies (no Kafka, no Quix)
so they are easy to test in isolation.
"""

import pytest
from run import init_candle, update_candle

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def first_trade():
    return {
        'pair': 'BTC/USD',
        'price': 65000.0,
        'volume': 0.5,
        'timestamp_ms': 1_704_900_600_000,
    }


@pytest.fixture
def second_trade():
    return {
        'pair': 'BTC/USD',
        'price': 65100.0,
        'volume': 0.3,
        'timestamp_ms': 1_704_900_630_000,
    }


@pytest.fixture
def third_trade():
    return {
        'pair': 'BTC/USD',
        'price': 64900.0,
        'volume': 0.2,
        'timestamp_ms': 1_704_900_660_000,
    }


# ── init_candle tests ──────────────────────────────────────────────────────────


class TestInitCandle:
    """Tests for init_candle() — initializes candle from first trade."""

    def test_open_equals_first_trade_price(self, first_trade):
        candle = init_candle(first_trade)
        assert candle['open'] == 65000.0

    def test_high_equals_first_trade_price(self, first_trade):
        candle = init_candle(first_trade)
        assert candle['high'] == 65000.0

    def test_low_equals_first_trade_price(self, first_trade):
        candle = init_candle(first_trade)
        assert candle['low'] == 65000.0

    def test_close_equals_first_trade_price(self, first_trade):
        candle = init_candle(first_trade)
        assert candle['close'] == 65000.0

    def test_volume_equals_first_trade_volume(self, first_trade):
        candle = init_candle(first_trade)
        assert candle['volume'] == 0.5

    def test_pair_is_set(self, first_trade):
        candle = init_candle(first_trade)
        assert candle['pair'] == 'BTC/USD'

    def test_timestamp_ms_is_set(self, first_trade):
        candle = init_candle(first_trade)
        assert candle['timestamp_ms'] == 1_704_900_600_000

    def test_ohlc_all_equal_on_init(self, first_trade):
        """
        On initialization all of open/high/low/close must equal
        the first trade price — no other trades have arrived yet.
        """
        candle = init_candle(first_trade)
        assert candle['open'] == candle['high'] == candle['low'] == candle['close']


# ── update_candle tests ────────────────────────────────────────────────────────


class TestUpdateCandle:
    """Tests for update_candle() — updates candle with each subsequent trade."""

    def test_close_updates_to_latest_price(self, first_trade, second_trade):
        """
        close must always reflect the most recent trade price.
        """
        candle = init_candle(first_trade)
        candle = update_candle(candle, second_trade)
        assert candle['close'] == 65100.0

    def test_open_never_changes(self, first_trade, second_trade, third_trade):
        """
        open must remain the price of the very first trade
        regardless of subsequent trades.
        """
        candle = init_candle(first_trade)
        candle = update_candle(candle, second_trade)
        candle = update_candle(candle, third_trade)
        assert candle['open'] == 65000.0

    def test_high_tracks_maximum_price(self, first_trade, second_trade, third_trade):
        """
        high must be the maximum price across all trades in the candle.
        65000 → 65100 → 64900: high should be 65100
        """
        candle = init_candle(first_trade)
        candle = update_candle(candle, second_trade)
        candle = update_candle(candle, third_trade)
        assert candle['high'] == 65100.0

    def test_low_tracks_minimum_price(self, first_trade, second_trade, third_trade):
        """
        low must be the minimum price across all trades in the candle.
        65000 → 65100 → 64900: low should be 64900
        """
        candle = init_candle(first_trade)
        candle = update_candle(candle, second_trade)
        candle = update_candle(candle, third_trade)
        assert candle['low'] == 64900.0

    def test_volume_accumulates(self, first_trade, second_trade, third_trade):
        """
        volume must be the sum of all trade volumes.
        0.5 + 0.3 + 0.2 = 1.0
        """
        candle = init_candle(first_trade)
        candle = update_candle(candle, second_trade)
        candle = update_candle(candle, third_trade)
        assert abs(candle['volume'] - 1.0) < 1e-9

    def test_timestamp_ms_updates_to_latest(self, first_trade, second_trade):
        """
        timestamp_ms must reflect the most recent trade timestamp.
        """
        candle = init_candle(first_trade)
        candle = update_candle(candle, second_trade)
        assert candle['timestamp_ms'] == 1_704_900_630_000

    def test_high_does_not_decrease(self, first_trade, third_trade):
        """
        If a lower-priced trade arrives, high must not change.
        first_trade=65000, third_trade=64900: high stays at 65000
        """
        candle = init_candle(first_trade)
        candle = update_candle(candle, third_trade)
        assert candle['high'] == 65000.0

    def test_low_does_not_increase(self, first_trade, second_trade):
        """
        If a higher-priced trade arrives, low must not change.
        first_trade=65000, second_trade=65100: low stays at 65000
        """
        candle = init_candle(first_trade)
        candle = update_candle(candle, second_trade)
        assert candle['low'] == 65000.0


# ── Integration: full candle sequence ─────────────────────────────────────────


class TestFullCandleSequence:
    """
    Tests a complete candle built from multiple trades
    to verify all fields are correct together.
    """

    def test_full_candle_ohlcv(self):
        """
        Build a complete candle from 5 trades and verify all OHLCV values.

        Trades:
          1. BTC/USD @ $65,000 vol=1.0  ← open
          2. BTC/USD @ $65,500 vol=0.5  ← high candidate
          3. BTC/USD @ $64,800 vol=0.3  ← low candidate
          4. BTC/USD @ $65,200 vol=0.8
          5. BTC/USD @ $65,100 vol=0.4  ← close

        Expected:
          open  = 65,000  (first trade)
          high  = 65,500  (max of all)
          low   = 64,800  (min of all)
          close = 65,100  (last trade)
          volume = 3.0    (sum of all)
        """
        trades = [
            {'pair': 'BTC/USD', 'price': 65000.0, 'volume': 1.0, 'timestamp_ms': 1000},
            {'pair': 'BTC/USD', 'price': 65500.0, 'volume': 0.5, 'timestamp_ms': 2000},
            {'pair': 'BTC/USD', 'price': 64800.0, 'volume': 0.3, 'timestamp_ms': 3000},
            {'pair': 'BTC/USD', 'price': 65200.0, 'volume': 0.8, 'timestamp_ms': 4000},
            {'pair': 'BTC/USD', 'price': 65100.0, 'volume': 0.4, 'timestamp_ms': 5000},
        ]

        candle = init_candle(trades[0])
        for trade in trades[1:]:
            candle = update_candle(candle, trade)

        assert candle['open'] == 65000.0
        assert candle['high'] == 65500.0
        assert candle['low'] == 64800.0
        assert candle['close'] == 65100.0
        assert abs(candle['volume'] - 3.0) < 1e-9
        assert candle['timestamp_ms'] == 5000
        assert candle['pair'] == 'BTC/USD'
