"""
tests/test_trade.py

Unit tests for the Trade class in services/trades/kraken_api/trade.py

Critical bug fixed:
    datetime.fromtimestamp() without tz=timezone.utc uses LOCAL time.
    On IST (UTC+5:30), all timestamps were offset by +5.5 hours.
    Fix: always pass tz=timezone.utc explicitly.

These tests verify the UTC fix is in place and timestamps are correct.
"""

from kraken_api.trade import Trade


class TestTimestampUTCFix:
    """
    Tests that verify timestamps are always in UTC regardless of
    the local machine timezone. This was the critical bug that caused
    5.5 hour offset on IST machines.
    """

    def test_milliseconds2datestr_returns_utc(self):
        """
        _milliseconds2datestr must return UTC time string.

        Unix timestamp 0 = 1970-01-01T00:00:00.000000Z in UTC.
        Without tz=timezone.utc, on IST this would return
        1970-01-01T05:30:00 — wrong by 5.5 hours.
        """
        result = Trade._milliseconds2datestr(0)
        assert result == '1970-01-01T00:00:00.000000Z', (
            f'Expected UTC time 1970-01-01T00:00:00.000000Z, got {result}. '
            f'Timezone bug: datetime.fromtimestamp() is using local time instead of UTC.'
        )

    def test_milliseconds2datestr_known_timestamp(self):
        """
        Known timestamp: 1000000000000 ms = 2001-09-09T01:46:40.000000Z UTC
        Verifies conversion is correct for a real-world value.
        """
        result = Trade._milliseconds2datestr(1_000_000_000_000)
        assert (
            result == '2001-09-09T01:46:40.000000Z'
        ), f'Timestamp conversion wrong: got {result}'

    def test_datestr2milliseconds_parses_utc(self):
        """
        _datestr2milliseconds must parse the date string as UTC.

        '1970-01-01T00:00:00.000000Z' should return 0 ms (Unix epoch).
        Without .replace(tzinfo=timezone.utc), on IST this would
        return -19800000 (negative — 5.5 hours before epoch).
        """
        result = Trade._datestr2milliseconds('1970-01-01T00:00:00.000000Z')
        assert result == 0, (
            f'Expected 0 ms (Unix epoch), got {result}. '
            f'Timezone bug: datestr is being parsed as local time instead of UTC.'
        )

    def test_roundtrip_milliseconds_to_datestr_and_back(self):
        """
        Roundtrip: ms → datestr → ms must return the same value.
        Tests that _milliseconds2datestr and _datestr2milliseconds
        are consistent with each other.
        """
        original_ms = 1_704_900_600_000  # 2024-01-10T15:30:00Z

        datestr = Trade._milliseconds2datestr(original_ms)
        recovered_ms = Trade._datestr2milliseconds(datestr)

        assert (
            recovered_ms == original_ms
        ), f'Roundtrip failed: {original_ms} → {datestr} → {recovered_ms}'

    def test_datestr_ends_with_z(self):
        """
        All date strings must end with Z (UTC indicator).
        This ensures downstream services correctly interpret timestamps as UTC.
        """
        result = Trade._milliseconds2datestr(1_704_900_600_000)
        assert result.endswith('Z'), f'Date string must end with Z (UTC). Got: {result}'


class TestTradeFromKrakenRestAPI:
    """
    Tests for Trade.from_kraken_rest_api_response()
    Verifies the full Trade object is created correctly from raw API data.
    """

    def test_creates_trade_with_correct_fields(self):
        """
        from_kraken_rest_api_response must correctly populate all fields.
        """
        trade = Trade.from_kraken_rest_api_response(
            pair='BTC/USD',
            price=65000.0,
            volume=0.5,
            timestamp_sec=1_704_900_600.0,
        )

        assert trade.pair == 'BTC/USD'
        assert trade.price == 65000.0
        assert trade.volume == 0.5
        assert trade.timestamp_ms == 1_704_900_600_000

    def test_timestamp_ms_correct_from_sec(self):
        """
        timestamp_sec (float, seconds) must be converted to
        timestamp_ms (int, milliseconds) correctly.
        1704900600.123 → 1704900600123
        """
        trade = Trade.from_kraken_rest_api_response(
            pair='BTC/USD',
            price=65000.0,
            volume=0.5,
            timestamp_sec=1_704_900_600.123,
        )
        assert trade.timestamp_ms == 1_704_900_600_123

    def test_timestamp_string_is_utc(self):
        """
        The timestamp string in the Trade object must be UTC (ends with Z).
        """
        trade = Trade.from_kraken_rest_api_response(
            pair='BTC/USD',
            price=65000.0,
            volume=0.5,
            timestamp_sec=0.0,
        )
        assert (
            trade.timestamp == '1970-01-01T00:00:00.000000Z'
        ), f'Expected UTC epoch, got {trade.timestamp}'


class TestTradeFromKrakenWebsocketAPI:
    """
    Tests for Trade.from_kraken_websocket_api_response()
    """

    def test_creates_trade_from_websocket_response(self):
        """
        from_kraken_websocket_api_response must parse the ISO timestamp
        correctly and convert to milliseconds.
        """
        trade = Trade.from_kraken_websocket_api_response(
            pair='BTC/USD',
            price=65000.0,
            volume=0.5,
            timestamp='2024-01-10T15:30:00.000000Z',
        )

        assert trade.pair == 'BTC/USD'
        assert trade.price == 65000.0
        assert trade.volume == 0.5
        assert trade.timestamp == '2024-01-10T15:30:00.000000Z'
        assert trade.timestamp_ms == 1_704_900_600_000

    def test_websocket_roundtrip_consistency(self):
        """
        Timestamp from REST API and WebSocket API must be consistent
        for the same point in time.
        """
        timestamp_sec = 1_704_900_600.0
        timestamp_str = '2024-01-10T15:30:00.000000Z'

        rest_trade = Trade.from_kraken_rest_api_response(
            pair='BTC/USD',
            price=65000.0,
            volume=0.5,
            timestamp_sec=timestamp_sec,
        )

        ws_trade = Trade.from_kraken_websocket_api_response(
            pair='BTC/USD',
            price=65000.0,
            volume=0.5,
            timestamp=timestamp_str,
        )

        assert rest_trade.timestamp_ms == ws_trade.timestamp_ms, (
            f'REST and WebSocket timestamps differ: '
            f'{rest_trade.timestamp_ms} vs {ws_trade.timestamp_ms}'
        )
