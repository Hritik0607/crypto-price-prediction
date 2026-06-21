"""
tests/test_feature_reader.py

Unit tests for the critical preprocessing logic in feature_reader.py.

Key fix tested:
    BEFORE: target = future_price (absolute)
            Model memorized price levels ($67k-$82k range)
            Test MAE: $859 (14x worse than baseline)

    AFTER:  target = future_price - current_price (price DELTA)
            Model learns price MOVEMENT patterns
            Test MAE: $65 (near baseline of $64)

We test _preprocess_raw_features_into_features_and_target() in isolation
by instantiating a minimal version without Hopsworks connection.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Helper — create FeatureReader without Hopsworks ───────────────────────────


def make_feature_reader(
    pair_to_predict='BTC/USD',
    pairs_as_features=None,
    technical_indicators=None,
    prediction_seconds=300,
):
    """
    Creates a FeatureReader instance with Hopsworks connection mocked out.
    Allows testing preprocessing logic without real credentials.
    """
    if pairs_as_features is None:
        pairs_as_features = ['BTC/USD']
    if technical_indicators is None:
        technical_indicators = ['rsi_14', 'macd', 'adx']

    with patch('feature_reader.hopsworks') as mock_hopsworks:
        mock_project = MagicMock()
        mock_fs = MagicMock()
        mock_fv = MagicMock()
        mock_hopsworks.login.return_value = mock_project
        mock_project.get_feature_store.return_value = mock_fs
        mock_fs.get_feature_view.return_value = mock_fv

        from feature_reader import FeatureReader

        reader = FeatureReader(
            hopsworks_host='mock_host',
            hopsworks_project_name='mock_project',
            hopsworks_api_key='mock_key',
            feature_view_name='price_predictor',
            feature_view_version=2,
            pair_to_predict=pair_to_predict,
            candle_seconds=60,
            pairs_as_features=pairs_as_features,
            technical_indicators_as_features=technical_indicators,
            prediction_seconds=prediction_seconds,
            llm_model_name_news_signals='dummy',
        )
    return reader


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def reader():
    return make_feature_reader()


@pytest.fixture
def sample_data():
    """
    Minimal DataFrame replicating the output of the Parquet JOIN.
    4 rows for BTC/USD at 1-minute intervals.
    prediction_seconds=300 (5 min) so target uses price 5 candles ahead.
    """
    return pd.DataFrame(
        {
            'pair': ['BTC/USD'] * 6,
            'window_end_ms': [
                1_000_000,  # t=0  (will become target for t=-300000)
                1_060_000,  # t+1min
                1_120_000,  # t+2min
                1_180_000,  # t+3min
                1_240_000,  # t+4min
                1_300_000,  # t+5min → target for t=0
            ],
            'open': [65000.0, 65010.0, 65020.0, 65030.0, 65040.0, 65060.0],
            'close': [65000.0, 65010.0, 65020.0, 65030.0, 65040.0, 65060.0],
            'rsi_14': [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
            'macd': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'adx': [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
            'news_signals_signal': [0, 0, 0, 0, 0, 0],
            'coin': ['BTC', 'BTC', 'BTC', 'BTC', 'BTC', 'BTC'],
        }
    )


# ── Tests: price delta target ──────────────────────────────────────────────────


class TestPriceDeltaTarget:
    """
    The most critical fix: target must be price DELTA not absolute price.

    BEFORE fix: target = future_close ($65,060)
      - Model memorizes price levels
      - Fails when BTC moves to new regime ($63k vs training $67k-$82k)
      - Test MAE: $859

    AFTER fix: target = future_close - current_close ($60.0)
      - Model learns price movement patterns
      - Generalizes across price regimes
      - Test MAE: $65
    """

    def test_target_is_price_delta_not_absolute(self, reader, sample_data):
        """
        Target must be future_price - current_price (delta),
        NOT future_price (absolute).
        """
        result = reader._preprocess_raw_features_into_features_and_target(
            sample_data.copy(),
            add_target_column=True,
        )

        # Target should be a small delta, not an absolute price like $65,000
        assert 'target' in result.columns
        max_abs_target = result['target'].abs().max()

        # Delta should be small (price change over 5 min)
        # NOT close to $65,000 (absolute price)
        assert max_abs_target < 1000, (
            f'Target looks like absolute price (max={max_abs_target:.2f}). '
            f'Expected small delta. '
            f'Bug: target = future_price instead of future_price - current_price'
        )

    def test_target_equals_future_minus_current(self, reader, sample_data):
        """
        For a known sequence, verify target = future_close - current_close.

        Given prediction_seconds=300 (5 min = 5 rows at 60s intervals):
          Row at t=0: close=$65,000, future close (t+5min)=$65,060
          Expected target = 65,060 - 65,000 = +$60
        """
        result = reader._preprocess_raw_features_into_features_and_target(
            sample_data.copy(),
            add_target_column=True,
        )

        assert len(result) > 0, 'Result is empty — no rows with valid targets'

        # First row: close=65000, future close (5 rows ahead)=65060
        # target should be 65060 - 65000 = 60
        first_row = result.iloc[0]
        expected_delta = 65060.0 - 65000.0  # = 60.0
        actual_delta = first_row['target']

        assert abs(actual_delta - expected_delta) < 0.01, (
            f'Target delta wrong: expected {expected_delta}, got {actual_delta}. '
            f'Bug: target computation is incorrect.'
        )

    def test_target_can_be_negative(self, reader):
        """
        Target must be negative when future price is LOWER than current.
        This tests that the model can learn downward movements too.
        """
        # Price goes DOWN: 65000 → 64900
        data = pd.DataFrame(
            {
                'pair': ['BTC/USD'] * 6,
                'window_end_ms': [
                    1_000_000,
                    1_060_000,
                    1_120_000,
                    1_180_000,
                    1_240_000,
                    1_300_000,
                ],
                'open': [65000.0] * 6,
                'close': [
                    65000.0,
                    65000.0,
                    65000.0,
                    65000.0,
                    65000.0,
                    64900.0,
                ],  # drops at end
                'rsi_14': [50.0] * 6,
                'macd': [0.1] * 6,
                'adx': [30.0] * 6,
                'news_signals_signal': [0] * 6,
                'coin': ['BTC'] * 6,
            }
        )

        result = reader._preprocess_raw_features_into_features_and_target(
            data,
            add_target_column=True,
        )

        assert len(result) > 0
        # First row: current=65000, future=64900, delta = -100
        first_target = result.iloc[0]['target']
        assert first_target < 0, (
            f'Target should be negative when price drops. Got {first_target}. '
            f'Bug: delta sign is wrong.'
        )

    def test_target_zero_when_price_unchanged(self, reader):
        """
        Target must be zero when future price equals current price.
        """
        data = pd.DataFrame(
            {
                'pair': ['BTC/USD'] * 6,
                'window_end_ms': [
                    1_000_000,
                    1_060_000,
                    1_120_000,
                    1_180_000,
                    1_240_000,
                    1_300_000,
                ],
                'open': [65000.0] * 6,
                'close': [65000.0] * 6,  # price never changes
                'rsi_14': [50.0] * 6,
                'macd': [0.1] * 6,
                'adx': [30.0] * 6,
                'news_signals_signal': [0] * 6,
                'coin': ['BTC'] * 6,
            }
        )

        result = reader._preprocess_raw_features_into_features_and_target(
            data,
            add_target_column=True,
        )

        assert len(result) > 0
        assert (
            abs(result.iloc[0]['target']) < 0.01
        ), f'Target should be ~0 when price unchanged. Got {result.iloc[0]["target"]}'


# ── Tests: add_target_column=False (inference mode) ───────────────────────────


class TestInferenceMode:
    """
    Tests that add_target_column=False works correctly for inference.
    At inference time we do NOT have future prices — no target column.
    """

    def test_no_target_column_when_flag_false(self, reader, sample_data):
        """
        When add_target_column=False, result must NOT have a 'target' column.
        """
        result = reader._preprocess_raw_features_into_features_and_target(
            sample_data.copy(),
            add_target_column=False,
        )
        assert (
            'target' not in result.columns
        ), 'target column should not exist in inference mode'

    def test_all_rows_preserved_in_inference_mode(self, reader, sample_data):
        """
        When add_target_column=False, no rows should be dropped.
        (In training mode, rows without a future price are dropped.)
        """
        result = reader._preprocess_raw_features_into_features_and_target(
            sample_data.copy(),
            add_target_column=False,
        )
        assert len(result) == len(sample_data), (
            f'Expected {len(sample_data)} rows, got {len(result)}. '
            f'Rows should not be dropped in inference mode.'
        )


# ── Tests: preprocessing output structure ─────────────────────────────────────


class TestPreprocessingStructure:
    """
    Tests that the output DataFrame has the correct structure.
    """

    def test_pair_columns_dropped(self, reader, sample_data):
        """
        Columns starting with 'pair' must be dropped.
        They are categorical strings — not useful as model features.
        """
        result = reader._preprocess_raw_features_into_features_and_target(
            sample_data.copy(),
            add_target_column=False,
        )
        pair_cols = [c for c in result.columns if c.startswith('pair')]
        assert (
            len(pair_cols) == 0
        ), f'pair columns should be dropped but found: {pair_cols}'

    def test_window_end_ms_renamed_to_timestamp_ms(self, reader, sample_data):
        """
        window_end_ms must be renamed to timestamp_ms in the output.
        """
        result = reader._preprocess_raw_features_into_features_and_target(
            sample_data.copy(),
            add_target_column=False,
        )
        assert (
            'timestamp_ms' in result.columns
        ), 'window_end_ms should be renamed to timestamp_ms'
        assert (
            'window_end_ms' not in result.columns
        ), 'window_end_ms should not exist after rename'

    def test_output_sorted_by_timestamp(self, reader, sample_data):
        """
        Output must be sorted by timestamp_ms ascending.
        Important for time-series train/test split.
        """
        # Shuffle input to verify sorting happens
        shuffled = sample_data.sample(frac=1, random_state=42).reset_index(drop=True)

        result = reader._preprocess_raw_features_into_features_and_target(
            shuffled,
            add_target_column=False,
        )

        timestamps = result['timestamp_ms'].tolist()
        assert timestamps == sorted(
            timestamps
        ), 'Output must be sorted by timestamp_ms ascending'

    def test_feature_columns_present(self, reader, sample_data):
        """
        All technical indicator columns must be present in output.
        """
        result = reader._preprocess_raw_features_into_features_and_target(
            sample_data.copy(),
            add_target_column=False,
        )
        for feature in ['rsi_14', 'macd', 'adx']:
            assert feature in result.columns, f'Feature {feature} missing from output'

    def test_rows_dropped_when_no_future_price(self, reader, sample_data):
        """
        In training mode, rows without a matching future price must be dropped.
        With 6 rows and prediction_seconds=300 (5 rows ahead),
        only the first row will have a valid target.
        Later rows do not have a price 5 rows ahead → dropped.
        """
        result = reader._preprocess_raw_features_into_features_and_target(
            sample_data.copy(),
            add_target_column=True,
        )
        # 6 rows - 5 rows without future = 1 row with valid target
        assert len(result) == 1, f'Expected 1 row with valid target, got {len(result)}'
