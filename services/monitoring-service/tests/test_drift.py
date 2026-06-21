"""
tests/test_drift.py

Unit tests for drift detection logic in services/monitoring-service/drift.py

Tests the core mathematical logic:
  1. Z-score computation for data drift detection
  2. Rolling MAE threshold for concept drift detection
  3. Severity classification (warning vs critical)
  4. Cooldown mechanism (avoid alert spam)
  5. Skip features (open, close, news_signals_signal)

All ES and CometML calls are mocked — pure math only.
"""

from collections import defaultdict, deque
from unittest.mock import MagicMock, patch

import pytest

# ── Mock config before importing drift ────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_config(monkeypatch):
    """Mock config and comet_ml_credentials for all tests."""
    mock_cfg = MagicMock()
    mock_cfg.drift_z_threshold = 3.0
    mock_cfg.concept_drift_threshold = 130.58
    mock_cfg.concept_drift_window = 10
    mock_cfg.training_mae = 65.29
    mock_cfg.model_drift_index = 'model_drift'
    mock_cfg.model_name = (
        'price_predictor_pair_BTC_USD_candle_seconds_60_prediction_seconds_300'
    )
    mock_cfg.model_status = 'Production'

    mock_creds = MagicMock()
    mock_creds.api_key = 'mock_api_key'
    mock_creds.workspace = 'mock_workspace'

    import drift
    monkeypatch.setattr(drift, 'config', mock_cfg)
    monkeypatch.setattr(drift, 'comet_ml_credentials', mock_creds)

    yield mock_cfg


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_mock_es():
    es = MagicMock()
    es.indices.exists.return_value = True
    return es


def set_training_stats(stats: dict):
    import drift

    drift.TRAINING_STATS = stats


def reset_drift_state():
    import drift

    drift.TRAINING_STATS = {}
    drift._last_drift_alert = {}


# ── Z-score math tests ─────────────────────────────────────────────────────────


class TestZScoreMath:
    def test_zscore_no_drift_within_threshold(self, mock_config):
        """Value within 3σ must NOT trigger drift. z=1.0 → no alert."""
        import drift

        reset_drift_state()
        set_training_stats(
            {'rsi_14': {'mean': 50.0, 'std': 20.0, 'p5': 7.0, 'p95': 92.0}}
        )
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD', 'rsi_14': 70.0}, es)
        es.index.assert_not_called()

    def test_zscore_drift_above_threshold(self, mock_config):
        """Value beyond 3σ must trigger drift alert. z=3.25 → alert."""
        import drift

        reset_drift_state()
        set_training_stats(
            {'rsi_14': {'mean': 50.0, 'std': 20.0, 'p5': 7.0, 'p95': 92.0}}
        )
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD', 'rsi_14': 115.0}, es)
        es.index.assert_called_once()

    def test_zscore_drift_negative_direction(self, mock_config):
        """Negative z-score beyond threshold must also trigger alert. z=-4.0 → alert."""
        import drift

        reset_drift_state()
        set_training_stats(
            {'macd': {'mean': 0.0, 'std': 50.0, 'p5': -80.0, 'p95': 83.0}}
        )
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD', 'macd': -200.0}, es)
        es.index.assert_called_once()

    def test_zscore_exactly_at_threshold_no_alert(self, mock_config):
        """Value exactly at threshold (z=3.0) must NOT trigger. Only strictly greater."""
        import drift

        reset_drift_state()
        set_training_stats(
            {'rsi_14': {'mean': 50.0, 'std': 20.0, 'p5': 7.0, 'p95': 92.0}}
        )
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD', 'rsi_14': 110.0}, es)  # z=3.0
        es.index.assert_not_called()

    def test_zero_std_skipped(self, mock_config):
        """Features with std=0 must be skipped — division by zero prevention."""
        import drift

        reset_drift_state()
        set_training_stats(
            {'some_constant': {'mean': 5.0, 'std': 0.0, 'p5': 5.0, 'p95': 5.0}}
        )
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD', 'some_constant': 999.0}, es)
        es.index.assert_not_called()

    def test_missing_feature_in_candle_skipped(self, mock_config):
        """Missing candle features must be silently skipped — no crash."""
        import drift

        reset_drift_state()
        set_training_stats(
            {'rsi_14': {'mean': 50.0, 'std': 20.0, 'p5': 7.0, 'p95': 92.0}}
        )
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD'}, es)  # rsi_14 missing
        es.index.assert_not_called()


# ── Skip features tests ────────────────────────────────────────────────────────


class TestSkipFeatures:
    def test_open_never_triggers_drift(self, mock_config):
        """open is absolute price — ignored for drift detection."""
        import drift

        reset_drift_state()
        set_training_stats(
            {'open': {'mean': 73933.0, 'std': 4675.0, 'p5': 66748.0, 'p95': 81037.0}}
        )
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD', 'open': 999999.0}, es)
        es.index.assert_not_called()

    def test_close_never_triggers_drift(self, mock_config):
        """close is absolute price — ignored for drift detection."""
        import drift

        reset_drift_state()
        set_training_stats(
            {'close': {'mean': 73933.0, 'std': 4675.0, 'p5': 66748.0, 'p95': 81037.0}}
        )
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD', 'close': 1.0}, es)
        es.index.assert_not_called()

    def test_news_signals_never_triggers_drift(self, mock_config):
        """news_signals_signal is always 0 — ignored for drift detection."""
        import drift

        reset_drift_state()
        set_training_stats(
            {'news_signals_signal': {'mean': 0.0, 'std': 0.0, 'p5': 0.0, 'p95': 0.0}}
        )
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD', 'news_signals_signal': 999.0}, es)
        es.index.assert_not_called()


# ── Severity tests ─────────────────────────────────────────────────────────────


class TestSeverityClassification:
    def _get_doc(self, es):
        call_args = es.index.call_args
        return call_args.kwargs.get('document') or call_args[1].get('document')

    def test_warning_severity_between_3_and_4_sigma(self, mock_config):
        """z between 3σ and 4σ → warning. mean=0, std=10, live=35 → z=3.5"""
        import drift

        reset_drift_state()
        set_training_stats(
            {'atr': {'mean': 0.0, 'std': 10.0, 'p5': -20.0, 'p95': 20.0}}
        )
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD', 'atr': 35.0}, es)
        assert self._get_doc(es)['severity'] == 'warning'

    def test_critical_severity_beyond_4_sigma(self, mock_config):
        """z beyond 4σ → critical. mean=0, std=10, live=50 → z=5.0"""
        import drift

        reset_drift_state()
        set_training_stats(
            {'atr': {'mean': 0.0, 'std': 10.0, 'p5': -20.0, 'p95': 20.0}}
        )
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD', 'atr': 50.0}, es)
        assert self._get_doc(es)['severity'] == 'critical'


# ── Cooldown tests ─────────────────────────────────────────────────────────────


class TestCooldown:
    def test_second_alert_within_cooldown_suppressed(self, mock_config):
        """Two drift events within 5 minutes must only write ONE alert to ES."""
        import drift

        reset_drift_state()
        set_training_stats(
            {'atr': {'mean': 0.0, 'std': 10.0, 'p5': -20.0, 'p95': 20.0}}
        )
        es = make_mock_es()
        candle = {'pair': 'BTC/USD', 'atr': 50.0}

        drift.compute_data_drift(candle, es)
        assert es.index.call_count == 1

        drift.compute_data_drift(candle, es)
        assert es.index.call_count == 1  # still 1 — suppressed by cooldown


# ── Concept drift tests ────────────────────────────────────────────────────────


class TestConceptDrift:
    def _make_error_history(self, abs_errors: list) -> dict:
        history = defaultdict(lambda: deque(maxlen=100))
        for err in abs_errors:
            history['BTC/USD'].append({'abs_error': err})
        return history

    def _get_doc(self, es):
        call_args = es.index.call_args
        return call_args.kwargs.get('document') or call_args[1].get('document')

    def test_no_drift_when_mae_below_threshold(self, mock_config):
        """Rolling MAE $65 < threshold $130.58 → no alert."""
        import drift

        reset_drift_state()
        es = make_mock_es()
        drift.compute_concept_drift(
            'BTC/USD', self._make_error_history([65.0] * 10), es
        )
        es.index.assert_not_called()

    def test_drift_when_mae_above_threshold(self, mock_config):
        """Rolling MAE $200 > threshold $130.58 → alert."""
        import drift

        reset_drift_state()
        es = make_mock_es()
        drift.compute_concept_drift(
            'BTC/USD', self._make_error_history([200.0] * 10), es
        )
        es.index.assert_called_once()

    def test_no_drift_when_insufficient_errors(self, mock_config):
        """Need at least 10 errors before computing rolling MAE."""
        import drift

        reset_drift_state()
        es = make_mock_es()
        drift.compute_concept_drift(
            'BTC/USD', self._make_error_history([200.0] * 5), es
        )
        es.index.assert_not_called()

    def test_rolling_mae_uses_last_n_errors_only(self, mock_config):
        """
        Rolling MAE uses LAST N errors only.
        First 90: $200 (high), last 10: $50 (low) → no alert.
        """
        import drift

        reset_drift_state()
        es = make_mock_es()
        drift.compute_concept_drift(
            'BTC/USD', self._make_error_history([200.0] * 90 + [50.0] * 10), es
        )
        es.index.assert_not_called()

    def test_concept_drift_warning_severity(self, mock_config):
        """Rolling MAE $150 between threshold and 1.5x → warning."""
        import drift

        reset_drift_state()
        es = make_mock_es()
        drift.compute_concept_drift(
            'BTC/USD', self._make_error_history([150.0] * 10), es
        )
        assert self._get_doc(es)['severity'] == 'warning'

    def test_concept_drift_critical_severity(self, mock_config):
        """Rolling MAE $516 (real June 2026 value) beyond 1.5x threshold → critical."""
        import drift

        reset_drift_state()
        es = make_mock_es()
        drift.compute_concept_drift(
            'BTC/USD', self._make_error_history([516.0] * 10), es
        )
        assert self._get_doc(es)['severity'] == 'critical'

    def test_concept_drift_document_fields(self, mock_config):
        """Drift document must have all required fields."""
        import drift

        reset_drift_state()
        es = make_mock_es()
        drift.compute_concept_drift(
            'BTC/USD', self._make_error_history([200.0] * 10), es
        )
        doc = self._get_doc(es)
        assert doc['drift_type'] == 'concept'
        assert doc['feature'] == 'rolling_mae'
        assert doc['pair'] == 'BTC/USD'
        assert 'value' in doc
        assert 'threshold' in doc
        assert 'severity' in doc
        assert 'details' in doc
        assert 'timestamp_ms' in doc
        assert 'timestamp_iso' in doc

    def test_no_drift_when_training_stats_empty(self, mock_config):
        """If TRAINING_STATS empty (CometML failed), data drift silently disabled."""
        import drift

        reset_drift_state()
        drift.TRAINING_STATS = {}
        es = make_mock_es()
        drift.compute_data_drift({'pair': 'BTC/USD', 'rsi_14': 999.0}, es)
        es.index.assert_not_called()
