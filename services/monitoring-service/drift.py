"""
drift.py — Model drift detection for the monitoring service.

Two types of drift detected:

1. DATA DRIFT — input features shift from training distribution
   Method: Z-score on each incoming candle feature
   Why z-score over KS test: KS test needs 50-100 accumulated values
   (50-100 min delay). Z-score works on a single candle immediately.
   Our architecture is real-time (1 candle/min) → z-score is correct.

2. CONCEPT DRIFT — model performance degrades over time
   Method: Rolling MAE threshold
   If rolling MAE > 2x training MAE → model is degrading
"""

from datetime import datetime, timezone

from comet_ml import API
from config import comet_ml_credentials, config
from elasticsearch import Elasticsearch
from loguru import logger

# ── Constants ──────────────────────────────────────────────────────────────────

# Features to skip for drift detection:
#   open/close — absolute price, always changes as BTC price changes (expected)
#   news_signals_signal — always 0 during training, no variance
DRIFT_SKIP_FEATURES = {'open', 'close', 'news_signals_signal'}

# Cooldown — avoid spamming ES with duplicate alerts
# Only write one drift event per feature per 5 minutes
DRIFT_ALERT_COOLDOWN_MS = 5 * 60 * 1000

# ── State ──────────────────────────────────────────────────────────────────────

# Loaded from CometML at startup
TRAINING_STATS: dict = {}

# Tracks last alert time per feature to enforce cooldown
# { 'rsi_14': 1234567890000, 'concept_BTC/USD': 1234567890000 }
_last_drift_alert: dict = {}


# ── Load training stats from CometML ──────────────────────────────────────────


def load_training_stats() -> dict:
    comet_api_key = comet_ml_credentials.api_key
    comet_workspace = comet_ml_credentials.workspace
    model_name = config.model_name
    model_status = config.model_status

    """
    Downloads training_feature_stats.json from CometML model registry.

    Replicates the same pattern used by price_predictor.py to load the model:
      1. Get model from registry
      2. Find Production version
      3. Get experiment key
      4. Download stats asset from that experiment

    Returns:
        dict of feature stats {feature: {mean, std, p5, p95}}
        Empty dict if download fails (drift detection disabled gracefully)
    """
    global TRAINING_STATS

    try:
        logger.info(
            f'Loading training feature stats from CometML ({model_status} model)'
        )

        comet_api = API(api_key=comet_api_key)

        # Step 1 — Get model from registry (same as price_predictor.py)
        model = comet_api.get_model(
            workspace=comet_workspace,
            model_name=model_name,
        )

        # Step 2 — Find Production version (same as price_predictor.py)
        model_versions = model.find_versions(status=model_status)
        model_version = sorted(model_versions, reverse=True)[0]

        # Step 3 — Get experiment key (same as price_predictor.py)
        experiment_key = model.get_details(version=model_version)['experimentKey']
        logger.info(f'Found {model_status} model experiment: {experiment_key}')

        # Step 4 — Get experiment and download stats asset
        experiment = comet_api.get_experiment_by_key(experiment_key)

        # Get list of assets to find our stats file
        assets = experiment.get_asset_list()
        stats_asset = next(
            (a for a in assets if a['fileName'] == 'training_feature_stats.json'),
            None,
        )

        if stats_asset is None:
            logger.warning(
                'training_feature_stats.json not found in CometML experiment. '
                'Data drift detection disabled. '
                'Run training.py to generate and upload the stats file.'
            )
            return {}

        # Download asset content
        asset_content = experiment.get_asset(
            stats_asset['assetId'],
            return_type='json',
        )

        TRAINING_STATS = asset_content
        logger.info(
            f'Loaded training feature stats for {len(TRAINING_STATS)} features: '
            f'{list(TRAINING_STATS.keys())}'
        )
        return TRAINING_STATS

    except Exception as e:
        logger.error(
            f'Failed to load training feature stats from CometML: {e}. '
            f'Drift detection disabled.'
        )
        return {}


# ── ElasticSearch index ────────────────────────────────────────────────────────


def ensure_model_drift_index(es: Elasticsearch):
    """
    Creates the model_drift ElasticSearch index if it does not exist.
    Stores both data drift and concept drift events.
    """
    if not es.indices.exists(index=config.model_drift_index):
        es.indices.create(
            index=config.model_drift_index,
            body={
                'mappings': {
                    'properties': {
                        'drift_type': {'type': 'keyword'},  # 'data' or 'concept'
                        'feature': {'type': 'keyword'},  # feature name or 'rolling_mae'
                        'value': {'type': 'float'},  # z-score or rolling MAE
                        'threshold': {'type': 'float'},  # threshold that was crossed
                        'severity': {'type': 'keyword'},  # 'warning' or 'critical'
                        'details': {'type': 'text'},  # human readable description
                        'pair': {'type': 'keyword'},
                        'timestamp_ms': {'type': 'long'},
                        'timestamp_iso': {'type': 'keyword'},
                    }
                }
            },
        )
        logger.info(f'Created ElasticSearch index: {config.model_drift_index}')
    else:
        logger.info(f'ElasticSearch index already exists: {config.model_drift_index}')


# ── Data drift ─────────────────────────────────────────────────────────────────


def compute_data_drift(candle: dict, es: Elasticsearch):
    """
    Checks if any feature in the incoming candle has drifted significantly
    from the training distribution using z-score.

    Z-score = (live_value - training_mean) / training_std

    Threshold: |z| > config.drift_z_threshold (default 3.0)
      → value is beyond 99.7% of training distribution
      → data drift detected

    Why z-score and not KS test:
      KS test needs 50-100 accumulated values → 50-100 min delay
      Z-score works on a single candle → immediate detection
      Real-time architecture (1 candle/min) → z-score is correct

    Args:
        candle: raw candle dict from Kafka with all technical indicators
        es: ElasticSearch client
    """
    if not TRAINING_STATS:
        return  # stats not loaded, drift detection disabled

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for feature, stats in TRAINING_STATS.items():
        # Skip absolute price and zero-variance features
        if feature in DRIFT_SKIP_FEATURES:
            continue

        live_value = candle.get(feature)
        if live_value is None:
            continue

        mean = stats['mean']
        std = stats['std']

        if std == 0:
            continue  # no variance in training — cannot compute z-score

        z_score = (live_value - mean) / std

        if abs(z_score) > config.drift_z_threshold:
            # Cooldown — avoid spamming same alert
            last_alert_ms = _last_drift_alert.get(feature, 0)
            if now_ms - last_alert_ms < DRIFT_ALERT_COOLDOWN_MS:
                continue

            # Beyond 4σ = critical, beyond 3σ = warning
            severity = 'critical' if abs(z_score) > 4.0 else 'warning'

            timestamp_iso = datetime.fromtimestamp(
                now_ms / 1000, tz=timezone.utc
            ).isoformat()

            drift_doc = {
                'drift_type': 'data',
                'feature': feature,
                'value': round(z_score, 2),
                'threshold': config.drift_z_threshold,
                'severity': severity,
                'details': (
                    f'{feature} z-score={z_score:.2f} '
                    f'(live={live_value:.4f}, '
                    f'training_mean={mean:.4f}, '
                    f'training_std={std:.4f})'
                ),
                'pair': candle.get('pair', 'unknown'),
                'timestamp_ms': now_ms,
                'timestamp_iso': timestamp_iso,
            }

            try:
                es.index(index=config.model_drift_index, document=drift_doc)
                _last_drift_alert[feature] = now_ms
                logger.warning(
                    f'DATA DRIFT: {feature} z={z_score:.2f} ({severity}) | '
                    f'live={live_value:.4f} vs training mean={mean:.4f} std={std:.4f}'
                )
            except Exception as e:
                logger.error(f'Failed to write data drift event: {e}')


# ── Concept drift ──────────────────────────────────────────────────────────────


def compute_concept_drift(pair: str, error_history: dict, es: Elasticsearch):
    """
    Checks if the model is degrading by computing rolling MAE.

    If avg MAE of last N predictions > 2x training MAE:
      → model performing significantly worse than during training
      → concept drift detected

    Training MAE: config.training_mae ($65.29 from last training run)
    Threshold:    config.concept_drift_threshold ($130.58 = 2x training MAE)
    Window:       config.concept_drift_window (last 10 predictions)

    Args:
        pair: e.g. 'BTC/USD'
        error_history: shared dict of recent errors per pair
        es: ElasticSearch client
    """
    errors = list(error_history[pair])

    if len(errors) < config.concept_drift_window:
        return  # not enough errors yet

    recent_errors = errors[-config.concept_drift_window :]
    rolling_mae = sum(e['abs_error'] for e in recent_errors) / len(recent_errors)

    if rolling_mae > config.concept_drift_threshold:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Cooldown per pair
        cooldown_key = f'concept_{pair}'
        last_alert_ms = _last_drift_alert.get(cooldown_key, 0)
        if now_ms - last_alert_ms < DRIFT_ALERT_COOLDOWN_MS:
            return

        # 1.5x threshold = critical
        severity = (
            'critical'
            if rolling_mae > config.concept_drift_threshold * 1.5
            else 'warning'
        )

        timestamp_iso = datetime.fromtimestamp(
            now_ms / 1000, tz=timezone.utc
        ).isoformat()

        drift_doc = {
            'drift_type': 'concept',
            'feature': 'rolling_mae',
            'value': round(rolling_mae, 2),
            'threshold': config.concept_drift_threshold,
            'severity': severity,
            'details': (
                f'Rolling MAE=${rolling_mae:.2f} over last '
                f'{config.concept_drift_window} predictions '
                f'exceeds threshold ${config.concept_drift_threshold:.2f} '
                f'(2x training MAE ${config.training_mae:.2f})'
            ),
            'pair': pair,
            'timestamp_ms': now_ms,
            'timestamp_iso': timestamp_iso,
        }

        try:
            es.index(index=config.model_drift_index, document=drift_doc)
            _last_drift_alert[cooldown_key] = now_ms
            logger.warning(
                f'CONCEPT DRIFT: {pair} rolling MAE=${rolling_mae:.2f} > '
                f'threshold=${config.concept_drift_threshold:.2f} ({severity})'
            )
        except Exception as e:
            logger.error(f'Failed to write concept drift event: {e}')


# ── Get recent alerts for dashboard ───────────────────────────────────────────


def get_drift_alerts_from_es(es: Elasticsearch) -> list:
    """
    Returns drift alerts from the last 1 hour.
    Used to populate the dashboard alert banner.
    Returns empty list if no alerts or drift detection disabled.
    """
    if not TRAINING_STATS:
        return []

    try:
        one_hour_ago_ms = int((datetime.now(timezone.utc).timestamp() - 3600) * 1000)
        result = es.search(
            index=config.model_drift_index,
            body={
                'query': {'range': {'timestamp_ms': {'gte': one_hour_ago_ms}}},
                'sort': [{'timestamp_ms': {'order': 'desc'}}],
                'size': 5,
            },
        )
        return [hit['_source'] for hit in result['hits']['hits']]
    except Exception as e:
        logger.error(f'Failed to get drift alerts from ES: {e}')
        return []
