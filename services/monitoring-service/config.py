from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='settings.env',
        env_file_encoding='utf-8',
    )

    kafka_broker_address: str
    kafka_input_topic: str = 'candles'
    kafka_consumer_group: str = 'monitoring_service_group'

    elasticsearch_url: str = 'http://localhost:9200'
    predictions_index: str = 'price_prediction'
    model_errors_index: str = 'model_errors'
    model_drift_index: str = 'model_drift'

    pairs_to_monitor: List[str] = ['BTC/USD', 'ETH/USD', 'XRP/USD']
    prediction_seconds: int = 300  # 5 minutes

    port: int = 8082

    # ── Drift detection ────────────────────────────────────────────────────
    drift_z_threshold: float = 3.0  # z-score beyond which = data drift
    training_mae: float = 65.29  # from last training run
    concept_drift_threshold: float = 130.58  # 2x training MAE
    concept_drift_window: int = 10  # rolling window of errors to average

    # ── model info ─────────────────────────────────
    model_name: str = (
        'price_predictor_pair_BTC_USD_candle_seconds_60_prediction_seconds_300'
    )
    model_status: str = 'Production'


config = Config()


class CometMlCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_file='comet_ml_credentials.env')
    api_key: str
    project_name: str
    workspace: str


comet_ml_credentials = CometMlCredentials()
