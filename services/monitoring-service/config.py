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

    pairs_to_monitor: List[str] = ['BTC/USD', 'ETH/USD', 'XRP/USD']
    prediction_seconds: int = 300  # 5 minutes

    port: int = 8082


config = Config()
