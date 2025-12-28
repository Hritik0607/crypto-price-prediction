from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file='settings.env')
    kafka_broker_address: str
    kafka_input_topic: str
    kafka_output_topic: str
    kafka_consumer_group: str
    max_candles_in_state: int
    candle_seconds: int
    data_source: Literal['live', 'historical', 'test']


config = Config()
