from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Configuration for the news service.
    """

    model_config = SettingsConfigDict(env_file='settings.env')
    kafka_broker_address: str
    kafka_topic: str
    data_source: Literal['live', 'historical']

    polling_interval_sec: Optional[int] = 10
    historical_data_source_url_rar_file: Optional[str] = None
    historical_data_source_csv_file: Optional[str] = None
    historical_days_back: Optional[int] = 180


config = Config()


class CryptopanicConfig(BaseSettings):
    """
    Configuration for the Cryptopanic API.
    """

    model_config = SettingsConfigDict(env_file='cryptopanic_credentials.env')
    api_key: str


cryptopanic_config = CryptopanicConfig()
