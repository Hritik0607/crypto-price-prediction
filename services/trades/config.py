from typing import List, Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='settings.env', env_file_encoding='utf-8'
    )
    kafka_broker_address: str
    kafka_topic: str
    pairs: List[str]
    data_source: Literal['live', 'historical', 'test']
    last_n_days: Optional[int] = None


config = Config()
