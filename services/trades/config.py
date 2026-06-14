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
    rest_api_max_retries: int = 5
    rest_api_initial_delay_seconds: int = 1
    cursor_dir: Optional[str] = 'cursors'


config = Config()
