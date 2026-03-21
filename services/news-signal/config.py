from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file='settings.env')

    kafka_broker_address: str
    kafka_input_topic: str
    kafka_output_topic: str
    kafka_consumer_group: str

    model_provider: Literal['anthropic', 'ollama', 'dummy']
    data_source: Literal['live', 'historical']


config = Config()
