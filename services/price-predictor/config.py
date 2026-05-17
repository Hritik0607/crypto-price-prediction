from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='training.settings.env')

    # feature view to read data form both at training time and prediction time
    feature_view_name: str = Field(description='The name of the feature view')
    feature_view_version: int = Field(description='The version of the feature view')

    pair_to_predict: str = Field(description='The pair to train the model on')
    candle_seconds: int = Field(description='The number of seconds per candle')
    prediction_seconds: int = Field(
        description='The number of seconds into the future to predict'
    )
    pairs_as_features: list[str] = Field(
        description='The pairs to use for the features'
    )
    technical_indicators_as_features: list[str] = Field(
        description='The technical indicators to use for from the technical_indicators feature group'
    )
    days_back: int = Field(
        description='The number of days to consider for the historical data'
    )
    llm_model_name_news_signals: str = Field(
        description='The name of the LLM model to use for the news signals'
    )

    # hyperparameter tuning
    hyperparameter_tuning_search_trials: Optional[int] = Field(
        default=0,
        description='The number of trials to perform for hyperparameter tuning',
    )
    hyperparameter_tuning_n_splits: Optional[int] = Field(
        default=3,
        description='The number of splits to perform for hyperparameter tuning',
    )

    # model registry
    model_status: Literal['Development', 'Staging', 'Production'] = Field(
        default='Development',
        description='The status of the model in the model registry',
    )

    technical_indicators_feature_group_name: Optional[str] = None
    technical_indicators_feature_group_version: Optional[int] = None
    news_signals_feature_group_name: Optional[str] = None
    news_signals_feature_group_version: Optional[int] = None


training_config = TrainingConfig()


class InferenceConfig(BaseSettings):
    """
    Configuration for the inference job

    These are the parameters that are used to load the model from the model registry
    and to generate the predictions

    Observe how most of the parameters from the trianing job are not here. Instead,
    they are loaded from the model registry. More precisely, from the experiment run
    that generated the model artifact we load from the registry.
    """

    model_config = SettingsConfigDict(env_file='inference.settings.env')

    # pair and frequency for which we want to predict
    pair_to_predict: str = Field(description='The pair to train the model on')
    candle_seconds: int = Field(description='The number of seconds per candle')
    prediction_seconds: int = Field(
        description='The number of seconds into the future to predict'
    )

    # environment of the model we want to use for inference
    model_status: Literal['Development', 'Staging', 'Production'] = Field(
        description='The status of the model in the model registry that we want to use for inference'
    )

    # kafka config
    # To listen to candle events and trigger predictions
    kafka_broker_address: str = Field(description='The address of the kafka broker')
    kafka_input_topic: str = Field(
        description='The topic to listen to for candle events'
    )
    kafka_consumer_group: str = Field(
        description='The consumer group to use for the kafka consumer'
    )

    # Elastic Search config
    elasticsearch_url: str = Field(description='The URL of the Elastic Search instance')
    elasticsearch_index: str = Field(
        description='The index to write the predictions to'
    )


inference_config = InferenceConfig()


class HopsworksCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_file='hopsworks_credentials.env')
    api_key: str
    project_name: str
    hopsworks_host: str


class CometMlCredentials(BaseSettings):
    model_config = SettingsConfigDict(env_file='comet_ml_credentials.env')
    api_key: str
    project_name: str
    workspace: str


hopsworks_credentials = HopsworksCredentials()
comet_ml_credentials = CometMlCredentials()
