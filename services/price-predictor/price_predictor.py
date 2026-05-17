import json
import time
from datetime import datetime, timezone
from typing import Literal, Tuple

import joblib
import pandas as pd
from comet_ml.api import API
from config import CometMlCredentials as CometConfig
from config import HopsworksCredentials as HopsworksConfig
from feature_reader import FeatureReader
from loguru import logger
from models.xgboost_model import XGBRegressor
from names import get_model_name
from pydantic import BaseModel

Model = Tuple[XGBRegressor]


class PredictionOutput(BaseModel):
    pair: str
    candle_seconds: int
    prediction_seconds: int
    prediction: float

    # the timestamp when we make the prediction
    timestamp_ms: int
    timestamp_iso: str

    # the timestamp for which we want to predict the predicted price is `prediction`
    predicted_timestamp_ms: int
    predicted_timestamp_iso: str

    def to_dict(self) -> dict:
        return self.model_dump()


class PricePredictor:
    def __init__(
        self,
        pair_to_predict: str,
        candle_seconds: int,
        prediction_seconds: int,
        model_status: Literal['Development', 'Staging', 'Production'],
        comet_config: CometConfig,
        hopsworks_config: HopsworksConfig,
    ):
        """
        Loads the model from the model registry and the necessary metadata.

        Args:
            pair_to_predict: The pair to predict.
            candle_seconds: The number of seconds in each candle.
            prediction_seconds: The number of seconds to predict.
            model_status: The status of the model to load.
            comet_config:
                The Comet configuration with credentials necesseray to load
                model artifacts from the model registry, and experiment runs data from CometML.
            hopsworks_config:
                The Hopsworks configuration with credentials necesseray to load
                features from the feature store.
        """
        self.pair_to_predict = pair_to_predict
        self.candle_seconds = candle_seconds
        self.prediction_seconds = prediction_seconds

        self.model_name = get_model_name(
            pair_to_predict, candle_seconds, prediction_seconds
        )
        comet_api = API(api_key=comet_config.api_key)

        logger.info(f'Loading model {self.model_name} from the model registry')
        self.model, self.experiment_key = self._get_model_from_model_registry(
            model_name=self.model_name,
            model_status=model_status,
            comet_api=comet_api,
            comet_config=comet_config,
        )
        logger.info(f'Loaded model {self.model_name} from the model registry')

        logger.info(
            f'Loading inference params from experiment {self.experiment_key} metadata from CometML'
        )
        self._get_and_set_inference_params(
            comet_api=comet_api,
            experiment_key=self.experiment_key,
        )
        logger.info('Loaded inference params!')

        # Initialize the feature reader
        # This is the object that talks to Hopsworks which helps us get the features in
        # real time that our self.model needs to make predictions
        self.feature_reader = self._get_feature_reader(hopsworks_config)

        logger.info(f'Model {self.model_name} is ready for inference!')

    def _get_feature_reader(self, hopsworks_config: HopsworksConfig) -> FeatureReader:
        """
        Initializes the feature reader object that talks to Hopsworks which helps us get the features in
        real time that our self.model needs to make predictions
        """
        logger.info('Initializing feature reader')
        return FeatureReader(
            hopsworks_project_name=hopsworks_config.project_name,
            hopsworks_api_key=hopsworks_config.api_key,
            feature_view_name=self.inference_params['feature_view_name'],
            feature_view_version=self.inference_params['feature_view_version'],
            pair_to_predict=self.inference_params['pair_to_predict'],
            candle_seconds=self.inference_params['candle_seconds'],
            pairs_as_features=self.inference_params['pairs_as_features'],
            technical_indicators_as_features=self.inference_params[
                'technical_indicators_as_features'
            ],
            prediction_seconds=self.inference_params['prediction_seconds'],
            llm_model_name_news_signals=self.inference_params[
                'llm_model_name_news_signals'
            ],
        )

    def predict(self) -> PredictionOutput:
        """
        Generates a new prediction using the latest features from the feature store.

        Steps:
        - Get the latest features from the feature store
        - Make the prediction using the `self.model` and these features
        - Return the prediction
        """
        # get the latest features from the feature store
        features: pd.DataFrame = self.feature_reader.get_inference_features()

        # make the prediction
        prediction: float = self.model.predict(features)[0]

        # build the output
        timestamp_ms = int(time.time() * 1000)
        predicted_timestamp_ms = timestamp_ms + self.prediction_seconds * 1000

        # transform unix milliseconds to iso format
        timestamp_iso = datetime.fromtimestamp(
            timestamp_ms / 1000, tz=timezone.utc
        ).isoformat()
        predicted_timestamp_iso = datetime.fromtimestamp(
            predicted_timestamp_ms / 1000, tz=timezone.utc
        ).isoformat()

        return PredictionOutput(
            pair=self.pair_to_predict,
            candle_seconds=self.candle_seconds,
            prediction_seconds=self.prediction_seconds,
            prediction=prediction,
            # the timestamp when we make the prediction
            timestamp_ms=timestamp_ms,
            timestamp_iso=timestamp_iso,
            # the timestamp when we want to predict
            predicted_timestamp_ms=predicted_timestamp_ms,
            predicted_timestamp_iso=predicted_timestamp_iso,
        )

    def _get_model_from_model_registry(
        self,
        model_name: str,
        model_status: Literal['Development', 'Staging', 'Production'],
        comet_api: API,
        comet_config: CometConfig,
    ) -> Tuple[Model, str]:
        """
        Loads the model from the model registry, and returns the model and the
        corresponding experiment run key that generated that model artifact.

        Args:
            model_name: The name of the model to load from the model registry
            model_status: The status of the model to load from the model registry
            comet_config: The Comet configuration with credentials necesseray to load
                model artifacts from the model registry, and experiment runs data from CometML.

        Returns:
            model: The model object loaded from the model registry
            comet_ml_experiment_key: The experiment run key that generated the model artifact
        """
        # Step 1: Download the model artifact from the model registry
        model = comet_api.get_model(
            workspace=comet_config.workspace,
            model_name=model_name,
        )

        # find the version for the current model with the given `status`
        # Here I am assuming there is only one model version for that status.
        # As for dev, or staging, can have multiple versions, so we sort by
        # version and get the latest one.
        model_versions = model.find_versions(status=model_status)

        # sort the model versions list from high to low and pick the first element
        model_version = sorted(model_versions, reverse=True)[0]

        # download the model artifact for this `model_version`
        model.download(version=model_version, output_folder='./')

        # find the experiment associated with this model
        experiment_key = model.get_details(version=model_version)['experimentKey']

        # load the model from the file to memory
        model_file = f'./{model_name}.joblib'
        model = joblib.load(model_file)

        return model, experiment_key

    def _get_and_set_inference_params(
        self,
        comet_api: API,
        experiment_key: str,
    ):
        """
        Fetches the experiment metadata from CometML and stores it in the class.

        These are the parameters we need for inference:

        These are the ones we already have
        - pair_to_predict: str
        - candle_seconds: int
        - prediction_seconds: int

        These are the ones we need to get from the experiment metadata
        - feature_view_name: str,
        - feature_view_version: int,
        - pairs_as_features: list[str]
        - technical_indicators_as_features: list[str]
        - llm_model_name_news_signals: str
        """
        # get the experiment object
        experiment = comet_api.get_experiment_by_key(experiment_key)

        # feature view name and version
        feature_view_name = experiment.get_parameters_summary('feature_view_name')[
            'valueCurrent'
        ]
        feature_view_version = int(
            experiment.get_parameters_summary('feature_view_version')['valueCurrent']
        )

        # pairs_as_features
        pairs_as_features: str = experiment.get_parameters_summary('pairs_as_features')[
            'valueCurrent'
        ]
        # parse string into list
        pairs_as_features = json.loads(pairs_as_features)

        # technical_indicators_as_features
        technical_indicators_as_features: str = experiment.get_parameters_summary(
            'technical_indicators_as_features'
        )['valueCurrent']
        # parse string into list
        technical_indicators_as_features = json.loads(technical_indicators_as_features)

        # llm_model_name_news_signals
        llm_model_name_news_signals = experiment.get_parameters_summary(
            'llm_model_name_news_signals'
        )['valueCurrent']

        # These are parameters we need for inference
        self.inference_params = {
            'pair_to_predict': self.pair_to_predict,
            'candle_seconds': self.candle_seconds,
            'prediction_seconds': self.prediction_seconds,
            'feature_view_name': feature_view_name,
            'feature_view_version': feature_view_version,
            'pairs_as_features': pairs_as_features,
            'technical_indicators_as_features': technical_indicators_as_features,
            'llm_model_name_news_signals': llm_model_name_news_signals,
        }

        logger.info(f'Inference params: {self.inference_params}')


if __name__ == '__main__':
    from config import (
        comet_ml_credentials,
        hopsworks_credentials,
    )
    from config import (
        inference_config as config,
    )

    price_predictor = PricePredictor(
        pair_to_predict=config.pair_to_predict,
        candle_seconds=config.candle_seconds,
        prediction_seconds=config.prediction_seconds,
        model_status=config.model_status,
        comet_config=comet_ml_credentials,
        hopsworks_config=hopsworks_credentials,
    )

    output = price_predictor.predict()
    logger.info(f'Prediction output: {output}')
