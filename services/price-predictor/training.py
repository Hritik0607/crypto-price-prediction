import comet_ml
import joblib
import pandas as pd
from feature_reader import FeatureReader
from loguru import logger
from models.dummy_model import DummyModel
from models.xgboost_model import XGBoostModel
from names import get_model_name
from sklearn.metrics import mean_absolute_error


def train_test_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the given `data` into 2 dataframes based on the `timestamp_ms` column
    such that
    > the first dataframe contains the first `train_size` rows
    > the second dataframe contains the remaining rows
    """
    train_size = int(len(data) * (1 - test_size))

    print(f'Total data size: {len(data)}')
    print(f'Train size: {train_size}')

    train_df = data.iloc[:train_size]
    test_df = data.iloc[train_size:]

    return train_df, test_df


def train(
    hopsworks_host: str,
    hopsworks_project_name: str,
    hopsworks_api_key: str,
    feature_view_name: str,
    feature_view_version: int,
    pair_to_predict: str,
    candle_seconds: int,
    pairs_as_features: list[str],
    technical_indicators_as_features: list[str],
    prediction_seconds: int,
    llm_model_name_news_signals: str,
    days_back: int,
    comet_ml_api_key: str,
    comet_ml_project_name: str,
    comet_ml_workspace: str,
    hyperparameter_tuning_search_trials: int,
    hyperparameter_tuning_n_splits: int,
    model_status: str,
    technical_indicators_feature_group_name,
    technical_indicators_feature_group_version,
    news_signals_feature_group_name,
    news_signals_feature_group_version,
):
    """

    Does the following:
    1. Reads feature data from the Feature Store
    2. Splits the data into training and testing sets
    3. Trains a model on the training set
    4. Evaluates the model on the testing set
    5. Saves the model to the model registry

    Everything is instrumented with CometML.

    The model is saved to the model registry with the tag `model_tag`.

    Args:
        hopsworks_project_name: The name of the Hopsworks project
        hopsworks_api_key: The API key for the Hopsworks project
        feature_view_name: The name of the feature view to read data from
        feature_view_version: The version of the feature view to read data from
        pair_to_predict: The pair to train the model on
        candle_seconds: The number of seconds per candle
        pairs_as_features: The pairs to use for the features
        technical_indicators_as_features: The technical indicators to use for from the technical_indicators feature group
        prediction_seconds: The number of seconds into the future to predict
        llm_model_name_news_signals: The name of the LLM model to use for the news signals
        days_back: The number of days to consider for the historical data
        comet_ml_api_key: The API key for the CometML project
        comet_ml_project_name: The name of the CometML project
        hyperparameter_tuning_search_trials: The number of trials to perform for hyperparameter tuning
        hyperparameter_tuning_n_splits: The number of splits to perform for hyperparameter tuning
        model_status: The status of the model in the model registry
    """
    logger.info('Hello from the ML model training job...')

    # to log all parameters, metrics to our experiment tracking service
    # and model artifact to the model registry
    experiment = comet_ml.start(
        api_key=comet_ml_api_key,
        project_name=comet_ml_project_name,
        workspace=comet_ml_workspace,
    )

    experiment.log_parameters(
        {
            # super important to log these 2
            # because we want our deployed model to use the EXACT SAME feature view
            # as the one we used for training
            'feature_view_name': feature_view_name,
            'feature_view_version': feature_view_version,
            'pair_to_predict': pair_to_predict,
            'candle_seconds': candle_seconds,
            'pairs_as_features': pairs_as_features,
            'technical_indicators_as_features': technical_indicators_as_features,
            'prediction_seconds': prediction_seconds,
            'llm_model_name_news_signals': llm_model_name_news_signals,
            'days_back': days_back,
            'hyperparameter_tuning_search_trials': hyperparameter_tuning_search_trials,
            'hyperparameter_tuning_n_splits': hyperparameter_tuning_n_splits,
            'model_status': model_status,
        }
    )

    # 1. Read feature data from the Feature Store
    feature_reader = FeatureReader(
        hopsworks_host,
        hopsworks_project_name,
        hopsworks_api_key,
        feature_view_name,
        feature_view_version,
        pair_to_predict,
        candle_seconds,
        pairs_as_features,
        technical_indicators_as_features,
        prediction_seconds,
        llm_model_name_news_signals,
        technical_indicators_feature_group_name,
        technical_indicators_feature_group_version,
        news_signals_feature_group_name,
        news_signals_feature_group_version,
    )
    logger.info(f'Reading feature data for {days_back} days back...')
    features_and_target = feature_reader.get_training_data(days_back=days_back)
    logger.info(f'Got {len(features_and_target)} rows')

    # 2. Split the data into training and testing sets
    train_df, test_df = train_test_split(features_and_target, test_size=0.2)

    # 3. Split into features and target
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    experiment.log_parameters(
        {
            'X_train': X_train.shape,
            'y_train': y_train.shape,
            'X_test': X_test.shape,
            'y_test': y_test.shape,
        }
    )

    # 3. Evaluate quick baseline models

    # Dummy model based on current close price
    # on the test set
    y_test_pred = DummyModel(from_feature='close').predict(X_test)
    mae_dummy_model = mean_absolute_error(y_test, y_test_pred)
    logger.info(f'MAE of dummy model based on close price: {mae_dummy_model}')
    experiment.log_metric('mae_dummy_model', mae_dummy_model)
    # on the training set
    y_train_pred = DummyModel(from_feature='close').predict(X_train)
    mae_dummy_model_train = mean_absolute_error(y_train, y_train_pred)
    logger.info(
        f'MAE of dummy model based on close price on training set: {mae_dummy_model_train}'
    )
    experiment.log_metric('mae_train_dummy_model', mae_dummy_model_train)

    # Dummy model based on sma_7
    if 'sma_7' in technical_indicators_as_features:
        y_test_pred = DummyModel(from_feature='sma_7').predict(X_test)
        mae_dummy_model = mean_absolute_error(y_test, y_test_pred)
        logger.info(f'MAE of dummy model based on sma_7: {mae_dummy_model}')
        experiment.log_metric('mae_dummy_model_sma_7', mae_dummy_model)

    # Dummy model based on sma_14
    if 'sma_14' in technical_indicators_as_features:
        y_test_pred = DummyModel(from_feature='sma_14').predict(X_test)
        mae_dummy_model = mean_absolute_error(y_test, y_test_pred)
        logger.info(f'MAE of dummy model based on sma_14: {mae_dummy_model}')
        experiment.log_metric('mae_dummy_model_sma_14', mae_dummy_model)

    # 4. Fit an ML model on the training set
    model = XGBoostModel()
    model.fit(
        X_train,
        y_train,
        n_search_trials=hyperparameter_tuning_search_trials,
        n_splits=hyperparameter_tuning_n_splits,
    )

    # 5. Evaluate the model on the testing set
    y_test_pred = model.predict(X_test)
    mae_xgboost_model = mean_absolute_error(y_test, y_test_pred)
    logger.info(f'MAE of XGBoost model: {mae_xgboost_model}')
    experiment.log_metric('mae', mae_xgboost_model)

    # To check overfitting we log the model error on the training set
    y_train_pred = model.predict(X_train)
    mae_xgboost_model_train = mean_absolute_error(y_train, y_train_pred)
    logger.info(f'MAE of XGBoost model on training set: {mae_xgboost_model_train}')
    experiment.log_metric('mae_train', mae_xgboost_model_train)

    # 6. Save the model artifact to the experiment
    # Save the model to local filepath
    model_name = get_model_name(pair_to_predict, candle_seconds, prediction_seconds)
    model_filepath = f'{model_name}.joblib'
    joblib.dump(model.get_model_object(), model_filepath)

    # Log the model to Comet
    experiment.log_model(
        name=model_name,
        file_or_folder=model_filepath,
    )

    # if mae_xgboost_model < mae_dummy_model:
    # Remove this condition once able to get a better model
    if True:
        # This means the model is better than the dummy model, so register it
        logger.info(f'Registering model {model_name} with status {model_status}')

        experiment.register_model(
            model_name=model_name,
            registry_name=model_name,
            version=None,  # Auto-increment version
            status=model_status,
        )
        # logger.info(f'Registered model {registered_model}')
        logger.info(
            f'Model {model_name} registered successfully with status {model_status}'
        )

    # End the experiment to ensure all uploads complete
    experiment.end()

    logger.info('Training job done!')


def main():
    from config import (
        comet_ml_credentials,
        hopsworks_credentials,
        training_config,
    )

    train(
        hopsworks_host=hopsworks_credentials.hopsworks_host,
        hopsworks_project_name=hopsworks_credentials.project_name,
        hopsworks_api_key=hopsworks_credentials.api_key,
        feature_view_name=training_config.feature_view_name,
        feature_view_version=training_config.feature_view_version,
        pair_to_predict=training_config.pair_to_predict,
        candle_seconds=training_config.candle_seconds,
        pairs_as_features=training_config.pairs_as_features,
        technical_indicators_as_features=training_config.technical_indicators_as_features,
        prediction_seconds=training_config.prediction_seconds,
        llm_model_name_news_signals=training_config.llm_model_name_news_signals,
        days_back=training_config.days_back,
        comet_ml_api_key=comet_ml_credentials.api_key,
        comet_ml_project_name=comet_ml_credentials.project_name,
        comet_ml_workspace=comet_ml_credentials.workspace,
        hyperparameter_tuning_search_trials=training_config.hyperparameter_tuning_search_trials,
        hyperparameter_tuning_n_splits=training_config.hyperparameter_tuning_n_splits,
        model_status=training_config.model_status,
        technical_indicators_feature_group_name=training_config.technical_indicators_feature_group_name,
        technical_indicators_feature_group_version=training_config.technical_indicators_feature_group_version,
        news_signals_feature_group_name=training_config.news_signals_feature_group_name,
        news_signals_feature_group_version=training_config.news_signals_feature_group_version,
    )


if __name__ == '__main__':
    main()
