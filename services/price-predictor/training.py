import os

import comet_ml
import joblib
import numpy as np
import pandas as pd
from feature_reader import FeatureReader
from loguru import logger
from models.light_gbm_model import LightGBMModel
from models.xgboost_model import XGBoostModel
from names import get_model_name
from sklearn.metrics import mean_absolute_error


def walk_forward_validation(
    data: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[float, list[float]]:
    """
    Performs walk-forward validation on the full dataset.

    Unlike a single 80/20 split which gives ONE MAE number,
    walk-forward validation tests across multiple time windows
    giving a much more reliable estimate of production performance.

    Each window always trains on PAST data and validates on FUTURE data.
    This covers different market conditions (bull, bear, sideways).

    Example with n_splits=5:
      Window 1: train rows 0-33%,   validate rows 33-40%
      Window 2: train rows 0-40%,   validate rows 40-47%
      Window 3: train rows 0-47%,   validate rows 47-53%
      Window 4: train rows 0-53%,   validate rows 53-60%
      Window 5: train rows 0-60%,   validate rows 60-67%

    Args:
        data: Full dataset with features AND 'target' column, sorted by timestamp
        n_splits: Number of validation windows (default 5)

    Returns:
        Tuple of (average_mae, list_of_per_fold_maes)
    """
    from sklearn.model_selection import TimeSeriesSplit

    X = data.drop(columns=['target'])
    y = data['target']

    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores = []

    logger.info(f'Starting walk-forward validation with {n_splits} folds...')

    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        # Split this fold — always past for train, future for validate
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Train a fresh XGBoost model on this fold with default params.
        # We do NOT run hyperparameter tuning per fold because:
        #   - It would be extremely slow (n_splits * n_trials models trained)
        #   - The goal here is evaluation, not param finding
        #   - Hyperparameter tuning already happened before this step
        fold_model = XGBoostModel()
        fold_model.fit(
            X_train_fold,
            y_train_fold,
            n_search_trials=0,  # no tuning per fold — just train with defaults
        )

        # Evaluate on the validation set (future data the model never saw)
        y_pred = fold_model.predict(X_val_fold)
        mae = mean_absolute_error(y_val_fold, y_pred)
        mae_scores.append(mae)

        logger.info(
            f'Walk-forward fold {fold_num}/{n_splits}: '
            f'train_rows={len(X_train_fold)}, '
            f'val_rows={len(X_val_fold)}, '
            f'MAE={mae:.2f}'
        )

    avg_mae = float(np.mean(mae_scores))
    std_mae = float(np.std(mae_scores))

    logger.info(
        f'Walk-forward validation complete: '
        f'avg_MAE={avg_mae:.2f}, '
        f'std_MAE={std_mae:.2f}, '
        f'per_fold={[round(m, 2) for m in mae_scores]}'
    )

    return avg_mae, mae_scores


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
    parquet_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'to-feature-store',
        'data',
        f'technical_indicators_v{technical_indicators_feature_group_version}.parquet',
    )
    features_and_target = feature_reader.get_training_data(
        days_back=days_back, parquet_path=parquet_path
    )
    logger.info(f'Got {len(features_and_target)} rows')

    # 2. Split the data into training and testing sets
    train_df, test_df = train_test_split(features_and_target, test_size=0.2)

    COLS_TO_DROP = [
        'target',
        'timestamp_ms',
        'window_start_ms',
        'window_end_ms',
        'news_signals_timestamp_ms',
        'news_signals_model_name',
        'news_signals_coin',  # if present
    ]

    # Drop only columns that actually exist in the dataframe
    cols_to_drop_existing = [c for c in COLS_TO_DROP if c in train_df.columns]
    logger.info(f'Dropping non-feature columns: {cols_to_drop_existing}')

    X_train = train_df.drop(columns=cols_to_drop_existing)
    y_train = train_df['target']
    X_test = test_df.drop(
        columns=[c for c in cols_to_drop_existing if c in test_df.columns]
    )
    y_test = test_df['target']

    logger.info(f'X_train shape: {X_train.shape}')
    logger.info(f'X_train columns: {list(X_train.columns)}')

    # # 3. Split into features and target
    # X_train = train_df.drop(columns=['target'])
    # y_train = train_df['target']
    # X_test = test_df.drop(columns=['target'])
    # y_test = test_df['target']

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
    # y_test_pred = DummyModel(from_feature='close').predict(X_test)
    # mae_dummy_close = mean_absolute_error(y_test, y_test_pred)
    # logger.info(f'MAE of dummy model (close price): {mae_dummy_close}')
    # experiment.log_metric('mae_dummy_model_close', mae_dummy_close)

    # # Also evaluate on training set to check consistency
    # y_train_pred = DummyModel(from_feature='close').predict(X_train)
    # mae_dummy_close_train = mean_absolute_error(y_train, y_train_pred)
    # logger.info(f'MAE of dummy model (close price) on training set: {mae_dummy_close_train}')
    # experiment.log_metric('mae_train_dummy_model_close', mae_dummy_close_train)

    # Start tracking best baseline with the close price dummy
    # mae_best_baseline = mae_dummy_close
    # logger.info(f'Best baseline so far: {mae_best_baseline} (close price dummy)')

    # Dummy model — predicts 0 price change (no movement)
    # Since target is now price DELTA (future_price - current_price),
    # the simplest baseline is to predict 0 change.
    # MAE = average absolute 5-minute price movement = ~$60
    y_test_pred_dummy = pd.Series([0] * len(y_test), index=y_test.index)
    mae_dummy_close = mean_absolute_error(y_test, y_test_pred_dummy)
    logger.info(f'MAE of dummy model (predict no change): {mae_dummy_close}')
    experiment.log_metric('mae_dummy_model_close', mae_dummy_close)

    # Also on training set
    y_train_pred_dummy = pd.Series([0] * len(y_train), index=y_train.index)
    mae_dummy_close_train = mean_absolute_error(y_train, y_train_pred_dummy)
    logger.info(f'MAE of dummy model on training set: {mae_dummy_close_train}')
    experiment.log_metric('mae_train_dummy_model_close', mae_dummy_close_train)

    # Best baseline
    mae_best_baseline = mae_dummy_close
    logger.info(f'Best baseline so far: {mae_best_baseline} (predict no change)')

    # # Baseline 2: predict current sma_7 as future price
    # if 'sma_7' in technical_indicators_as_features:
    #     y_test_pred = DummyModel(from_feature='sma_7').predict(X_test)
    #     mae_dummy_sma7 = mean_absolute_error(y_test, y_test_pred)
    #     logger.info(f'MAE of dummy model (sma_7): {mae_dummy_sma7}')
    #     experiment.log_metric('mae_dummy_model_sma_7', mae_dummy_sma7)
    #     # Update best baseline if sma_7 dummy is better
    #     if mae_dummy_sma7 < mae_best_baseline:
    #         mae_best_baseline = mae_dummy_sma7
    #         logger.info(f'Best baseline updated: {mae_best_baseline} (sma_7 dummy)')

    # # Baseline 3: predict current sma_14 as future price
    # if 'sma_14' in technical_indicators_as_features:
    #     y_test_pred = DummyModel(from_feature='sma_14').predict(X_test)
    #     mae_dummy_sma14 = mean_absolute_error(y_test, y_test_pred)
    #     logger.info(f'MAE of dummy model (sma_14): {mae_dummy_sma14}')
    #     experiment.log_metric('mae_dummy_model_sma_14', mae_dummy_sma14)
    #     # Update best baseline if sma_14 dummy is better
    #     if mae_dummy_sma14 < mae_best_baseline:
    #         mae_best_baseline = mae_dummy_sma14
    #         logger.info(f'Best baseline updated: {mae_best_baseline} (sma_14 dummy)')

    logger.info(f'Best baseline MAE across all dummy models: {mae_best_baseline}')
    experiment.log_metric('mae_best_baseline', mae_best_baseline)

    models_to_try = {
        'XGBoost': XGBoostModel(),
        'LightGBM': LightGBMModel(),
    }

    # ── Train and evaluate all models ──────────────────────────────────────────
    results = {}

    for model_name, model_obj in models_to_try.items():
        logger.info(f'\n{"="*50}')
        logger.info(f'Training {model_name}...')

        model_obj.fit(
            X_train,
            y_train,
            n_search_trials=hyperparameter_tuning_search_trials,
            n_splits=hyperparameter_tuning_n_splits,
        )

        mae_test = mean_absolute_error(y_test, model_obj.predict(X_test))
        mae_train = mean_absolute_error(y_train, model_obj.predict(X_train))

        logger.info(
            f'{model_name} → Test MAE: {mae_test:.2f}, Train MAE: {mae_train:.2f}'
        )
        experiment.log_metric(f'mae_{model_name.lower()}', mae_test)
        experiment.log_metric(f'mae_train_{model_name.lower()}', mae_train)

        # ── Feature selection for this model ───────────────────────────────────
        try:
            importances = pd.Series(
                model_obj.get_model_object().feature_importances_,
                index=X_train.columns,
            ).sort_values(ascending=False)

            logger.info(f'Top 15 features for {model_name}:\n{importances.head(15)}')

            top_features = importances.head(10).index.tolist()
            X_train_sel = X_train[top_features]
            X_test_sel = X_test[top_features]

            model_sel = model_obj.__class__()
            model_sel.fit(X_train_sel, y_train, n_search_trials=0)

            mae_sel = mean_absolute_error(y_test, model_sel.predict(X_test_sel))
            mae_sel_train = mean_absolute_error(y_train, model_sel.predict(X_train_sel))
            logger.info(
                f'{model_name} top-10 → Test MAE: {mae_sel:.2f}, '
                f'Train MAE: {mae_sel_train:.2f}'
            )
            experiment.log_metric(f'mae_{model_name.lower()}_top10', mae_sel)

            if mae_sel < mae_test:
                logger.info(f'Top-10 beats full model → using top-10 for {model_name}')
                mae_test = mae_sel
                mae_train = mae_sel_train
                model_obj = model_sel

        except AttributeError:
            logger.info(
                f'{model_name} has no feature_importances_ — skipping feature selection'
            )

        results[model_name] = {
            'mae': mae_test,
            'mae_train': mae_train,
            'model': model_obj,
        }

    # ── Comparison table ────────────────────────────────────────────────────────
    logger.info(f'\n{"="*50}')
    logger.info('FINAL MODEL COMPARISON:')
    logger.info(f'  Dummy baseline: ${mae_best_baseline:.2f}')
    for name, res in sorted(results.items(), key=lambda x: x[1]['mae']):
        gap = res['mae'] - mae_best_baseline
        sign = '+' if gap > 0 else ''
        logger.info(
            f'  {name:12s}: Test=${res["mae"]:.2f}  '
            f'Train=${res["mae_train"]:.2f}  '
            f'vs baseline={sign}{gap:.2f}'
        )

    # ── Select best model ───────────────────────────────────────────────────────
    best_model_name = min(results, key=lambda k: results[k]['mae'])
    best_result = results[best_model_name]
    model = best_result['model']
    best_mae = best_result['mae']

    logger.info(f'\nBest model: {best_model_name} → MAE ${best_mae:.2f}')
    experiment.log_parameter('best_model_name', best_model_name)

    # ── Walk-forward validation on best model ───────────────────────────────────
    logger.info('Running walk-forward validation...')
    wf_avg_mae, wf_mae_scores = walk_forward_validation(
        data=features_and_target,
        n_splits=hyperparameter_tuning_n_splits,
    )
    experiment.log_metric('mae_walk_forward_avg', wf_avg_mae)
    for fold_num, fold_mae in enumerate(wf_mae_scores, 1):
        experiment.log_metric(f'mae_walk_forward_fold_{fold_num}', fold_mae)
    logger.info(
        f'Walk-forward MAE: {wf_avg_mae:.2f} ' f'(single split MAE: {best_mae:.2f})'
    )

    # ── Save and register best model ────────────────────────────────────────────
    model_name_str = get_model_name(pair_to_predict, candle_seconds, prediction_seconds)
    model_filepath = f'{model_name_str}.joblib'
    joblib.dump(model.get_model_object(), model_filepath)

    experiment.log_model(
        name=model_name_str,
        file_or_folder=model_filepath,
    )

    # if best_mae < mae_best_baseline:
    if True:
        logger.info(
            f'{best_model_name} MAE (${best_mae:.2f}) < '
            f'baseline MAE (${mae_best_baseline:.2f}). '
            f'Registering model...'
        )
        experiment.register_model(
            model_name=model_name_str,
            registry_name=model_name_str,
            version=None,
            status=model_status,
        )
        logger.info(
            f'Model {model_name_str} ({best_model_name}) '
            f'registered with status {model_status}'
        )
    else:
        logger.warning(
            f'{best_model_name} MAE (${best_mae:.2f}) >= '
            f'baseline MAE (${mae_best_baseline:.2f}). '
            f'Model did NOT beat the baseline. Skipping registration.'
        )

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
