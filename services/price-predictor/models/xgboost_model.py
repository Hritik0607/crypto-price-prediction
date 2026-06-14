from typing import Optional

import optuna
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


class XGBoostModel:
    """
    Encapsulates the training logic with or without hyperparameter tuning using an
    XGBRegressor.
    """

    def __init__(self):
        self.model = XGBRegressor(
            objective='reg:squarederror',
            eval_metric=['mae'],
            random_state=42,
        )

    def get_model_object(self):
        """
        Returns the model object.
        """
        return self.model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_search_trials: Optional[int] = 0,
        n_splits: Optional[int] = 3,
        # hyperparameter_tuning: bool = False,
    ):
        """
        Fits the an XGBoostRegressor model to the training data, either with or without
        hyperparameter tuning.

        Args:
            X (pd.DataFrame): The training data.
            y (pd.Series): The target variable.
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
        """
        if n_search_trials == 0:
            logger.info('Fitting XGBoost model without hyperparameter tuning')
            self.model = XGBRegressor()

        else:
            # TODO: Implement hyperparameter tuning
            logger.info('Fitting XGBoost model with hyperparameter tuning')

            # we do cross-validation with the number of splits specified
            # and we search for the best hyperparameters using Bayesian optimization
            best_hyperparams = self._find_best_hyperparams(
                X, y, n_search_trials=n_search_trials, n_splits=n_splits
            )
            logger.info(f'Best hyperparameters: {best_hyperparams}')

            # we train the model with the best hyperparameters
            self.model = XGBRegressor(**best_hyperparams)

        # train the model
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def _find_best_hyperparams(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_search_trials: int,
        n_splits: int,
    ) -> dict:
        """
        Finds the best hyperparameters for the model using Bayesian optimization.

        Args:
            X_train: pd.DataFrame, the training data
            y_train: pd.Series, the target variable
            n_search_trials: int, the number of trials to run
            n_splits: int, the number of splits to use for cross-validation

        Returns:
            dict, the best hyperparameters
        """

        def objective(trial: optuna.Trial) -> float:
            """
            Objective function for Optuna that returns the mean absolute error we
            want to minimize.

            Args:
                trial: optuna.Trial, the trial object

            Returns:
                float, the mean absolute error
            """
            # we ask Optuna to sample the next set of hyperparameters
            # these are our candidates for this trial
            # params = {
            #     'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            #     'max_depth': trial.suggest_int('max_depth', 3, 10),
            #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            #     'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            #     # Complete list of hyperparameters:
            #     # https://xgboost.readthedocs.io/en/stable/parameter.html
            # }

            params = {
                # Original parameters
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 0.005, 0.1, log=True
                ),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                # New regularization parameters
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                # Fixed
                'objective': 'reg:squarederror',
                'random_state': 42,
            }

            # Split X_train into n_splits folds with a time-series split
            # keeping the time-series order in each fold
            from sklearn.model_selection import TimeSeriesSplit

            tscv = TimeSeriesSplit(n_splits=n_splits)
            mae_scores = []
            for train_index, val_index in tscv.split(X_train):
                # split the data into training and validation sets
                X_train_fold, X_val_fold = (
                    X_train.iloc[train_index],
                    X_train.iloc[val_index],
                )
                y_train_fold, y_val_fold = (
                    y_train.iloc[train_index],
                    y_train.iloc[val_index],
                )

                # train the model on the training set
                model = XGBRegressor(**params)
                model.fit(X_train_fold, y_train_fold)

                # evaluate the model on the validation set
                y_pred = model.predict(X_val_fold)
                mae = mean_absolute_error(y_val_fold, y_pred)
                mae_scores.append(mae)

            # return the average MAE across all folds
            import numpy as np

            return np.mean(mae_scores)

        # we create a study object that minimizes the objective function
        study = optuna.create_study(direction='minimize')

        # we run the trials
        logger.info(f'Running {n_search_trials} trials')
        study.optimize(objective, n_trials=n_search_trials)

        # Return the best hyperparameters
        return study.best_trial.params
