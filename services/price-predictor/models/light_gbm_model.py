import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit


class LightGBMModel:
    def __init__(self):
        self.model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_search_trials: int = 0,
        n_splits: int = 3,
        best_params: dict = None,
    ):
        if n_search_trials > 0:
            logger.info(
                f'Fitting LightGBM with {n_search_trials} hyperparameter trials'
            )
            best_params = self._find_best_hyperparams(X, y, n_search_trials, n_splits)
        elif best_params is not None:  # ← add this block
            logger.info('Fitting LightGBM with provided best params')
        else:
            logger.info('Fitting LightGBM without hyperparameter tuning')
            best_params = {
                'n_estimators': 300,
                'max_depth': 4,
                'learning_rate': 0.01,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
            }

        clean_params = {
            k: v for k, v in best_params.items() if k not in ('random_state', 'verbose')
        }

        self.model = lgb.LGBMRegressor(
            **clean_params,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y)
        logger.info('LightGBM model fitted')

    def _find_best_hyperparams(self, X, y, n_trials, n_splits):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            tscv = TimeSeriesSplit(n_splits=n_splits)
            mae_scores = []
            for train_idx, val_idx in tscv.split(X):
                model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[val_idx])
                mae = np.mean(np.abs(y.iloc[val_idx] - preds))
                mae_scores.append(mae)
            return np.mean(mae_scores)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_model_object(self):
        return self.model
