from datetime import datetime, timedelta
from typing import Optional

import hopsworks
import pandas as pd
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore
from hsfs.feature_view import FeatureView
from loguru import logger


class FeatureReader:
    """
    Reads features from our 2 features groups
    - technical_indicators
    - news_signals
    and preprocess it so that it has the format (features, target) we need for
    training and for inference.
    """

    def __init__(
        self,
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
        # Optional. Only required if the feature view above does not exist and needs
        # to be created
        technical_indicators_feature_group_name: Optional[str] = None,
        technical_indicators_feature_group_version: Optional[int] = None,
        news_signals_feature_group_name: Optional[str] = None,
        news_signals_feature_group_version: Optional[int] = None,
    ):
        """ """
        self.pair_to_predict = pair_to_predict
        self.candle_seconds = candle_seconds
        self.pairs_as_features = pairs_as_features
        self.technical_indicators_as_features = technical_indicators_as_features
        self.prediction_seconds = prediction_seconds
        self.llm_model_name_news_signals = llm_model_name_news_signals

        # connect to the Hopsworks Feature Store
        self._feature_store = self._get_feature_store(
            hopsworks_host,
            hopsworks_project_name,
            hopsworks_api_key,
        )

        if technical_indicators_feature_group_name is not None:
            logger.info(
                f'Attempt to create the feature view {feature_view_name}-{feature_view_version}'
            )
            self._feature_view = self._create_feature_view(
                feature_view_name,
                feature_view_version,
                technical_indicators_feature_group_name,
                technical_indicators_feature_group_version,
                news_signals_feature_group_name,
                news_signals_feature_group_version,
            )
        else:
            self._feature_view = self._get_feature_view(
                feature_view_name,
                feature_view_version,
            )

    def _get_feature_group(self, name: str, version: int) -> FeatureGroup:
        """
        Returns a feature group object given its name and version.
        """
        return self._feature_store.get_feature_group(
            name=name,
            version=version,
        )

    def _create_feature_view(
        self,
        feature_view_name: str,
        feature_view_version: int,
        technical_indicators_feature_group_name,
        technical_indicators_feature_group_version,
        news_signals_feature_group_name,
        news_signals_feature_group_version,
    ) -> FeatureView:
        """
        Creates a feature view by joining the technical_indicators and news_signals
        feature groups.

        Args:
            feature_view_name: The name of the feature view to create.
            feature_view_version: The version of the feature view to create.
            technical_indicators_feature_group_name: The name of the technical_indicators
                feature group.
            technical_indicators_feature_group_version: The version of the technical_indicators
                feature group.
            news_signals_feature_group_name: The name of the news_signals feature group.
            news_signals_feature_group_version: The version of the news_signals feature group.

        Returns:
            The feature view object.
        """

        # we get the 2 features groups we need to join
        technical_indicators_fg = self._get_feature_group(
            technical_indicators_feature_group_name,
            technical_indicators_feature_group_version,
        )
        # print("Technical FG:", len(technical_indicators_fg.read()))

        news_signals_fg = self._get_feature_group(
            news_signals_feature_group_name,
            news_signals_feature_group_version,
        )
        # print("News Signals FG:", len(news_signals_fg.read()))

        # Debug: confirm actual column names so join keys are correct
        logger.info(
            'Technical indicators FG columns: '
            + str([f.name for f in technical_indicators_fg.features])
        )
        logger.info(
            'News signals FG columns: '
            + str([f.name for f in news_signals_fg.features])
        )

        # # Query in 3-steps
        # # Step 1. Filter rows from news_signals_fg for the model_name we need and drop the model_name column
        # news_signal_query = news_signals_fg \
        #     .select_all() \
        #     .filter(news_signals_fg.model_name == self.llm_model_name_news_signals)
        # # query = news_signal_query

        # # # Step 2. Filter rows from technical_indicators_fg for the candle_seconds we need
        # technical_indicators_query = technical_indicators_fg \
        #     .select_all() \
        #     .filter(technical_indicators_fg.candle_seconds == self.candle_seconds) \

        # # Step 3. Join the 2 queries on the `coin` column
        # query = technical_indicators_query.join(
        #     news_signal_query,
        #     on=["coin"],
        #     join_type="left",
        #     prefix='news_signals_',
        # )

        # Attempt to create the feature view in one query
        query = (
            technical_indicators_fg.select_all()
            .join(
                news_signals_fg.select_all(),
                on=['coin'],
                join_type='left',
                prefix='news_signals_',
            )
            .filter(
                (technical_indicators_fg.candle_seconds == self.candle_seconds)
                & (news_signals_fg.model_name == self.llm_model_name_news_signals)
            )
        )

        # tech_query = technical_indicators_fg.select_all().filter(
        #     technical_indicators_fg.candle_seconds == self.candle_seconds
        # )
        # news_query = news_signals_fg.select_all().filter(
        #     news_signals_fg.model_name == self.llm_model_name_news_signals
        # )

        # query = tech_query.join(
        #     news_query,
        #     on=['coin'],            # <-- THE KEY FIX: coin==coin, not pair==coin
        #     join_type='left',
        #     prefix='news_signals_',
        # )

        # attempt to create the feature view
        feature_view = self._feature_store.create_feature_view(
            name=feature_view_name,
            version=feature_view_version,
            query=query,
            # This seemingly innocent flag makes this crash
            # logging_enabled=True,
        )
        logger.info(f'Feature view {feature_view_name}-{feature_view_version} created')

        return feature_view

    def _get_feature_view(
        self, feature_view_name: str, feature_view_version: int
    ) -> FeatureView:
        """
        Returns a feature view object given its name and version.
        """
        # raise NotImplementedError('Feature view creation is not supported yet')
        return self._feature_store.get_feature_view(
            name=feature_view_name,
            version=feature_view_version,
        )

    def _get_feature_store(
        self, host_dns, project_name: str, api_key: str
    ) -> FeatureStore:
        """
        Returns a feature store object.
        """
        logger.info('Getting feature store')
        project = hopsworks.login(
            host=host_dns, project=project_name, api_key_value=api_key
        )
        fs = project.get_feature_store()
        return fs

    def get_training_data(self, days_back: int):
        """
        Use the self._feature_view to get the training data going back `days_back` days.
        """
        # get raw features from the Feature Store
        logger.info(f'Getting training data going back {days_back} days')
        raw_features = self._feature_view.get_batch_data(
            start_time=datetime.now() - timedelta(days=days_back),
            end_time=datetime.now(),
        )

        print('RAW FEATURES:', len(raw_features))
        print(raw_features.head())

        # horizontally stack the features for each pair
        # we want the output to be a daframe with (features, target)
        features = self._preprocess_raw_features_into_features_and_target(
            raw_features,
            add_target_column=True,
        )

        return features

    def _preprocess_raw_features_into_features_and_target(
        self,
        data: pd.DataFrame,
        add_target_column: bool,
    ) -> pd.DataFrame:
        """
        Preprocess the features into features and possibly targets.
        Horizontally stack the features for each pair, matching the timestamps.

        Args:
            data: The raw features from the Feature Store.
            add_target_column: Whether to add the target column to the features.

        Returns:
            The features and possibly targets.

        TODO: Join using the `window_end_ms` columns but keep the original `timestamp_ms`
        column.
        """
        # append self.pair_to_predict to the list of pairs
        if self.pair_to_predict != self.pairs_as_features[0]:
            # TODO: move this validation to the config.py
            raise ValueError(
                f'Pair {self.pair_to_predict} not found as the first feature in pairs_as_features'
            )

        # Horizontally stack the features for each pair
        df_all = None
        for pair in self.pairs_as_features:
            logger.info(f'Horizontally stacking features for pair {pair}')

            # filer rows for this pair
            df = data[data['pair'] == pair]

            # keep only the columns we need
            df = df[
                ['pair', 'window_end_ms', 'open', 'close']
                + self.technical_indicators_as_features
                + ['news_signals_signal']
            ]

            # rename the window_end_ms column to timestamp_ms
            # df.rename(columns={'window_end_ms': 'timestamp_ms'}, inplace=True)

            if df_all is not None:
                # if we already have a df, we need to match the timestamps
                # left join between df_all and df on the timestamp column
                df_all = df_all.merge(
                    df,
                    on='window_end_ms',
                    how='left',
                    suffixes=('', f'_{pair}'),
                )
            else:
                df_all = df

        if add_target_column:
            logger.info('Adding target column to the dataset')

            # extract the timestamp_ms and close columns where the targets are
            df_target = df_all[['window_end_ms', 'close']]

            # move the timestamp_ms column to the prediction_seconds seconds from now
            # so we can join on it
            df_target['window_end_ms'] = (
                df_target['window_end_ms'] - self.prediction_seconds * 1000
            )

            # left join on the timestamp_ms columns
            df_all = df_all.merge(
                df_target,
                on='window_end_ms',
                how='left',
                suffixes=('', '_target'),
            )

            # drop rows for which the column `close_target` is NaN
            df_all = df_all[df_all['close_target'].notna()]

            # rename the close_target column to target
            df_all.rename(columns={'close_target': 'target'}, inplace=True)

        # rename the window_end_ms column to timestamp_ms and sort by it
        df_all.rename(columns={'window_end_ms': 'timestamp_ms'}, inplace=True)
        df_all.sort_values(by='timestamp_ms', inplace=True)

        # drop the pair_{pair} columns
        # These are categorical features and we don't need for the model
        df_all.drop(
            columns=[col for col in df_all.columns if col.startswith('pair')],
            inplace=True,
        )

        return df_all

    def get_inference_features(self):
        """
        Get the latest features from the ONLINE feature store.
        """
        # get raw features from the Feature Store
        logger.info('Getting latest features from the online feature store')
        keys_to_read = self._get_online_store_keys()
        logger.info(f'Keys to read: {keys_to_read}')
        raw_features = self._feature_view.get_feature_vectors(
            entry=keys_to_read,
            return_type='pandas',
        )

        # horizontally stack the features for each pair
        # we want the outpu to be a daframe with (features, target)
        features = self._preprocess_raw_features_into_features_and_target(
            raw_features,
            add_target_column=False,
        )

        return features

    def _get_online_store_keys(self) -> list[dict]:
        """
        Get the keys we need to get features from the online store.

        Serving keys:
        - pair
        - candle_seconds
        - news_signals_coin -> TODO: remove this as a serving key. For that, update the query that behind this feature view.

        Returns:
            A list of dictionaries with the keys we need to get features from the online store.
        """
        # breakpoint()
        keys = []
        for pair in self.pairs_as_features:
            for candle_seconds in [self.candle_seconds]:
                keys.append(
                    {
                        'pair': pair,
                        'candle_seconds': candle_seconds,
                        # TODO: remove this hack once the query is updated
                        'news_signals_coin': pair.split('/')[0],
                    }
                )
        return keys


if __name__ == '__main__':
    pass

    # feature_reader = FeatureReader(
    #     hopsworks_host=hopsworks_credentials.hopsworks_host,
    #     hopsworks_project_name=hopsworks_credentials.project_name,
    #     hopsworks_api_key=hopsworks_credentials.api_key,
    #     feature_view_name='price_predictor',
    #     feature_view_version=1,
    #     pair_to_predict='BTC/USD',
    #     candle_seconds=60,
    #     pairs_as_features=['BTC/USD', 'ETH/USD'],
    #     technical_indicators_as_features=[
    #         'rsi_9',
    #         'rsi_14',
    #         'rsi_21',
    #         'macd',
    #         'macd_signal',
    #         'macd_hist',
    #         'bbands_upper',
    #         'bbands_middle',
    #         'bbands_lower',
    #         'stochrsi_fastk',
    #         'stochrsi_fastd',
    #         'adx',
    #         'volume_ema',
    #         'ichimoku_conv',
    #         'ichimoku_base',
    #         'ichimoku_span_a',
    #         'ichimoku_span_b',
    #         'mfi',
    #         'atr',
    #         'price_roc',
    #         'sma_7',
    #         'sma_14',
    #         'sma_21',
    #     ],
    #     prediction_seconds=60 * 5,
    #     llm_model_name_news_signals='dummy',
    #     # Optional. Only required if the feature view above does not exist and needs
    #     # to be created
    #     technical_indicators_feature_group_name='technical_indicators',
    #     technical_indicators_feature_group_version=1,
    #     news_signals_feature_group_name='news_signals',
    #     news_signals_feature_group_version=1,
    # )

    # feature_reader = FeatureReader(
    #     hopsworks_host=hopsworks_credentials.hopsworks_host,
    #     hopsworks_project_name=hopsworks_credentials.project_name,
    #     hopsworks_api_key=hopsworks_credentials.api_key,
    #     feature_view_name=training_config.feature_view_name,
    #     feature_view_version=training_config.feature_view_version,
    #     pair_to_predict=training_config.pair_to_predict,
    #     candle_seconds=training_config.candle_seconds,
    #     pairs_as_features=training_config.pairs_as_features,
    #     technical_indicators_as_features=training_config.technical_indicators_as_features,
    #     prediction_seconds=60 * 5,
    #     llm_model_name_news_signals='dummy',
    #     # Optional. Only required if the feature view above does not exist and needs
    #     # to be created
    #     technical_indicators_feature_group_name='technical_indicators',
    #     technical_indicators_feature_group_version=1,
    #     news_signals_feature_group_name='news_signals',
    #     news_signals_feature_group_version=1,
    # )

    # training_data = feature_reader.get_training_data(days_back=90)
    # print(training_data)
    # breakpoint()

    # latest_features = feature_reader.get_inference_features()
    # print(latest_features)
