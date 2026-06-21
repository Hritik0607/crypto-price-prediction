from datetime import datetime, timedelta
from pathlib import Path
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

    def get_training_data(self, days_back: int, parquet_path: Optional[str] = None):
        """
        Reads training data from either:
        - Local Parquet files (no Spark needed) — used when parquet_path exists
        - Hopsworks Feature View (requires Spark) — fallback

        Parquet approach replicates the Hopsworks Feature View:
        technical_indicators JOIN news_signals ON coin
        WHERE candle_seconds = 60 AND model_name = 'dummy'
        """

        tech_path = Path(parquet_path) if parquet_path else None

        if tech_path and tech_path.exists():
            # ── Step 1: Read technical_indicators Parquet ──────────────────────────
            logger.info(f'Reading technical_indicators from Parquet: {tech_path}')
            tech_df = pd.read_parquet(tech_path)
            logger.info(f'Loaded {len(tech_df)} rows')

            # Filter candle_seconds (replicates Feature View filter)
            tech_df = tech_df[tech_df['candle_seconds'] == self.candle_seconds]
            logger.info(
                f'After candle_seconds={self.candle_seconds} filter: {len(tech_df)} rows'
            )

            # Filter to requested time range
            cutoff_ms = int(
                (datetime.now() - timedelta(days=days_back)).timestamp() * 1000
            )
            tech_df = tech_df[tech_df['timestamp_ms'] >= cutoff_ms]
            logger.info(f'After {days_back} days filter: {len(tech_df)} rows')

            # Add window_end_ms if missing
            if 'window_end_ms' not in tech_df.columns:
                tech_df['window_end_ms'] = tech_df['timestamp_ms']

            # ── Step 2: Read news_signals Parquet ─────────────────────────────────
            news_path = tech_path.parent / 'news_signals_v1.parquet'

            if news_path.exists():
                logger.info(f'Reading news_signals from Parquet: {news_path}')
                news_df = pd.read_parquet(news_path)

                # Filter model_name (replicates Feature View filter)
                news_df = news_df[
                    news_df['model_name'] == self.llm_model_name_news_signals
                ]
                logger.info(
                    f'news_signals after model_name filter: {len(news_df)} rows'
                )

                # Rename news columns with prefix 'news_signals_'
                # Replicates: prefix='news_signals_' in Hopsworks join
                news_df = news_df.rename(
                    columns={
                        col: f'news_signals_{col}'
                        for col in news_df.columns
                        if col != 'coin'
                    }
                )

                # ── Step 3: JOIN (replicates Hopsworks Feature View point-in-time join)
                # Hopsworks Feature View uses event_time for point-in-time lookup
                # internally even though the query says ON=['coin']
                # merge_asof replicates this: for each tech_indicator row at time T,
                # find the nearest news_signal for same coin at time <= T
                tech_df = tech_df.sort_values('timestamp_ms')
                news_df = news_df.sort_values('news_signals_timestamp_ms')

                raw_features = pd.merge_asof(
                    tech_df,
                    news_df,
                    left_on='timestamp_ms',
                    right_on='news_signals_timestamp_ms',
                    by='coin',
                    direction='nearest',
                )
                logger.info(f'After JOIN: {len(raw_features)} rows')

                # Fill any nulls from left join misses
                raw_features['news_signals_signal'] = raw_features[
                    'news_signals_signal'
                ].fillna(0)

            else:
                logger.warning(
                    f'news_signals Parquet not found at {news_path} — '
                    f'adding news_signals_signal=0 for all rows'
                )
                tech_df['news_signals_signal'] = 0
                raw_features = tech_df

            logger.info(f'Final training data: {len(raw_features)} rows')
            logger.info(f'\n{raw_features.head(3)}')

        else:
            # get raw features from the Feature Store
            logger.info(f'Getting training data going back {days_back} days')
            raw_features = self._feature_view.get_batch_data(
                start_time=datetime.now() - timedelta(days=days_back),
                end_time=datetime.now(),
            )

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

            df_target = df_all[['window_end_ms', 'close']].copy()
            df_target['window_end_ms'] = (
                df_target['window_end_ms'] - self.prediction_seconds * 1000
            )
            df_all = df_all.reset_index(drop=True)
            df_all = df_all.merge(
                df_target, on='window_end_ms', how='left', suffixes=('', '_target')
            )
            df_all = df_all[df_all['close_target'].notna()]

            # Target = price change not absolute price
            # Model learns: "given these indicators, price will move by X dollars"
            df_all['target'] = df_all['close_target'] - df_all['close']
            df_all.drop(columns=['close_target'], inplace=True)

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

    def get_inference_features(self, fresh_candle: Optional[dict] = None):
        """
        Get the latest features for inference.

        If a fresh_candle is provided (from the technical_indicators Kafka topic),
        its features are used directly for pair_to_predict instead of reading
        from the Online Store (which can be up to 15 minutes stale).

        For other pairs (e.g. ETH/USD) and news signals, the Online Store
        is still used as a fallback regardless.

        Args:
            fresh_candle: Optional dict from the technical_indicators topic.
                        Contains all 22 technical indicators for one pair.
                        If None, reads everything from Online Store.
        """
        if (
            fresh_candle is not None
            and fresh_candle.get('pair') == self.pair_to_predict
        ):
            logger.info(
                f'Using FRESH candle features for {self.pair_to_predict} '
                f'(window_end_ms={fresh_candle.get("window_end_ms")})'
            )
            return self._get_features_with_fresh_candle(fresh_candle)

        # Fallback: read everything from Online Store
        logger.info('No fresh candle provided — reading all features from Online Store')
        return self._get_features_from_online_store()

    def _get_features_from_online_store(self) -> pd.DataFrame:
        """
        Original behaviour: read all features from the Hopsworks Online Store.
        This is the fallback when no fresh candle is available.
        """
        logger.info('Getting latest features from the online feature store')
        keys_to_read = self._get_online_store_keys()
        logger.info(f'Keys to read: {keys_to_read}')
        raw_features = self._feature_view.get_feature_vectors(
            entry=keys_to_read,
            return_type='pandas',
        )
        features = self._preprocess_raw_features_into_features_and_target(
            raw_features,
            add_target_column=False,
        )
        return features

    def _get_features_with_fresh_candle(self, fresh_candle: dict) -> pd.DataFrame:
        """
        Build inference features using:
        - FRESH technical indicators for pair_to_predict (from Kafka message)
        - Online Store for other pairs (e.g. ETH/USD) and news signals

        Why we still need Online Store here:
        - fresh_candle only has indicators for ONE pair (pair_to_predict)
        - news_signals_signal is from a separate pipeline, not in candle
        - other pairs_as_features (ETH/USD) need their own indicators

        Args:
            fresh_candle: Dict from the technical_indicators topic with all
                        22 technical indicators + OHLCV for pair_to_predict.
        """
        # ── Step 1: Get Online Store features ─────────────────────────────────
        # We still need Online Store for:
        #   - news_signals_signal for pair_to_predict
        #   - all features for other pairs (ETH/USD etc.)
        logger.info('Getting Online Store features for other pairs and news signals')
        keys_to_read = self._get_online_store_keys()
        raw_features_online = self._feature_view.get_feature_vectors(
            entry=keys_to_read,
            return_type='pandas',
        )

        # ── Step 2: Extract news signal from Online Store ──────────────────────
        # The technical_indicators message does NOT have news_signals_signal.
        # It comes from a completely separate pipeline (news → news-signal → feature store).
        # We get it from Online Store and inject it into our fresh candle row.
        news_signal_value = 0  # safe default: neutral signal
        if (
            not raw_features_online.empty
            and 'news_signals_signal' in raw_features_online.columns
        ):
            pair_online_row = raw_features_online[
                raw_features_online['pair'] == self.pair_to_predict
            ]
            if not pair_online_row.empty:
                news_signal_value = pair_online_row['news_signals_signal'].iloc[0]
                logger.info(
                    f'Got news_signals_signal={news_signal_value} '
                    f'for {self.pair_to_predict} from Online Store'
                )
            else:
                logger.warning(
                    f'No Online Store row found for {self.pair_to_predict}. '
                    f'Defaulting news_signals_signal=0'
                )
        else:
            logger.warning(
                'Online Store returned empty or no news_signals_signal column. '
                'Defaulting news_signals_signal=0'
            )

        # ── Step 3: Build fresh row for pair_to_predict ───────────────────────
        # Convert the fresh candle dict to a single-row DataFrame.
        # Add news_signals_signal from Online Store (extracted above).
        fresh_row = pd.DataFrame([fresh_candle])
        fresh_row['news_signals_signal'] = news_signal_value

        # Ensure window_end_ms exists (some messages use timestamp_ms instead)
        if (
            'window_end_ms' not in fresh_row.columns
            and 'timestamp_ms' in fresh_row.columns
        ):
            fresh_row['window_end_ms'] = fresh_row['timestamp_ms']

        logger.info(
            f'Built fresh row for {self.pair_to_predict}: '
            f'window_end_ms={fresh_candle.get("window_end_ms")}, '
            f'close={fresh_candle.get("close")}, '
            f'rsi_14={fresh_candle.get("rsi_14")}'
        )

        # ── Step 4: Combine fresh row with other pairs from Online Store ───────
        # Keep rows for pairs OTHER than pair_to_predict (e.g. ETH/USD).
        # These are still stale from Online Store — acceptable trade-off.
        other_pairs_online = raw_features_online[
            raw_features_online['pair'] != self.pair_to_predict
        ]

        if not other_pairs_online.empty:
            combined = pd.concat([fresh_row, other_pairs_online], ignore_index=True)
            logger.info(
                f'Combined fresh {self.pair_to_predict} row with '
                f'{len(other_pairs_online)} other pair rows from Online Store'
            )
        else:
            combined = fresh_row
            logger.info(
                f'Only one pair ({self.pair_to_predict}) — using fresh row only'
            )

        # ── Step 5: Preprocess into model-ready format ─────────────────────────
        features = self._preprocess_raw_features_into_features_and_target(
            combined,
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
