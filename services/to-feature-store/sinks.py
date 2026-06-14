import os

os.environ['HOPSWORKS_CERT_DIR'] = r'D:\tmp'

from datetime import datetime, timezone
from pathlib import Path

import hopsworks
import pandas as pd
from loguru import logger
from quixstreams.sinks.base import BatchingSink, SinkBackpressureError, SinkBatch


class ParquetSink(BatchingSink):
    """
    Writes feature data to local Parquet files instead of Hopsworks.

    Used when DATA_SINK=parquet in settings.env.
    This mirrors the pattern used in production at companies like Uber and Google:
      - Offline training data stored in Parquet/S3
      - Online serving data stored in Redis/Hopsworks Online Store

    Benefits over Hopsworks Offline Store:
      - No Spark cluster needed
      - Instant writes (no materialization job)
      - Works locally without cloud infrastructure
      - Same pattern as S3 + Parquet in production

    Files are written to:
      data/{feature_group_name}_v{version}.parquet
    """

    def __init__(
        self,
        feature_group_name: str,
        feature_group_version: int,
        output_dir: str = 'data',
    ):
        super().__init__()
        self.feature_group_name = feature_group_name
        self.feature_group_version = feature_group_version
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File path: data/technical_indicators_v1.parquet
        self.file_path = (
            self.output_dir / f'{feature_group_name}_v{feature_group_version}.parquet'
        )

        logger.info(f'ParquetSink initialized. ' f'Writing to: {self.file_path}')

    def write(self, batch: SinkBatch):
        """
        Appends batch data to the Parquet file.
        If file exists → appends new rows.
        If file does not exist → creates new file.
        """
        data = [item.value for item in batch]
        data = pd.DataFrame(data)

        if data.empty:
            logger.warning('Batch is empty — skipping write')
            return

        # Validate — drop NaN rows
        nan_counts = data.isna().sum()
        columns_with_nan = nan_counts[nan_counts > 0]
        if not columns_with_nan.empty:
            logger.warning(
                f'Found NaN values in {len(columns_with_nan)} columns. '
                f'Dropping affected rows.'
            )
            rows_before = len(data)
            data = data.dropna()
            logger.warning(
                f'Dropped {rows_before - len(data)} rows with NaN. '
                f'{len(data)} clean rows remaining.'
            )

        if data.empty:
            logger.warning('All rows had NaN — skipping write')
            return

        try:
            if self.file_path.exists():
                # Append to existing file
                existing = pd.read_parquet(self.file_path)
                combined = pd.concat([existing, data], ignore_index=True)
                # Remove duplicates based on primary keys
                # combined = combined.drop_duplicates(
                #     subset=['pair', 'candle_seconds', 'timestamp_ms'],
                #     keep='last'
                # )
                combined = combined.drop_duplicates(keep='last')
                combined.to_parquet(self.file_path, index=False)
                logger.info(
                    f'Appended {len(data)} rows to {self.file_path}. '
                    f'Total rows: {len(combined)}'
                )
            else:
                # Create new file
                data.to_parquet(self.file_path, index=False)
                logger.info(f'Created {self.file_path} with {len(data)} rows')
        except Exception as e:
            logger.error(f'Failed to write to Parquet: {e}')
            raise SinkBackpressureError(retry_after=5.0) from e


class HopsworksFeatureStoreSink(BatchingSink):
    """
    Some sink writing data to a database
    """

    def __init__(
        self,
        host_dns: str,
        api_key: str,
        project_name: str,
        feature_group_name: str,
        feature_group_version: int,
        feature_group_primary_keys: list[str],
        feature_group_event_time: str,
        feature_group_materialization_interval_minutes: int,
        enable_online: bool,
    ):
        """
        Establish a connection to the Hopsworks Feature Store
        """
        self.feature_group_name = feature_group_name
        self.feature_group_version = feature_group_version
        self.materialization_interval_minutes = (
            feature_group_materialization_interval_minutes
        )
        self.enable_online = enable_online

        # Establish a connection to the Hopsworks Feature Store
        project = hopsworks.login(
            host=host_dns, project=project_name, api_key_value=api_key
        )
        self._fs = project.get_feature_store()

        # Get the feature group
        self._feature_group = self._fs.get_or_create_feature_group(
            name=feature_group_name,
            version=feature_group_version,
            primary_key=feature_group_primary_keys,
            event_time=feature_group_event_time,
            # online_enabled=True,
            online_enabled=enable_online,
        )

        if enable_online:
            # set the materialization interval
            try:
                self._feature_group.materialization_job.schedule(
                    cron_expression=f'0 0/{self.materialization_interval_minutes} * ? * * *',
                    start_time=datetime.now(tz=timezone.utc),
                )
            # TODO: handle the FeatureStoreException
            except Exception as e:
                logger.error(f'Failed to schedule materialization job: {e}')

        # call constructor of the base class to make sure the batches are initialized
        super().__init__()

    def write(self, batch: SinkBatch):
        # Extract values from batch
        data = [item.value for item in batch]
        data = pd.DataFrame(data)

        logger.info(
            f'Writing batch of {len(data)} rows to feature group '
            f'"{self.feature_group_name}" v{self.feature_group_version}'
        )

        # Validate — check for empty DataFrame
        if data.empty:
            logger.warning('Batch is empty — skipping insert')
            return

        # Validate — check for NaN values
        # NaN values corrupt the Feature Store and break model training.
        # They can appear if:
        #   - TA-Lib had insufficient candle history
        #   - A field was missing from the Kafka message
        #   - An indicator computation failed silently
        nan_counts = data.isna().sum()
        columns_with_nan = nan_counts[nan_counts > 0]

        if not columns_with_nan.empty:
            logger.warning(
                f'Found NaN values in {len(columns_with_nan)} columns: '
                f'{columns_with_nan.to_dict()}. '
                f'Dropping affected rows before insert.'
            )
            rows_before = len(data)
            data = data.dropna()
            rows_after = len(data)
            logger.warning(
                f'Dropped {rows_before - rows_after} rows with NaN values. '
                f'{rows_after} clean rows remaining.'
            )

        # After dropping NaN rows, check if anything is left
        if data.empty:
            logger.warning(
                'All rows contained NaN values — skipping insert entirely. '
                'This may indicate insufficient candle history at startup. '
                'Check that MIN_CANDLES_REQUIRED is set correctly in technical_indicators.py'
            )
            return

        # Insert data into the feature group
        try:
            self._feature_group.insert(data)
            logger.info(
                f'Successfully inserted {len(data)} rows into '
                f'"{self.feature_group_name}" v{self.feature_group_version}'
            )
        except Exception as err:
            logger.error(
                f'Failed to insert {len(data)} rows into Hopsworks: {err}. '
                f'QuixStreams will retry after 30 seconds.'
            )
            raise SinkBackpressureError(
                retry_after=30.0,
            ) from err

        # try:
        #     # Try to write data to the db
        #     self._feature_group.insert(data)
        # except Exception as err:  # Capture the original exception
        #     # In case of timeout, tell the app to wait for 30s
        #     # and retry the writing later

        #     raise SinkBackpressureError(
        #         retry_after=30.0,
        #         topic=batch.topic,
        #         partition=batch.partition,
        #     ) from err  # Chain the exception
