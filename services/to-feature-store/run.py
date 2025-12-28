from typing import Literal

from loguru import logger
from quixstreams import Application
from sinks import HopsworksFeatureStoreSink


def main(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_consumer_group: str,
    output_sink: HopsworksFeatureStoreSink,
    data_source: Literal['live', 'historical', 'test'],
):
    """
    2 things:
    1. Read messages from Kafka topic
    2. Push messages to Feature Store

    Args:
        kafka_broker_address: The Kafka broker address
        kafka_input_topic: The Kafka input topic
        kafka_consumer_group: The Kafka consumer group
        output_sink: The output sink
        data_source: The data source (live, historical, test)
    Returns:
        None
    """
    logger.info('Hello from to-feature-store!')

    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
        auto_offset_reset='latest' if data_source == 'live' else 'earliest',
    )
    input_topic = app.topic(kafka_input_topic, value_deserializer='json')

    sdf = app.dataframe(input_topic)

    # Do some processing here ...
    # We need to extract the features we want to push to the feature store
    # TODO: Implement
    # Sink data to a CSV file
    # sdf.sink(csv_sink)

    # Sink data to the feature store
    sdf.sink(output_sink)

    app.run()


if __name__ == '__main__':
    from config import config, hopsworks_credentials

    # Sink to save data to the feature store
    hopsworks_sink = HopsworksFeatureStoreSink(
        # Hopsworks credentials
        api_key=hopsworks_credentials.hopsworks_api_key,
        project_name=hopsworks_credentials.hopsworks_project_name,
        # Feature group configuration
        feature_group_name=config.feature_group_name,
        feature_group_version=config.feature_group_version,
        feature_group_primary_keys=config.feature_group_primary_keys,
        feature_group_event_time=config.feature_group_event_time,
        feature_group_materialization_interval_minutes=config.feature_group_materialization_interval_minutes,
    )

    main(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        output_sink=hopsworks_sink,
        data_source=config.data_source,
    )
