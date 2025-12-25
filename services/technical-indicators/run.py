from candle import update_candles
from loguru import logger
from quixstreams import Application
from technical_indicators import compute_indicators


def main(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_consumer_group: str,
    max_candles_in_state: int,
    candle_seconds: int,
):
    """
    3 steps:
    1. Ingests candles from the kafka input topic
    2. Computes technical indicators
    3. Sends the technical indicators to the kafka output topic

    Args:
        kafka_broker_address: The address of the kafka broker
        kafka_input_topic: The topic to ingest candles from
        kafka_output_topic: The topic to send technical indicators to
        kafka_consumer_group: The consumer group for the kafka consumer
        max_candles_in_state: The maximum number of candles to keep in the state
        candle_seconds: The number of seconds per candle
    Returns:
        None
    """

    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
    )

    # Define the input and output topics of our streaming application
    input_topic = app.topic(
        name=kafka_input_topic,
        value_deserializer='json',
    )
    output_topic = app.topic(
        name=kafka_output_topic,
        value_serializer='json',
    )
    # Create a Streaming DataFrame so we can start transforming data in real time
    sdf = app.dataframe(topic=input_topic)

    # We only keep the candles with the same window size as the candle_seconds
    sdf = sdf[sdf['candle_seconds'] == candle_seconds]

    # Update the list of candles in the state
    sdf = sdf.apply(update_candles, stateful=True)

    # Compute the technical indicators from the candles in the state
    sdf = sdf.apply(compute_indicators, stateful=True)

    sdf = sdf.update(lambda value: logger.debug(f'Final message: {value}'))

    # Send the final messages to the output topic
    sdf = sdf.to_topic(output_topic)

    app.run()


if __name__ == '__main__':
    from config import config

    main(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_output_topic=config.kafka_output_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        max_candles_in_state=config.max_candles_in_state,
        candle_seconds=config.candle_seconds,
    )
