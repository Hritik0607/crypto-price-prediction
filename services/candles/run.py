from datetime import timedelta
from typing import Any, List, Optional, Tuple

from loguru import logger
from quixstreams import Application
from quixstreams.models import TimestampType


def custom_ts_extractor(
    value: Any,
    headers: Optional[List[Tuple[str, bytes]]],
    timestamp: float,
    timestamp_type: TimestampType,
) -> int:
    """
    Specifying a custom timestamp extractor to use the timestamp from the message payload
    instead of Kafka timestamp.
    """
    # logger.debug(f'Custom timestamp extractor: {value}')
    return value['timestamp_ms']


def init_candle(trade: dict) -> dict:
    """
    Initialize a candle with the first trade
    """
    # breakpoint()
    return {
        'open': trade['price'],
        'high': trade['price'],
        'low': trade['price'],
        'close': trade['price'],
        'volume': trade['volume'],
        'timestamp_ms': trade['timestamp_ms'],
        'pair': trade['pair'],
    }


def update_candle(candle: dict, trade: dict) -> dict:
    """
    Update the candle with the latest trade
    """
    # breakpoint()
    candle['close'] = trade['price']
    candle['high'] = max(candle['high'], trade['price'])
    candle['low'] = min(candle['low'], trade['price'])
    candle['volume'] += trade['volume']
    candle['timestamp_ms'] = trade['timestamp_ms']
    candle['pair'] = trade['pair']
    return candle


def main(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_consumer_group: str,
    candle_seconds: int,
    emit_incomplete_candles: bool,
):
    """
    3 steps:
    1. Ingests trades from Kafka
    2. Generates candles using tumbling windows and
    3. Outputs candles to Kafka

    Args:
        kafka_broker_address (str): Kafka broker address
        kafka_input_topic (str): Kafka input topic
        kafka_output_topic (str): Kafka output topic
        kafka_consumer_group (str): Kafka consumer group
        candle_seconds (int): Candle seconds
        emit_incomplete_candles (bool): Emit incomplete candles or just the final one
        data_source (Literal['live', 'historical', 'test']): Data source
    Returns:
        None
    """
    logger.info('Starting the candles service!')

    # Initialize the Quix Streams application
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
    )

    # Define the input and output topics
    input_topic = app.topic(
        name=kafka_input_topic,
        value_deserializer='json',
        timestamp_extractor=custom_ts_extractor,
    )
    output_topic = app.topic(
        name=kafka_output_topic,
        value_serializer='json',
    )

    # Create a Streaming DataFrame from the input topic
    sdf = app.dataframe(topic=input_topic)

    # Aggregation of trades into candles using tumbling windows
    sdf = (
        # Define a tumbling window of 10 minutes
        sdf.tumbling_window(timedelta(seconds=candle_seconds))
        # Create a "reduce" aggregation with "reducer" and "initializer" functions
        .reduce(reducer=update_candle, initializer=init_candle)
    )

    if emit_incomplete_candles:
        # Emit all intermediate candles to make the system more responsive
        sdf = sdf.current()
    else:
        # Emit only the final candle
        sdf = sdf.final()

    # Extract open, high, low, close, volume, timestamp_ms, pair from the dataframe
    sdf['open'] = sdf['value']['open']
    sdf['high'] = sdf['value']['high']
    sdf['low'] = sdf['value']['low']
    sdf['close'] = sdf['value']['close']
    sdf['volume'] = sdf['value']['volume']
    sdf['timestamp_ms'] = sdf['value']['timestamp_ms']
    sdf['pair'] = sdf['value']['pair']

    # Extract window start and end timestamps
    sdf['window_start_ms'] = sdf['start']
    sdf['window_end_ms'] = sdf['end']

    # keep only the relevant columns
    sdf = sdf[
        [
            'pair',
            'timestamp_ms',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'window_start_ms',
            'window_end_ms',
        ]
    ]

    sdf = sdf.update(lambda value: logger.info(f'Candle: {value}'))
    # sdf = sdf.update(lambda value: breakpoint())

    # push the candle to the output topic
    sdf = sdf.to_topic(topic=output_topic)

    app.run()


if __name__ == '__main__':
    from config import config

    main(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_output_topic=config.kafka_output_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        candle_seconds=config.candle_seconds,
        emit_incomplete_candles=config.emit_incomplete_candles,
    )
