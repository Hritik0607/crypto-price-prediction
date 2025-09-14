from quixstreams import Application

from loguru import logger


def init_candle(trade: dict) -> dict:
    """
    Initialize a candle with the first trade
    Returns the initial candle state

    Args:
        trade (dict): The first trade

    Returns:
        dict: The initial candle state
    """
    return {
        'open': trade['price'],
        'high': trade['price'],
        'low': trade['price'],
        'close': trade['price'],
        'volume': trade['quantity'],
        # 'timestamp_ms': trade['timestamp_ms'],
        'pair': trade['product_id'],
    }


def update_candle(candle: dict, trade: dict) -> dict:
    """
    Takes the current candle (aka state) and the new trade, and updates the candle state

    Args:
        candle (dict): The current candle state
        trade (dict): The new trade

    Returns:
        dict: The updated candle state
    """
    # open price does not change, so there is no need to update it
    candle['high'] = max(candle['high'], trade['price'])
    candle['low'] = min(candle['low'], trade['price'])
    candle['close'] = trade['price']
    candle['volume'] += trade['quantity']

    return candle


def run(
        #kafka parameters
        kafka_broker_address: str,
        kafka_input_topic: str,
        kafka_output_topic: str,
        kafka_consumer_group: str,
        # candles parameters
        candle_seconds: int,
        emit_intermediate_candles: bool = True,
):
    """
    Transforms a stream of input trades into a stream of output candles.

    In 3 steps:
    - Ingests trade from the `kafka_input_topic`
    - Aggregates trades into candles
    - Produces candles to the `kafka_output_topic`

    Args:
        kafka_broker_address (str): Kafka broker address
        kafka_input_topic (str): Kafka input topic name
        kafka_output_topic (str): Kafka output topic name
        kafka_consumer_group (str): Kafka consumer group name
        candle_seconds (int): Candle duration in seconds

    Returns:
        None
    """
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group
    )

    #input topic
    trades_topic = app.topic(name=kafka_input_topic, value_serializer='json')
    #output topic
    candles_topic = app.topic(name=kafka_output_topic, value_serializer='json')

    # Step 1. Ingest trades from the input kafka topic
    # Create a Streaming DataFrame connected to the input Kafka topic
    sdf = app.dataframe(topic=trades_topic)

    # Step 2. Aggregate trades into candles
    # Print the input data
    # sdf = sdf.update(lambda message: logger.info(f"Input:  {message}"))

    # Aggregation of trades into candles using tumbling windows
    from datetime import timedelta

    sdf = (
        # Define a tumbling window of 10 minutes
        sdf.tumbling_window(timedelta(seconds=candle_seconds))
        # Create a "reduce" aggregation with "reducer" and "initializer" functions
        .reduce(reducer=update_candle, initializer=init_candle)
    )

    # we emit all intermediate candles to make the system more responsive
    sdf = sdf.current()

    # Extract open, high, low, close, volume, timestamp_ms, pair from the dataframe
    sdf['open'] = sdf['value']['open']
    sdf['high'] = sdf['value']['high']
    sdf['low'] = sdf['value']['low']
    sdf['close'] = sdf['value']['close']
    sdf['volume'] = sdf['value']['volume']
    # sdf['timestamp_ms'] = sdf['value']['timestamp_ms']
    sdf['pair'] = sdf['value']['pair']

    # Extract window start and end timestamps
    sdf['window_start_ms'] = sdf['start']
    sdf['window_end_ms'] = sdf['end']

    # keep only the relevant columns
    sdf = sdf[
        [
            'pair',
            # 'timestamp_ms',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'window_start_ms',
            'window_end_ms',
        ]
    ]

    sdf['candle_seconds'] = candle_seconds

    # logging on the console
    sdf = sdf.update(lambda value: logger.debug(f'Candle: {value}'))


    # Step 3. Produce the candles to the output kafka topic
    # Produce alerts to the output topic
    sdf = sdf.to_topic(candles_topic)

    #starts the streaming app
    app.run()


if __name__ == '__main__':
    from config import config

    run(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_output_topic=config.kafka_output_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        candle_seconds=config.candle_seconds
    )
