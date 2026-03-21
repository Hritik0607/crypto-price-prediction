from loguru import logger
from quixstreams import Application
from sources.news_data_source import NewsDataSource


def main(
    kafka_broker_address: str,
    kafka_topic: str,
    news_source: NewsDataSource,
):
    """
    Gets news from Cryptopanic and pushes it to a Kafka topic.

    Args:
        kafka_broker_address: The address of the Kafka broker.
        kafka_topic: The topic to push the news to.
        news_source: The news source to get the news from.
    Returns:
        None
    """
    logger.info('Hello from news!')

    app = Application(broker_address=kafka_broker_address)

    # Topic where we will push the news to
    output_topic = app.topic(name=kafka_topic, value_serializer='json')

    # Create the streaming dataframe
    sdf = app.dataframe(source=news_source)

    # Print to check this thing is working
    sdf.print(metadata=True)

    # Send the final messages to the output topic
    sdf = sdf.to_topic(output_topic)

    app.run()


if __name__ == '__main__':
    from config import config
    from sources import get_source

    news_source = get_source(
        config.data_source,
        polling_interval_sec=config.polling_interval_sec,
        url_rar_file=config.historical_data_source_url_rar_file,
        path_to_csv_file=config.historical_data_source_csv_file,
        days_back=config.historical_days_back,
    )

    # Run the streaming application
    main(
        kafka_broker_address=config.kafka_broker_address,
        kafka_topic=config.kafka_topic,
        news_source=news_source,
    )
