from time import sleep
from typing import List, Literal

from llms.claude import ClaudeNewsSignalExtractor
from loguru import logger
from quixstreams import Application

# Module-level defaults — overridden at startup from config via main()
MAX_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 1


def add_signal_to_news(value: dict) -> dict:
    """
    From the given news in value['title'] extract the news signal using the LLM.
    """
    title = value['title']
    # source = value.get('source', '')
    # # Include source context if available so LLM can weigh credibility
    # # "Bitcoin ETF approved (Source: reuters.com)" carries more weight
    # # than the same headline from an unknown blog
    # if source:
    #     title = f"{title} (Source: {source})"
    #     logger.debug(f'Extracting news signal from {title} for source {source}')
    # else:
    #     title = title
    #     logger.debug(f'Extracting news signal from {title}')
    logger.debug(f'Extracting news signal from {title}')
    news_signal = None
    retry_delay = INITIAL_RETRY_DELAY_SECONDS

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            news_signal: List[dict] = llm.get_signal(title, output_format='list')
            if attempt > 1:
                logger.info(f'Succeeded on attempt {attempt} for: {title}')
            break  # success — exit retry loop
        except Exception as e:
            if attempt <= MAX_RETRIES:
                logger.warning(
                    f'Attempt {attempt}/{MAX_RETRIES} failed for '
                    f'"{title}": {e}. '
                    f'Retrying in {retry_delay}s...'
                )
                sleep(retry_delay)
                retry_delay *= 2  # exponential backoff: 1s, 2s, 4s
            else:
                # All retries exhausted — skip this headline
                logger.warning(
                    f'All {MAX_RETRIES} retries exhausted for '
                    f'"{title}": {e}. '
                    f'Skipping headline — no signal generated.'
                )
                return []

    # Safety check — should not happen but prevents crashes
    if not news_signal:
        logger.warning(f'Empty signal returned for: {title}')
        return []

    model_name = llm.model_name
    timestamp_ms = value['timestamp_ms']

    try:
        output = [
            {
                'coin': n['coin'],
                'signal': n['signal'],
                'model_name': model_name,
                'timestamp_ms': timestamp_ms,
            }
            for n in news_signal
        ]
    except Exception as e:
        logger.error(f'Cannot extract news signal from {news_signal}')
        logger.error(f'Error extracting news signal: {e}')
        return []

    return output


def main(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_consumer_group: str,
    llm: ClaudeNewsSignalExtractor,
    data_source: Literal['live', 'historical', 'test'],
    max_retries: int,
    initial_retry_delay_seconds: int,
):
    # Override module-level constants with config values
    global MAX_RETRIES, INITIAL_RETRY_DELAY_SECONDS
    MAX_RETRIES = max_retries
    INITIAL_RETRY_DELAY_SECONDS = initial_retry_delay_seconds

    logger.info('Hello from news-signal!')

    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
        auto_offset_reset='latest' if data_source == 'live' else 'earliest',
        consumer_extra_config={
            'max.poll.interval.ms': 900000  # 15 minutes in milliseconds
        },
    )

    input_topic = app.topic(
        name=kafka_input_topic,
        value_deserializer='json',
    )

    output_topic = app.topic(
        name=kafka_output_topic,
        value_serializer='json',
    )

    sdf = app.dataframe(input_topic)

    sdf = sdf.apply(add_signal_to_news, expand=True)

    sdf = sdf.update(lambda value: logger.debug(f'Final message: {value}'))

    sdf = sdf.to_topic(output_topic)

    app.run()


if __name__ == '__main__':
    from config import config
    from llms.factory import get_llm

    logger.info(f'Using model provider: {config.model_provider}')
    llm = get_llm(config.model_provider)

    main(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_output_topic=config.kafka_output_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        llm=llm,
        data_source=config.data_source,
        max_retries=config.llm_max_retries,
        initial_retry_delay_seconds=config.llm_initial_retry_delay_seconds,
    )
