from loguru import logger
from price_predictor import PricePredictor
from quixstreams import Application
from sinks import ElasticSearchSink


def run(
    # the kafka topic that triggers the inference service
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_consumer_group: str,
    # filter only on candles with the given frequency
    candle_seconds: int,
    # encapsulate the inference logic in its .predict() method
    price_predictor: PricePredictor,
    # where to save the predictions
    elastic_search_sink: ElasticSearchSink,
):
    """
    Run the inference job as a Quix Streams application.

    Steps:
    1 - Load the model from the model registry
    2 - Generate predictions
    2 - Save predictions to Elastic Search

    Args:
        kafka_broker_address: the address of the Kafka broker
        kafka_input_topic: the topic to listen to for new data
        kafka_consumer_group: the consumer group to use
        pair_to_predict: the pair to predict
        candle_seconds: the number of seconds per candle
        prediction_seconds: the number of seconds into the future to predict
        model_status: the status of the model in the model registry
        elastic_search_sink: the sink to save the predictions to
    """
    # Quix Streams application to handles all low-level communication with Kafka
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
    )

    input_topic = app.topic(name=kafka_input_topic, value_deserializer='json')

    # Streaming Dataframe to define the business logic, aka the transformations from
    # input data to output data
    sdf = app.dataframe(input_topic)

    # We only react to candles with the given `candle_seconds` frequency
    sdf = sdf[sdf['candle_seconds'] == candle_seconds]

    # Generate a new prediction
    sdf = sdf.apply(lambda candle: price_predictor.predict(candle).to_dict())

    # logging the predictions
    sdf = sdf.update(lambda x: logger.info(x))

    # Save the predictions to Elastic Search sink
    sdf.sink(elastic_search_sink)

    app.run(sdf)


def main():
    from config import (
        comet_ml_credentials as comet_config,
    )
    from config import (
        hopsworks_credentials as hopsworks_config,
    )
    from config import (
        inference_config as config,
    )

    # Load the model from the model registry and the necessary metadata at initialization
    # It exposes a `predict` method that can be used to generate predictions
    # We initialize here so that the run() methods does not need to worry about credentials
    # and other low-level details about our PricePredictor class.
    price_predictor = PricePredictor(
        pair_to_predict=config.pair_to_predict,
        candle_seconds=config.candle_seconds,
        prediction_seconds=config.prediction_seconds,
        model_status=config.model_status,
        comet_config=comet_config,
        hopsworks_config=hopsworks_config,
    )

    # Create the Elastic Search sink
    elastic_search_sink = ElasticSearchSink(
        elasticsearch_url=config.elasticsearch_url,
        index_name=config.elasticsearch_index,
    )

    run(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        candle_seconds=config.candle_seconds,
        price_predictor=price_predictor,
        elastic_search_sink=elastic_search_sink,
    )


if __name__ == '__main__':
    main()
