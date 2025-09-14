# Create an Application instance with Kafka configs
from quixstreams import Application
from  loguru import logger
from kraken_api import KrakenAPI, Trade


def run(
        kafka_broker_address: str,
        kafka_topic_name: str,
        kraken_api: KrakenAPI
):
    app = Application(
        broker_address=kafka_broker_address
    )

    # Define a topic "my_topic" with JSON serialization
    topic = app.topic(name=kafka_topic_name, value_serializer='json')

    # Create a Producer instance
    with app.get_producer() as producer:

        while True:

            events: list[Trade] = kraken_api.get_trades()
            # event = {"id": "1", "text": "Lorem ipsum dolor sit amet"}
            
            for event in events:
                # Serialize an event using the defined Topic 
                message = topic.serialize(
                    key=event.product_id, 
                    value=event.to_dict()
                    )

                # Produce a message into the Kafka topic
                producer.produce(
                    topic=topic.name, 
                    value=message.value, 
                    key=message.key
                )

                # logger.info(f'Produced to {topic.name}')
                logger.info(f'Trade {event.to_dict()} pushed to kafka')

            # breakpoint()

            # import time
            # time.sleep(1)

if __name__ == '__main__':

    from config import config

    #create object that can talk to the Karen API and get us the trade data in real time
    # api = KrakenAPI(product_ids=['BTC/EUR'])
    api = KrakenAPI(product_ids=config.product_ids)

    run(
        # kafka_broker_address= 'localhost:31234',
        # kafka_broker_address= 'kafka-e11b-kafka-bootstrap.kafka.svc.cluster.local:9092',
        kafka_broker_address= config.kafka_broker_address,
        # kafka_topic_name= 'trades',
        kafka_topic_name= config.kafka_topic_name,
        kraken_api= api
    )
