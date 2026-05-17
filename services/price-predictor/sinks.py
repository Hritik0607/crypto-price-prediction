from elasticsearch import Elasticsearch
from quixstreams.sinks.base import BatchingSink, SinkBackpressureError, SinkBatch


class ElasticSearchSink(BatchingSink):
    """
    Custom Quix Streams sink that writes data to an Elastic Search index

    Example code from Elastic Search docs
    https://elasticsearch-py.readthedocs.io/en/v8.17.0/#example-usage

    ```
    client = Elasticsearch("http://localhost:9200/", api_key="YOUR_API_KEY")

    doc = {
        "author": "kimchy",
        "text": "Elasticsearch: cool. bonsai cool.",
        "timestamp": datetime.now(),
    }
    resp = client.index(index="test-index", id=1, document=doc)
    print(resp["result"])
    ```
    """

    def __init__(
        self,
        elasticsearch_url: str,
        index_name: str,
    ):
        # call constructor of the base class to make sure the batches are initialized
        super().__init__()

        self.client = Elasticsearch(elasticsearch_url)
        self.index_name = index_name

    def write(self, batch: SinkBatch):
        # Convert batch items to list of dictionaries
        documents = [item.value for item in batch]

        try:
            # TODO: Implement a bulk indexing, instead of this hack.
            for doc in documents:
                self.client.index(
                    index=self.index_name,
                    # id=doc["id"],
                    document=doc,
                )

        except TimeoutError as e:
            raise SinkBackpressureError(
                retry_after=30.0,
                topic=batch.topic,
                partition=batch.partition,
            ) from e

        # TODO: Implement a bulk indexing. The code commented below has some bug.
        # try:
        #     # Bulk index the documents
        #     operations = [
        #         {
        #             "_index": self.index_name,
        #             "_source": doc
        #         }
        #         for doc in documents
        #     ]
        #     response = self.client.bulk(operations=operations)

        #     # Check for errors in the response
        #     if response["errors"]:
        #         # Handle any failed operations
        #         failed = [item for item in response["items"] if item["index"].get("error")]
        #         raise Exception(f"Failed to index {len(failed)} documents: {failed}")

        # except TimeoutError as e:
        #     # In case of error, tell the app to wait for 30s and retry
        #     raise SinkBackpressureError(
        #         retry_after=30.0,
        #         topic=batch.topic,
        #         partition=batch.partition,
        #     )
