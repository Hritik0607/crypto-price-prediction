from time import sleep
from typing import Optional

from quixstreams.sources.base import StatefulSource

from .news_downloader import NewsDownloader


class NewsDataSource(StatefulSource):
    def __init__(
        self,
        news_downloader: NewsDownloader,
        polling_interval_sec: Optional[int] = 10,
    ):
        super().__init__(name='news_data_source')
        self.news_downloader = news_downloader
        self.polling_interval_sec = polling_interval_sec

    def run(self):
        last_published_at = self.state.get('last_published_at', None)

        while self.running:
            # download news
            # the output is sorted by published_at in increasing order
            news = self.news_downloader.get_news()

            # keep only the news that was published after the last published news
            if last_published_at is not None:
                news = [
                    news_item
                    for news_item in news
                    if news_item.published_at > last_published_at
                ]

            # produce news
            for news_item in news:
                # serialize the news item as bytes
                message = self.serialize(
                    key='news',
                    value=news_item.to_dict(),
                )
                # push the serialized news item to the topic
                self.produce(
                    key=message.key,
                    value=message.value,
                )

            # update the last published news
            if news:
                last_published_at = news[-1].published_at

            # update the state
            self.state.set('last_published_at', last_published_at)

            # flush the state
            self.flush()

            # wait for the next polling
            sleep(self.polling_interval_sec)
