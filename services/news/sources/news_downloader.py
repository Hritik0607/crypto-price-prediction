from typing import List, Tuple

import requests
from loguru import logger

from .news import News


class NewsDownloader:
    """
    This class is used to download news from the Cryptopanic API.
    """

    URL = 'https://cryptopanic.com/api/developer/v2/posts/'

    def __init__(
        self,
        cryptopanic_api_key: str,
    ):
        self.cryptopanic_api_key = cryptopanic_api_key
        # logger.debug(f"Cryptopanic API key: {self.cryptopanic_api_key}")
        # self._last_published_at = None

    def get_news(self) -> List[News]:
        """
        Keeps on calling _get_batch_of_news until it gets an empty list.
        """
        news = []
        url = self.URL + '?auth_token=' + self.cryptopanic_api_key

        print('url: ', url)

        while True:
            # logger.debug(f"Fetching news from {url}")
            batch_of_news, next_url = self._get_batch_of_news(url)
            news += batch_of_news
            logger.debug(f'Fetched {len(batch_of_news)} news items')

            if not batch_of_news:
                break
            if not next_url:
                logger.debug('No next URL found, breaking')
                break

            url = next_url

        # sort the news by published_at
        news.sort(key=lambda x: x.published_at, reverse=False)

        return news

    def _get_batch_of_news(self, url: str) -> Tuple[List[News], str]:
        """
        Connects to the Cryptopanic API and fetches one batch of news

        Args:
            url: The URL to fetch the news from.

        Returns:
            A tuple containing the list of news and the next URL to fetch from.
        """
        response = requests.get(url)

        try:
            response = response.json()
        except Exception as e:
            logger.error(f'Error parsing response: {e}')
            from time import sleep

            sleep(1)
            return ([], '')

        # parse the API response into a list of News objects
        news = [
            News(
                title=post['title'],
                published_at=post['published_at'],
                # source=post['domain'],
            )
            for post in response['results']
        ]

        # extract the next URL from the API response
        next_url = response['next']

        return news, next_url


if __name__ == '__main__':
    from config import cryptopanic_config

    news_downloader = NewsDownloader(cryptopanic_api_key=cryptopanic_config.api_key)
    news = news_downloader.get_news()
    logger.debug(f'Fetched {len(news)} news items')
    breakpoint()
