from typing import Literal, Optional, Union

from .historical_data_source import HistoricalNewsDataSource
from .news_data_source import NewsDataSource as LiveNewsDataSource

NewsDataSource = Union[LiveNewsDataSource, HistoricalNewsDataSource]


def get_source(
    data_source: Literal['live', 'historical'],
    polling_interval_sec: Optional[int] = 10,
    url_rar_file: Optional[str] = None,
    path_to_csv_file: Optional[str] = None,
    days_back: Optional[int] = 180,
) -> NewsDataSource:
    if data_source == 'live':
        # Set up the source to download news from the CryptoPanic API
        from config import cryptopanic_config

        from .news_downloader import NewsDownloader

        # News Downloader object
        news_downloader = NewsDownloader(cryptopanic_api_key=cryptopanic_config.api_key)

        # Quix Streams data source that wraps the news downloader
        news_source = LiveNewsDataSource(
            news_downloader=news_downloader,
            polling_interval_sec=polling_interval_sec,
        )

        return news_source

    elif data_source == 'historical':
        # We read the news from a CSV file, that we previously need to download from
        # an external URL
        # https://github.com/soheilrahsaz/cryptoNewsDataset/raw/refs/heads/main/CryptoNewsDataset_csvOutput.rar
        # uncompress and wrap it as Quix Streams CSVSource
        # All this logic is implemented in a Custom Quix Streams Source
        return HistoricalNewsDataSource(
            path_to_csv_file=path_to_csv_file,
            days_back=days_back,
        )

    else:
        raise ValueError(f'Invalid data source: {data_source}')
