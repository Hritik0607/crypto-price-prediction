import json
import time
from typing import List

import requests
from loguru import logger

from .base import TradesAPI
from .trade import Trade


class KrakenRestAPI(TradesAPI):
    def __init__(self, pairs: List[str], last_n_days: int):
        self.pairs = pairs
        self.last_n_days = last_n_days

        self.apis = [
            KrakenRestAPISinglePair(pair=pair, last_n_days=last_n_days)
            for pair in self.pairs
        ]

    def get_trades(self) -> List[Trade]:
        """
        Get trades for each pair, sort them by timestamp and return the trades.
        """
        trades = []
        for api in self.apis:
            if not api.is_done():
                trades += api.get_trades()

        # sort the trades by timestamp
        trades.sort(key=lambda x: x.timestamp_ms)

        # breakpoint()

        return trades

    def is_done(self) -> bool:
        """
        We are done when all the APIs are done.
        """
        for api in self.apis:
            if not api.is_done():
                return False
        return True


class KrakenRestAPISinglePair(TradesAPI):
    URL = 'https://api.kraken.com/0/public/Trades'

    def __init__(
        self,
        pair: str,
        last_n_days: int,
    ):
        self.pair = pair
        self.last_n_days = last_n_days
        self._is_done = False

        # get current timestamp in nanoseconds
        self.since_timestamp_ns = int(
            time.time_ns() - last_n_days * 24 * 60 * 60 * 1000000000
        )

        logger.info(
            f'Getting trades for pair {self.pair} for the last {self.since_timestamp_ns * 1000000000} seconds'
        )

    def get_trades(self) -> List[Trade]:
        """
        Sends a request to the Kraken API and returns the trades for the pair.
        """
        headers = {'Accept': 'application/json'}
        params = {
            'pair': self.pair,
            'since': self.since_timestamp_ns,
        }

        try:
            response = requests.request('GET', self.URL, headers=headers, params=params)
        except requests.exceptions.SSLError as e:
            logger.error(f'The Kraken API is not reachable. Error: {e}')

            # wait 10 seconds and try again
            # It would be better to make this source stateful and recoverable, so if
            # the container goes down and gets restarted by Kubernetes, it can resume
            # from where it left off.
            # TODO: reimplement this class a stateful Quix Streams data source so we don't
            # have sleep here.
            logger.error('Sleeping for 10 seconds and trying again...')
            time.sleep(10)

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse response as json: {e}')
            return []

        # TODO: check if we get a an error response. If so, implement a slow-down mechanism
        # so we don't get rate limited by Kraken.

        try:
            # Get the trades data
            trades = data['result'][self.pair]
        except KeyError as e:
            logger.error(f'Failed to get trades for pair {self.pair}: {e}')
            return []

        # convert the trades to Trade objects
        trades = [
            Trade.from_kraken_rest_api_response(
                pair=self.pair,
                price=trade[0],
                volume=trade[1],
                timestamp_sec=trade[2],
            )
            for trade in trades
        ]

        # update the since_timestamp_ns
        self.since_timestamp_ns = int(float(data['result']['last']))

        # check if we are done
        # TODO: check if this stopping conditions really work
        if self.since_timestamp_ns > int(time.time_ns() - 1000000000):
            self._is_done = True
        if self.since_timestamp_ns == 0:
            self._is_done = True

        return trades

    def is_done(self) -> bool:
        return self._is_done
